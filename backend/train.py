import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import cv2


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = None
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                # Keep non-float buffers/params as-is
                if k not in self.shadow:
                    self.shadow[k] = v.clone().detach()
                else:
                    self.shadow[k] = v.clone().detach()
                continue
            if k in self.shadow and torch.is_floating_point(self.shadow[k]):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone().detach()
    def apply_to(self, model: torch.nn.Module):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
    def restore(self, model: torch.nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_np_to_tensor(img_np: np.ndarray, size: int, train: bool) -> torch.Tensor:
    # basic augmentations
    if train:
        if np.random.rand() < 0.5:
            img_np = np.ascontiguousarray(np.flip(img_np, axis=1))
        if np.random.rand() < 0.3:
            # simple brightness/contrast
            alpha = 0.9 + 0.2*np.random.rand()
            beta = np.random.uniform(-10, 10)
            img_np = np.clip(alpha*img_np + beta, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.25:
            # light jpeg compression
            ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(60, 90)])
            if ok:
                img_np = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if np.random.rand() < 0.25:
            # gaussian noise
            noise = np.random.normal(0, 8, img_np.shape)
            img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # resize with padding to square
    h, w = img_np.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_np, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    x = torch.from_numpy(canvas).float().permute(2,0,1) / 255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    x = (x - mean) / std
    return x


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, folder, size=300, train=True):
        self.ds = datasets.ImageFolder(folder)
        self.size = size
        self.train = train
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        img = self.ds.loader(path).convert('RGB')
        img_np = np.array(img)
        x = preprocess_np_to_tensor(img_np, self.size, train=self.train)
        return x, label


def build_model(num_classes=2):
    try:
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    except Exception:
        m = models.efficientnet_b3(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feat, num_classes)
    return m


# Albumentations removed; using preprocess_np_to_tensor instead.


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam


def mixup_criterion(criterion, pred, targets):
    y_a, y_b = targets
    return (criterion(pred, y_a) + criterion(pred, y_b)) * 0.5


def focal_ce_loss(logits, targets, alpha=0.5, gamma=2.0, label_smoothing=0.0):
    ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1-1e-6)
    loss = (alpha * (1 - pt) ** gamma) * ce
    return loss.mean()


def train_one_epoch(model, loader, optimizer, scaler, device, label_smoothing=0.1, mixup_alpha=0.4, grad_clip=1.0, use_focal=False):
    model.train()
    total_loss = 0
    base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    pbar = tqdm(loader, desc='train', leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            if mixup_alpha > 0:
                x_m, y_m, lam = mixup_data(x, y, mixup_alpha)
                logits = model(x_m)
                if use_focal:
                    loss_a = focal_ce_loss(logits, y_m[0], alpha=0.5, gamma=2.0, label_smoothing=label_smoothing)
                    loss_b = focal_ce_loss(logits, y_m[1], alpha=0.5, gamma=2.0, label_smoothing=label_smoothing)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = lam * base_criterion(logits, y_m[0]) + (1 - lam) * base_criterion(logits, y_m[1])
            else:
                logits = model(x)
                if use_focal:
                    loss = focal_ce_loss(logits, y, alpha=0.5, gamma=2.0, label_smoothing=label_smoothing)
                else:
                    loss = base_criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, ema: EMA = None):
    if ema is not None:
        ema.apply_to(model)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='valid', leave=False):
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            ys.extend(y.numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
    if ema is not None:
        ema.restore(model)
    acc = accuracy_score(ys, ps)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, ps, average='binary', zero_division=0)
    return acc, prec, rec, f1


def find_best_threshold(model, loader, device, metric: str = 'accuracy'):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='calibrate', leave=False):
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:,1]
            probs.extend(p.cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    import numpy as np
    ys = np.array(ys)
    probs = np.array(probs)
    best_t, best_score = 0.5, -1
    for t in np.linspace(0.2, 0.8, 121):
        preds = (probs >= t).astype(int)
        tp = ((preds==1) & (ys==1)).sum()
        fp = ((preds==1) & (ys==0)).sum()
        fn = ((preds==0) & (ys==1)).sum()
        tn = ((preds==0) & (ys==0)).sum()
        acc = (tp+tn)/max(1,len(ys))
        tpr = tp/max(1,tp+fn)
        tnr = tn/max(1,tn+fp)
        prec_fake = tp/max(1,tp+fp)
        rec_fake = tpr
        if metric=='accuracy':
            score = acc
        elif metric=='f1_fake':
            if prec_fake+rec_fake==0: score=0
            else: score = 2*prec_fake*rec_fake/(prec_fake+rec_fake)
        elif metric=='recall_fake':
            score = rec_fake
        elif metric=='balanced':
            score = (tpr + tnr) / 2.0
        else:
            score = acc
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, float(best_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path containing train/ and val/')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'models'))
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--calibrate_only', action='store_true', help='Skip training; just compute threshold on val and save')
    parser.add_argument('--calib_metric', type=str, default='accuracy', choices=['accuracy','f1_fake','recall_fake','balanced'])
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    train_ds = SimpleDataset(train_dir, size=args.img_size, train=True)
    val_ds = SimpleDataset(val_dir, size=args.img_size, train=False)

    # If calibrate only, load best.pt and compute threshold, then exit
    if args.calibrate_only:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = build_model().to(device)
        best_path = os.path.join(args.model_dir, 'best.pt')
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)
        model.eval()
        best_t, best_score = find_best_threshold(model, val_loader, device, metric=args.calib_metric)
        with open(os.path.join(args.model_dir, 'threshold.txt'), 'w') as f:
            f.write(str(best_t))
        print(f"Saved calibrated threshold {best_t:.3f} ({args.calib_metric}={best_score:.4f}) to {os.path.join(args.model_dir, 'threshold.txt')}")

    # Balanced sampling if classes imbalanced
    class_counts = np.bincount([y for _, y in train_ds.ds.samples])
    if len(class_counts) == 2 and min(class_counts) > 0:
        weights = 1.0 / class_counts
        sample_weights = [weights[y] for _, y in train_ds.ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)

    # Freeze backbone for warmup epochs
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    # EMA of weights
    ema = EMA(model, decay=0.997)

    # Cosine schedule with warmup
    total_epochs = args.epochs
    warmup_epochs = max(2, int(0.1 * total_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))

    best_f1 = -1
    patience = args.patience
    no_improve = 0
    best_path = os.path.join(args.model_dir, 'best.pt')

    for epoch in range(1, args.epochs+1):
        # Unfreeze after warmup
        if epoch == warmup_epochs + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - epoch + 1))

        use_focal = epoch > warmup_epochs + 2
        train_loss = 0.0
        model.train()
        # train_one_epoch with EMA updates per batch
        model.train()
        total_loss = 0
        base_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        pbar = tqdm(train_loader, desc='train', leave=False)
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                if 0.2 > 0:
                    x_m, y_m, lam = mixup_data(x, y, 0.2)
                    logits = model(x_m)
                    if use_focal:
                        loss_a = focal_ce_loss(logits, y_m[0], alpha=0.5, gamma=2.0, label_smoothing=0.1)
                        loss_b = focal_ce_loss(logits, y_m[1], alpha=0.5, gamma=2.0, label_smoothing=0.1)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = lam * base_criterion(logits, y_m[0]) + (1 - lam) * base_criterion(logits, y_m[1])
                else:
                    logits = model(x)
                    loss = base_criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scaler.update()
            total_loss += loss.item() * x.size(0)
            # EMA update
            ema.update(model)
            pbar.set_postfix(loss=loss.item())
        train_loss = total_loss / len(train_loader.dataset)

        acc, prec, rec, f1 = evaluate(model, val_loader, device, ema=ema)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={acc:.4f} val_prec={prec:.4f} val_rec={rec:.4f} val_f1={f1:.4f}")

        # Step scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'f1': f1}, best_path)
            print(f"Saved best checkpoint to {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # After training: load best, calibrate threshold on val, and save
    try:
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        best_t, best_acc = find_best_threshold(model, val_loader, device)
        with open(os.path.join(args.model_dir, 'threshold.txt'), 'w') as f:
            f.write(str(best_t))
        print(f"Saved calibrated threshold {best_t:.3f} (val_acc={best_acc:.4f}) to {os.path.join(args.model_dir, 'threshold.txt')}")
    except Exception as e:
        print('Threshold calibration skipped due to error:', e)

    print("Training complete. Best F1:", best_f1)


if __name__ == '__main__':
    main()
