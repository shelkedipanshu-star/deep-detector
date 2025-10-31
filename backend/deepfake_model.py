import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None
from torchvision import models
from typing import List, Tuple

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None


class DeepfakeModel:
    def __init__(self, model_dir: str, img_size: int = 224, device: str = None):
        self.model_dir = model_dir
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(os.environ.get('DIP_THRESHOLD', '0.6'))  # tuned threshold
        self.invert_logits = os.environ.get('DIP_INVERT', '0') == '1'  # invert class mapping if needed
        # Build ensemble
        self.model_efficientnet, self.model_resnet = self._build_models()
        # Load calibrated threshold if present
        th_file = os.path.join(self.model_dir, 'threshold.txt')
        try:
            if os.path.isfile(th_file):
                with open(th_file, 'r') as f:
                    self.threshold = float(f.read().strip())
        except Exception:
            pass
        self.use_albu = A is not None
        self.transforms = self._build_transforms(img_size) if self.use_albu else None
        # Face detector (optional)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None

    def _build_models(self):
        # EfficientNet-B3
        try:
            me = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        except Exception:
            me = models.efficientnet_b3(weights=None)
        me.classifier[1] = nn.Linear(me.classifier[1].in_features, 2)
        # ResNet50
        try:
            mr = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            mr = models.resnet50(weights=None)
        mr.fc = nn.Linear(mr.fc.in_features, 2)
        # Load checkpoint (trained with effnet head; load where possible)
        ckpt = os.path.join(self.model_dir, 'best.pt')
        if os.path.isfile(ckpt):
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            sd = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
            try:
                me.load_state_dict(sd, strict=False)
            except Exception:
                pass
            # Attempt partial load into resnet (will mostly skip)
            try:
                mr.load_state_dict(sd, strict=False)
            except Exception:
                pass
        me.to(self.device).eval(); mr.to(self.device).eval()
        return me, mr

    def _build_transforms(self, size: int):
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=[0,0,0]),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])

    def _detect_and_crop_face(self, img_np: np.ndarray) -> np.ndarray:
        # Try OpenCV Haar cascade; fall back to center crop
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(64,64)) if self.face_cascade is not None else []
            if len(faces) > 0:
                # pick largest face
                x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
                pad = int(0.15*max(w,h))
                x0 = max(0, x-pad); y0 = max(0, y-pad); x1 = min(img_np.shape[1], x+w+pad); y1 = min(img_np.shape[0], y+h+pad)
                return img_np[y0:y1, x0:x1]
        except Exception:
            pass
        # Center crop square
        h, w = img_np.shape[:2]
        side = min(h, w)
        y0 = (h - side)//2; x0 = (w - side)//2
        return img_np[y0:y0+side, x0:x0+side]

    def _preprocess_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        # Fallback preprocessing without albumentations
        h, w = img_np.shape[:2]
        size = self.img_size
        scale = min(size / h, size / w)
        nh, nw = int(h * scale), int(w * scale)
        import cv2 as _cv2
        resized = _cv2.resize(img_np, (nw, nh))
        canvas = np.zeros((size, size, 3), dtype=resized.dtype)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        x = torch.from_numpy(canvas).float().permute(2,0,1) / 255.0
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        x = (x - mean) / std
        return x

    @torch.no_grad()
    def predict_image(self, img_np: np.ndarray, return_cam: bool = False):
        # TTA: rotations (0,90,270) and horizontal flips
        def _proc(arr):
            if self.use_albu:
                return self.transforms(image=arr)['image']
            return self._preprocess_tensor(arr)
        views = []
        for k in [0,1,3]:  # 0, 90, 270 degrees
            r = np.rot90(img_np, k)
            views.append(_proc(r))
            views.append(_proc(np.ascontiguousarray(np.flip(r, axis=1))))
        x = torch.stack(views, 0).to(self.device)
        # Use EfficientNet only (matches training architecture)
        logits_e = self.model_efficientnet(x)
        probs = F.softmax(logits_e, dim=1)
        fake_prob = float(probs[:, 1].mean().item())
        if self.invert_logits:
            fake_prob = 1.0 - fake_prob
        label = 'FAKE' if fake_prob >= self.threshold else 'REAL'
        conf = fake_prob if label=='FAKE' else 1.0 - fake_prob

        heatmap = None
        if return_cam and GradCAM is not None:
            target_layer = self._get_last_conv_layer()
            cam = GradCAM(model=self.model_efficientnet, target_layers=[target_layer], use_cuda=self.device=='cuda')
            # use first view for CAM
            grayscale_cam = cam(input_tensor=x[:1], targets=None)[0]
            heatmap = grayscale_cam
        return conf, label, heatmap

    def predict_image_file(self, path: str, return_cam: bool = False):
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        return self.predict_image(img_np, return_cam=return_cam)

    @torch.no_grad()
    def predict_video(self, frames: List[np.ndarray]) -> Tuple[List[float], float, str]:
        scores = []
        batch = []
        bs = 16
        for i, f in enumerate(frames):
            # TTA per frame: rotations + hflip (no face crop to match training)
            arrs = []
            for k in [0,1,3]:
                r = np.rot90(f, k)
                arrs.append(r)
                arrs.append(np.ascontiguousarray(np.flip(r, axis=1)))
            for arr in arrs:
                if self.use_albu:
                    batch.append(self.transforms(image=arr)['image'])
                else:
                    batch.append(self._preprocess_tensor(arr))
            if len(batch) >= bs or (i == len(frames)-1 and batch):
                x = torch.stack(batch, dim=0).to(self.device)
                # EfficientNet only
                le = self.model_efficientnet(x)
                probs = F.softmax(le, dim=1)
                probs = probs[:,1].view(-1, 2)  # group per-frame
                frame_probs = probs.mean(dim=1)
                scores.extend([float(p.item()) for p in frame_probs])
                batch = []
        # temporal smoothing (median then moving average)
        if len(scores) >= 5:
            import numpy as _np
            med = _np.copy(scores)
            k = 5
            half = k//2
            for t in range(len(scores)):
                l = max(0, t-half); r = min(len(scores), t+half+1)
                med[t] = float(_np.median(scores[l:r]))
            scores = _np.convolve(med, _np.ones(5)/5, mode='same').tolist()
        final = float(np.mean(scores)) if len(scores) > 0 else 0.0
        if self.invert_logits:
            final = 1.0 - final
        label = 'FAKE' if final >= self.threshold else 'REAL'
        conf = final if label=='FAKE' else 1.0 - final
        return scores, conf, label

    def _get_last_conv_layer(self):
        # EfficientNet last conv block for CAM
        return self.model_efficientnet.features[-1][0]
