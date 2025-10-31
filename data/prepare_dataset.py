import os
import argparse
import glob
import random
import shutil
import cv2


def extract_frames(videos_dir, out_dir, fps=1, img_size=300, label='real'):
    os.makedirs(out_dir, exist_ok=True)
    video_paths = []
    for ext in ('*.mp4','*.avi','*.mov','*.mkv','*.webm'):
        video_paths.extend(glob.glob(os.path.join(videos_dir, ext)))
    idx = 0
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        base = os.path.splitext(os.path.basename(vp))[0]
        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(frame_rate // fps))
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (img_size, img_size))
                save_dir = os.path.join(out_dir, label)
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"{base}_{idx}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                idx += 1
            fidx += 1
        cap.release()
    print(f"Saved {idx} frames to {out_dir}/{label}")


def split_dataset(root, val_ratio=0.2):
    # root contains combined/real and combined/fake, split into train/ and val/
    combined_real = os.path.join(root, 'combined', 'real')
    combined_fake = os.path.join(root, 'combined', 'fake')
    assert os.path.isdir(combined_real) and os.path.isdir(combined_fake), 'Expected combined/real and combined/fake folders'

    def _split_class(cls):
        files = glob.glob(os.path.join(root, 'combined', cls, '*'))
        random.shuffle(files)
        n_val = int(len(files) * val_ratio)
        val_files = set(files[:n_val])
        for f in files:
            split = 'val' if f in val_files else 'train'
            dst_dir = os.path.join(root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

    _split_class('real')
    _split_class('fake')
    print('Split complete')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('extract-frames')
    p1.add_argument('--videos_dir', type=str, required=True)
    p1.add_argument('--out_dir', type=str, required=True)
    p1.add_argument('--fps', type=int, default=1)
    p1.add_argument('--img_size', type=int, default=300)
    p1.add_argument('--label', type=str, choices=['real','fake'], default='real')

    p2 = sub.add_parser('split')
    p2.add_argument('--root', type=str, required=True)
    p2.add_argument('--val_ratio', type=float, default=0.2)

    # Build dataset from filename prefixes real-* and fake-* in a flat folder
    p3 = sub.add_parser('from-prefix')
    p3.add_argument('--src', type=str, required=True, help='Folder with files named like real-1.jpg, fake-1.png')
    p3.add_argument('--out', type=str, required=True, help='Output dataset root with train/ and val/')
    p3.add_argument('--val_ratio', type=float, default=0.2)

    args = parser.parse_args()
    if args.cmd == 'extract-frames':
        extract_frames(args.videos_dir, args.out_dir, args.fps, args.img_size, args.label)
    elif args.cmd == 'split':
        split_dataset(args.root, args.val_ratio)
    elif args.cmd == 'from-prefix':
        import shutil, random, os
        exts = ('.jpg','.jpeg','.png','.bmp','.webp')
        files = [f for f in glob.glob(os.path.join(args.src, '*')) if os.path.splitext(f)[1].lower() in exts]
        real = sorted([f for f in files if os.path.basename(f).lower().startswith('real-')])
        fake = sorted([f for f in files if os.path.basename(f).lower().startswith('fake-')])
        assert len(real) > 0 and len(fake) > 0, 'No real-* or fake-* files found'
        random.seed(42)
        random.shuffle(real); random.shuffle(fake)
        def split(lst):
            n_val = max(1, int(len(lst) * args.val_ratio)) if len(lst) > 1 else 1
            return lst[n_val:], lst[:n_val]
        real_train, real_val = split(real)
        fake_train, fake_val = split(fake)
        for split_name, items, cls in (
            ('train', real_train, 'real'), ('val', real_val, 'real'),
            ('train', fake_train, 'fake'), ('val', fake_val, 'fake')):
            dst_dir = os.path.join(args.out, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for src in items:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        print('Built dataset at', args.out)


if __name__ == '__main__':
    main()
