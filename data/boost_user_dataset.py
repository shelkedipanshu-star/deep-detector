import os
import argparse
import glob
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

random.seed(42)


def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return [p for p in glob.glob(os.path.join(folder, '*')) if os.path.splitext(p)[1].lower() in exts]


def augment_once(img: Image.Image):
    # Random geometric
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        k = random.choice([0, 1, 3])
        img = img.rotate(90*k, expand=True)
    # Color jitter
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(0.9 + 0.2*random.random())
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(0.9 + 0.3*random.random())
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(0.9 + 0.3*random.random())
    # Noise
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    # Blur/compression
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
    if random.random() < 0.4:
        # JPEG compression round-trip
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        q = random.randint(60, 90)
        ok, enc = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            arr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return img


def upsample_class(files, out_dir, target):
    os.makedirs(out_dir, exist_ok=True)
    # Copy originals first
    idx = 0
    for p in files:
        name = os.path.basename(p)
        Image.open(p).convert('RGB').save(os.path.join(out_dir, f'orig_{name}'))
        idx += 1
    # Augment until target
    base = [Image.open(p).convert('RGB') for p in files]
    i = 0
    while idx < target and len(base) > 0:
        img = augment_once(base[i % len(base)])
        img.save(os.path.join(out_dir, f'aug_{idx:05d}.jpg'), quality=95)
        idx += 1
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='ImageFolder dataset root with train/val/real,fake')
    ap.add_argument('--out', required=True, help='Output dataset root')
    ap.add_argument('--target_per_class', type=int, default=200)
    args = ap.parse_args()

    for split in ['train', 'val']:
        for cls in ['real', 'fake']:
            src_dir = os.path.join(args.src, split, cls)
            files = list_images(src_dir)
            if len(files) == 0:
                continue
            dst_dir = os.path.join(args.out, split, cls)
            upsample_class(files, dst_dir, args.target_per_class if split=='train' else max(len(files), 40))
    print('Boosted dataset built at', args.out)


if __name__ == '__main__':
    main()
