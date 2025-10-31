import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io


def make_background(size):
    w, h = size, size
    # Smooth radial gradient background
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    base = np.clip(1 - rr, 0, 1)
    colors = np.stack([
        base * (0.3 + 0.7*np.random.rand()),
        base * (0.3 + 0.7*np.random.rand()),
        base * (0.3 + 0.7*np.random.rand()),
    ], axis=-1)
    img = (colors * 255).astype(np.uint8)
    return Image.fromarray(img)


def draw_shapes(img, n=5):
    draw = ImageDraw.Draw(img, 'RGBA')
    w, h = img.size
    for _ in range(n):
        x1, y1 = random.randint(0, w-50), random.randint(0, h-50)
        x2, y2 = x1 + random.randint(20, w//2), y1 + random.randint(20, h//2)
        color = (random.randint(50,255), random.randint(50,255), random.randint(50,255), random.randint(80,160))
        if random.random() < 0.5:
            draw.ellipse([x1,y1,x2,y2], fill=color)
        else:
            draw.rectangle([x1,y1,x2,y2], fill=color)
    return img


def add_fake_artifacts(img):
    # Downscale-upscale (pixelation)
    w, h = img.size
    s = random.choice([int(w*0.5), int(w*0.4), int(w*0.3)])
    img_small = img.resize((max(8,s), max(8,int(h*s/w))), Image.NEAREST)
    img = img_small.resize((w,h), Image.NEAREST)
    # Add blur
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8,1.8)))
    # Strong JPEG compression artifacts
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=random.randint(10,25))
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    # Overlay grid-like compression pattern
    draw = ImageDraw.Draw(img, 'RGBA')
    step = random.choice([8,12,16])
    for x in range(0, w, step):
        draw.line([(x,0),(x,h)], fill=(0,0,0,20))
    for y in range(0, h, step):
        draw.line([(0,y),(w,y)], fill=(0,0,0,20))
    return img


def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format='JPEG', quality=95)


def generate_split(out_dir, count, size, label):
    split_dir = out_dir
    for i in range(count):
        img = make_background(size)
        img = draw_shapes(img, n=random.randint(3,8))
        if label == 'fake':
            img = add_fake_artifacts(img)
        save_img(img, os.path.join(split_dir, label, f"{label}_{i:05d}.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--n_train', type=int, default=400)
    ap.add_argument('--n_val', type=int, default=100)
    ap.add_argument('--size', type=int, default=300)
    args = ap.parse_args()

    train_dir = os.path.join(args.out, 'train')
    val_dir = os.path.join(args.out, 'val')

    # Train
    generate_split(os.path.join(train_dir), args.n_train, args.size, 'real')
    generate_split(os.path.join(train_dir), args.n_train, args.size, 'fake')
    # Val
    generate_split(os.path.join(val_dir), args.n_val, args.size, 'real')
    generate_split(os.path.join(val_dir), args.n_val, args.size, 'fake')

    print('Synthetic dataset generated at', args.out)

if __name__ == '__main__':
    main()
