#!/usr/bin/env python3
"""Turn a list of photos into a beat-synced 1080x1920 shorts video.

Each photo becomes a 4:5 card over a blurred copy of itself, with a slow
Ken Burns zoom/pan, a small zoom "bounce" on every beat, and a transition
centred on each bar line. Photos are sharpened (unsharp mask at source
resolution, then a light unsharp pass in ffmpeg) and colour-boosted.

usage: make_short.py OUT.mp4 AUDIO.wav IMG[:x0:x1] [IMG[:x0:x1] ...]
  x0/x1 = horizontal centre of the crop at clip start/end, as a fraction of
  the image width (default 0.5:0.5). Use it to keep the subject in frame.
"""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H = 1080, 1920
CARD_W, CARD_H = 1080, 1350
CARD_Y = (H - CARD_H) // 2
FPS = 30
BPM = 120
BEAT = 60.0 / BPM
BAR = BEAT * 4
TR = 0.5  # transition length, centred on the bar line
SRC_H = 1600  # working height of the source photos
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TITLE = ("MIYAKOJIMA", "summer memories")
OUTRO = "see you again"


def ease(p):
    return p * p * (3 - 2 * p)


class Clip:
    def __init__(self, path, x0=0.5, x1=0.5):
        im = Image.open(path).convert("RGB")
        im = im.resize((round(im.width * SRC_H / im.height), SRC_H), Image.LANCZOS)
        im = im.filter(ImageFilter.UnsharpMask(radius=2.2, percent=130, threshold=2))
        im = ImageEnhance.Color(im).enhance(1.12)
        self.src = im
        self.x0, self.x1 = x0, x1

        # blurred, darkened full-frame background (cover fit)
        s = max(W / im.width, H / im.height)
        bg = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        l, t = (bg.width - W) // 2, (bg.height - H) // 2
        bg = bg.crop((l, t, l + W, t + H)).filter(ImageFilter.GaussianBlur(45))
        bg = ImageEnhance.Brightness(bg).enhance(0.6)
        # soft drop shadow under the card
        sh = Image.new("L", (W, H), 0)
        ImageDraw.Draw(sh).rectangle((20, CARD_Y + 25, W - 20, CARD_Y + CARD_H + 25), fill=150)
        sh = sh.filter(ImageFilter.GaussianBlur(30))
        self.bg = Image.composite(Image.new("RGB", (W, H)), bg, sh)

    def card(self, u, zoom):
        """u = 0..1 progress through the clip, zoom >= 1."""
        im = self.src
        s = im.height / (CARD_H * zoom)  # source px per output px
        half = CARD_W * s / 2
        cx = (self.x0 + (self.x1 - self.x0) * ease(u)) * im.width
        cx = min(max(cx, half), im.width - half)
        cy = im.height / 2
        data = (s, 0, cx - half, 0, s, cy - CARD_H * s / 2)
        return im.transform((CARD_W, CARD_H), Image.AFFINE, data, Image.BICUBIC)

    def frame(self, u, T, extra_zoom=0.0):
        # slow push-in + a decaying bump on every beat
        bump = 0.03 * np.exp(-(T % BEAT) / 0.09) if T >= BAR else 0.0
        zoom = 1.0 + 0.10 * u + bump + extra_zoom
        f = self.bg.copy()
        f.paste(self.card(u, zoom), (0, CARD_Y))
        return f


def whip(a, b, p):
    """Horizontal whip-pan with motion blur."""
    e = ease(p)
    off = int(e * W)
    A, B = np.asarray(a, np.float32), np.asarray(b, np.float32)
    strip = np.concatenate([A, B], axis=1)[:, off:off + W]
    k = int(120 * np.sin(np.pi * p)) + 1
    acc = np.zeros_like(strip)
    for i in range(6):
        acc += np.roll(strip, -int(i * k / 6), axis=1)
    return Image.fromarray((acc / 6).astype(np.uint8))


def flash(a, b, p):
    m = Image.blend(a, b, ease(p))
    white = Image.new("RGB", (W, H), (255, 255, 255))
    return Image.blend(m, white, 0.85 * np.sin(np.pi * p) ** 2)


def draw_text(img, lines, alpha):
    if alpha <= 0:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sizes = [96, 44]
    y = CARD_Y - 230
    for line, size in zip(lines, sizes):
        font = ImageFont.truetype(FONT, size)
        w = d.textlength(line, font=font)
        a = int(255 * alpha)
        d.text(((W - w) / 2 + 3, y + 3), line, font=font, fill=(0, 0, 0, a // 2))
        d.text(((W - w) / 2, y), line, font=font, fill=(255, 255, 255, a))
        y += size + 24
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def main():
    out, audio = sys.argv[1], sys.argv[2]
    clips = []
    for arg in sys.argv[3:]:
        path, *xs = arg.split(":")
        clips.append(Clip(path, *map(float, xs)) if xs else Clip(path))

    n = len(clips)
    # clip k is "on screen" from start[k]; transition k-1 -> k is centred on bar k
    start = [0.0] + [k * BAR - TR / 2 for k in range(1, n)]
    total = n * BAR + 0.5
    end = start[1:] + [total]
    dur = [e - s for s, e in zip(start, end)]
    transitions = [whip, flash]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", audio,
        "-vf", ("unsharp=5:5:0.6:5:5:0,eq=contrast=1.05:saturation=1.08,"
                f"vignette=PI/6,fade=t=in:st=0:d=0.4,fade=t=out:st={total - 0.6}:d=0.6"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out,
    ]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def local(k, T):
        return min(max((T - start[k]) / dur[k], 0.0), 1.0)

    for i in range(int(total * FPS)):
        T = i / FPS
        k = max(j for j in range(n) if start[j] <= T)
        t = T - start[k]
        if k > 0 and t < TR:
            p = t / TR
            fx = transitions[(k - 1) % len(transitions)]
            a = clips[k - 1].frame(local(k - 1, T), T)
            b = clips[k].frame(local(k, T), T)
            img = fx(a, b, p)
        else:
            img = clips[k].frame(local(k, T), T)

        if k == 0:
            alpha = min(max((T - 0.2) / 0.4, 0), 1) * min(max((start[1] - T) / 0.3, 0), 1)
            img = draw_text(img, TITLE, alpha)
        if k == n - 1:
            img = draw_text(img, (OUTRO,), min(max((T - start[k] - 0.6) / 0.4, 0), 1))

        enc.stdin.write(img.tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{int(total * FPS)}", file=sys.stderr)
    enc.stdin.close()
    sys.exit(enc.wait())


if __name__ == "__main__":
    main()
