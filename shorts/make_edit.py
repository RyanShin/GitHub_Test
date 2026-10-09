#!/usr/bin/env python3
"""Cinematic "velocity edit" short, timed to a 100 BPM reference track.

Structure (mirrors the reference edit):
  0.0 - 2.0   gold script title writing on over drifting bokeh particles
  2.0 - 9.4   slow cinematic story shots with light-leak dissolves,
              ending on a polaroid print that the camera zoom-blurs into
  9.4 - 11.75 hero shot, gold script word writes on
  11.75-20.1  one cut per beat: zoom punches, shake, flash, RGB split,
              diagonal wipes, whip pans, bloom, star glints, B&W glow
  20.1 - 21.1 rapid-fire cuts
  21.1 - 22.4 bold caption pops in over a darkened shot
  22.4 - end  closing close-up, fade to black

usage: make_edit.py OUT.mp4 AUDIO P1 P2 P3 P4 P5 P6 P7
  P1 boy on beach, P2 woman on beach, P3 selfie in front of cafe,
  P4 cafe table (wide), P5 cafe close-up, P6 bridge (hands on head),
  P7 bridge (peace sign)
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
BEAT0, BEAT = 11.75, 0.5985  # first drop beat and beat period of the track
DUR = 23.18
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FONT = os.path.join(HERE, "fonts", "GreatVibes.woff")
BOLD_FONT = os.path.join(HERE, "fonts", "Anton.woff")

TITLE = "Miyakojima"
HERO_WORD = "Summer"
CAPTION = ("OUR", "SUMMER")


def B(n):
    return BEAT0 + n * BEAT


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def ease_out(p):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3


# ----------------------------------------------------------------------------
# sources

SRC = {}


def load_photo(path):
    im = Image.open(path).convert("RGB")
    # sharpen at full resolution; the edit crops in tight, so this matters
    return im.filter(ImageFilter.UnsharpMask(radius=2.5, percent=140, threshold=2))


def gold_text(text, font_path, size, pad=60):
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((pad - l, pad - t), text, font=font, fill=255)
    y = np.linspace(0, 1, h)[:, None, None]
    top, bot = np.array([255, 236, 170]), np.array([196, 140, 52])
    grad = (top * (1 - y) + bot * y) * np.ones((1, w, 1))
    a = np.asarray(mask, np.float32)[..., None] / 255
    glow = np.asarray(mask.filter(ImageFilter.GaussianBlur(18)), np.float32)[..., None] / 255
    glow = glow * np.array([1.0, 0.75, 0.35]) * 1.4
    return {"rgb": grad / 255, "a": a, "glow": glow, "w": w, "h": h}


def make_polaroid(photo):
    """Polaroid print lying on a warm blurred table, as a 1.5x canvas."""
    cw, ch = int(W * 1.5), int(H * 1.5)
    s = max(cw / photo.width, ch / photo.height)
    bg = photo.resize((int(photo.width * s) + 1, int(photo.height * s) + 1), Image.BILINEAR)
    bg = bg.crop((0, 0, cw, ch)).filter(ImageFilter.GaussianBlur(40))
    bg = np.asarray(bg, np.float32) / 255 * np.array([1.0, 0.8, 0.55]) * 0.7

    pw = int(cw * 0.78)
    ph = int(pw * photo.height / photo.width)
    pic = photo.resize((pw, ph), Image.LANCZOS)
    m, mb = 40, 230
    card = Image.new("RGB", (pw + 2 * m, ph + m + mb), (246, 243, 236))
    card.paste(pic, (m, m))
    d = ImageDraw.Draw(card)
    font = ImageFont.truetype(SCRIPT_FONT, 150)
    d.text((m + 30, ph + m + 30), "Blue Turtle", font=font, fill=(40, 45, 70))
    cmask = Image.new("L", card.size, 255).rotate(-4, expand=True)
    card = card.rotate(-4, resample=Image.BICUBIC, expand=True)

    canvas = Image.fromarray((np.clip(bg, 0, 1) * 255).astype(np.uint8))
    shadow = Image.new("L", canvas.size, 0)
    x0, y0 = (cw - card.width) // 2, (ch - card.height) // 2
    shadow.paste(cmask, (x0 + 30, y0 + 40))
    shadow = shadow.filter(ImageFilter.GaussianBlur(35))
    canvas = Image.composite(Image.new("RGB", canvas.size), canvas, shadow.point(lambda v: v * 0.7))
    canvas.paste(card, (x0, y0), cmask)
    # centre of the photo area in canvas fractions (for the zoom-in)
    return canvas, ((x0 + card.width / 2) / cw, (y0 + (m + ph / 2) + 20) / ch)


# ----------------------------------------------------------------------------
# frame helpers (float32 HxWx3 in 0..1)

def view(img, cx, cy, z, rot=0.0):
    """9:16 window centred at (cx, cy) (fractions), z=1 -> full image height."""
    sh = img.height / z
    sw = sh * W / H
    if sw > img.width:  # very tall source: fit width instead
        sw = img.width / z
        sh = sw * H / W
    px = min(max(cx * img.width, sw / 2), img.width - sw / 2)
    py = min(max(cy * img.height, sh / 2), img.height - sh / 2)
    s = sh / H
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    data = (s * c, -s * si, px - s * c * W / 2 + s * si * H / 2,
            s * si, s * c, py - s * si * W / 2 - s * c * H / 2)
    out = img.transform((W, H), Image.AFFINE, data, Image.BICUBIC)
    return np.asarray(out, np.float32) / 255


def to_pil(f):
    return Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8))


def grade(f, kind):
    if kind == "warm":
        f = f ** 0.95 * np.array([1.07, 1.0, 0.86])
        f = (f - 0.5) * 1.08 + 0.5
        sat = 1.05
    elif kind == "dark":
        f = f * 0.55
        sat = 0.6
    else:  # punchy teal/orange
        f = (f - 0.5) * 1.15 + 0.5
        lum = f.mean(axis=2, keepdims=True)
        f = f + (0.5 - lum) * np.array([-0.04, 0.0, 0.05]) * 0.6
        sat = 1.22
    lum = f.mean(axis=2, keepdims=True)
    return np.clip(lum + (f - lum) * sat, 0, 1)


def bloom(f, amt, thr=0.7, radius=10):
    small = to_pil(f).resize((W // 4, H // 4), Image.BILINEAR)
    s = np.asarray(small, np.float32) / 255
    s = np.clip((s - thr) / (1 - thr), 0, 1)
    s = to_pil(s).filter(ImageFilter.GaussianBlur(radius)).resize((W, H), Image.BILINEAR)
    g = np.asarray(s, np.float32) / 255 * amt
    return 1 - (1 - f) * (1 - g)


def bw(f):
    lum = (f * np.array([0.3, 0.59, 0.11])).sum(axis=2, keepdims=True)
    lum = np.clip((lum - 0.5) * 1.35 + 0.5, 0, 1)
    return bloom(np.repeat(lum, 3, axis=2), 0.9, thr=0.5, radius=14)


def radial_blur(f, amt, steps=7):
    if amt <= 0.002:
        return f
    img = to_pil(f)
    acc = np.zeros_like(f)
    for i in range(steps):
        s = 1 + amt * i / (steps - 1)
        cw, ch = W / s, H / s
        box = ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)
        acc += np.asarray(img.resize((W, H), Image.BILINEAR, box=box), np.float32) / 255
    return acc / steps


def rgb_split(f, k):
    k = int(k)
    if k < 1:
        return f
    out = f.copy()
    out[:, :, 0] = np.roll(f[:, :, 0], k, axis=1)
    out[:, :, 2] = np.roll(f[:, :, 2], -k, axis=1)
    return out


def whip(a, b, p):
    e = ease(p)
    off = int(e * W)
    strip = np.concatenate([a, b], axis=1)[:, off:off + W]
    k = int(160 * math.sin(math.pi * p)) + 1
    acc = np.zeros_like(strip)
    for i in range(6):
        acc += np.roll(strip, -int(i * k / 6), axis=1)
    return acc / 6


def diag_wipe(a, b, p):
    p = ease(p)
    yy, xx = np.mgrid[0:H, 0:W]
    d = (xx / W + yy / H) / 2  # 0 at top-left, 1 at bottom-right
    edge = p * 1.15 - 0.075
    m = np.clip((edge - d) / 0.01, 0, 1)[..., None]
    band = np.exp(-((d - edge) / 0.012) ** 2)[..., None] * (1 - abs(2 * p - 1))
    return np.clip(a * (1 - m) + b * m + band, 0, 1)


def light_leak(f, p):
    if p <= 0 or p >= 1:
        return f
    yy, xx = np.mgrid[0:H:8, 0:W:8]
    cx = -0.3 + 1.6 * p
    g = np.exp(-(((xx / W - cx) / 0.35) ** 2 + ((yy / H - 0.4) / 0.6) ** 2))
    g = to_pil(np.repeat(g[..., None], 3, axis=2)).resize((W, H), Image.BILINEAR)
    g = np.asarray(g, np.float32)[..., :1] / 255 * math.sin(math.pi * p) * 0.75
    return 1 - (1 - f) * (1 - g * np.array([1.0, 0.55, 0.2]))


STAR = None


def star_sprite(r=150):
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32) / r
    s = np.exp(-np.abs(x) * 30) * np.exp(-(y ** 2) * 2.5) + np.exp(-np.abs(y) * 30) * np.exp(-(x ** 2) * 2.5)
    xr, yr = (x + y) / 1.414, (x - y) / 1.414
    s += 0.35 * (np.exp(-np.abs(xr) * 40) * np.exp(-(yr ** 2) * 9) + np.exp(-np.abs(yr) * 40) * np.exp(-(xr ** 2) * 9))
    s += 0.8 * np.exp(-(x ** 2 + y ** 2) * 60)
    return np.clip(s, 0, 1.5)[..., None] * np.array([0.85, 0.92, 1.0])


def stars(f, amt, n=3):
    global STAR
    if amt <= 0.01:
        return f
    if STAR is None:
        STAR = star_sprite()
    r = STAR.shape[0] // 2
    lum = np.asarray(to_pil(f).convert("L").resize((W // 20, H // 20)), np.float32)
    out = f.copy()
    flat = np.argsort(lum.ravel())[::-1]
    picked = []
    for idx in flat:
        y, x = divmod(int(idx), lum.shape[1])
        if all(abs(x - px) + abs(y - py) > 12 for px, py in picked):
            picked.append((x, y))
        if len(picked) == n:
            break
    for i, (x, y) in enumerate(picked):
        cx, cy = x * 20 + 10, y * 20 + 10
        a = amt * (1.0 - 0.25 * i)
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        x1, y1 = min(cx + r + 1, W), min(cy + r + 1, H)
        spr = STAR[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
        out[y0:y1, x0:x1] += spr * a
    return np.clip(out, 0, 1)


def particles(f, T, n=90, strength=1.0, seed=3):
    rng = np.random.default_rng(seed)
    px, py = rng.random(n), rng.random(n)
    vy = 0.015 + rng.random(n) * 0.04
    size = 3 + rng.random(n) ** 3 * 22
    ph = rng.random(n) * 6.28
    out = f.copy()
    for i in range(n):
        x = int((px[i] + 0.01 * math.sin(T * 0.7 + ph[i])) * W)
        y = int(((py[i] - vy[i] * T) % 1.0) * H)
        r = int(size[i] * 2)
        tw = 0.55 + 0.45 * math.sin(T * 3 + ph[i])
        x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, W), max(y - r, 0), min(y + r + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (size[i] / 2) ** 2))
        a = strength * tw * (0.9 if size[i] < 8 else 0.35)
        out[y0:y1, x0:x1] += g[..., None] * a * np.array([1.0, 0.78, 0.4])
    return np.clip(out, 0, 1)


def write_on(f, txt, p, cx, cy, alpha=1.0):
    """Reveal gold text left->right with a glowing pen tip at the edge."""
    if p <= 0 or alpha <= 0:
        return f
    rgb, a, glow = txt["rgb"], txt["a"], txt["glow"]
    w, h = txt["w"], txt["h"]
    xs = np.arange(w)[None, :, None]
    edge = p * (w + 80) - 40
    m = np.clip((edge - xs) / 40, 0, 1) * alpha
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    sl = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    region = f[fy0:fy1, fx0:fx1]
    region = 1 - (1 - region) * (1 - glow[sl] * m[:, sl[1]] * 0.8)
    aa = a[sl] * m[:, sl[1]]
    region = region * (1 - aa) + rgb[sl] * aa
    out = f.copy()
    out[fy0:fy1, fx0:fx1] = region
    if p < 1:
        tip_x = x0 + edge
        rows = a[:, int(min(max(edge, 0), w - 1)), 0]
        tip_y = y0 + (np.argmax(rows) if rows.max() > 0 else h / 2)
        out = particles_tip(out, tip_x, tip_y)
    return np.clip(out, 0, 1)


def particles_tip(f, x, y, r=60):
    x, y = int(x), int(y)
    x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, W), max(y - r, 0), min(y + r + 1, H)
    if x0 >= x1 or y0 >= y1:
        return f
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 14 ** 2))
    f[y0:y1, x0:x1] += g[..., None] * np.array([1.0, 0.9, 0.6])
    return f


def pop_text(f, text, font_path, size, cy, color, t, outline=True):
    """Caption that slams in: scale 1.5 -> 1 over 0.12s."""
    if t < 0:
        return f
    s = 1 + 0.5 * (1 - ease_out(t / 0.12))
    font = ImageFont.truetype(font_path, int(size * s))
    l, tp, r, b = font.getbbox(text)
    w, h = r - l + 40, b - tp + 40
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((20 - l, 20 - tp), text, font=font, fill=255)
    a = np.asarray(m, np.float32)[..., None] / 255 * min(t / 0.05, 1)
    sh = np.asarray(m.filter(ImageFilter.GaussianBlur(12)), np.float32)[..., None] / 255
    glow = np.asarray(m.filter(ImageFilter.GaussianBlur(25)), np.float32)[..., None] / 255
    x0, y0 = (W - w) // 2, int(cy - h / 2)
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    sl = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    k = min(t / 0.05, 1)
    reg = f[fy0:fy1, fx0:fx1] * (1 - sh[sl] * 0.6 * k)
    reg = 1 - (1 - reg) * (1 - glow[sl] * np.array(color) * 0.5 * k)
    reg = reg * (1 - a[sl]) + np.array(color) * a[sl]
    out = f.copy()
    out[fy0:fy1, fx0:fx1] = reg
    return out


# ----------------------------------------------------------------------------
# timeline

def shot(src, t0, t1, c0, c1=None, grade="punch", fx=(), ink=None):
    return dict(src=src, t0=t0, t1=t1, c0=c0, c1=c1 or c0, grade=grade, fx=set(fx), ink=ink)


def build_timeline():
    S = []
    # story section (warm, slow) -- (cx, cy, zoom, rotation)
    S.append(shot("P1", 2.0, 3.66, (0.22, 0.5, 1.0, 0), (0.42, 0.5, 1.05, 0), "warm", ["fadein"]))
    S.append(shot("P2", 3.66, 5.2, (0.44, 0.6, 1.05, 0), (0.44, 0.62, 1.22, 0), "warm", ["leak"]))
    S.append(shot("P3", 5.2, 6.45, (0.66, 0.5, 1.0, 1.5), (0.68, 0.5, 1.15, 0), "warm", ["leak"]))
    S.append(shot("POLA", 6.45, 9.36, (0.5, 0.5, 1.0, 0), (0.5, 0.5, 1.7, 0), "warm", ["leak", "zoomout"]))
    S.append(shot("P6", 9.36, B(0), (0.4, 0.55, 1.05, 0), (0.4, 0.5, 1.2, 0), "punch",
                  ["zoomin", "bloom", "hero"]))
    # drop: one cut per beat
    beat_shots = [
        ("P6", (0.42, 0.45, 1.15), (0.42, 0.45, 1.25), ["punch", "flash", "stars", "bloom"]),
        ("P1", (0.38, 0.45, 1.7), (0.38, 0.45, 1.85), ["punch", "shake"]),
        ("P5", (0.6, 0.38, 1.5), (0.6, 0.38, 1.6), ["whip", "rgb"]),
        ("P7", (0.5, 0.5, 1.0), (0.5, 0.45, 1.1), ["wipe", "bloom"]),
        ("P3", (0.72, 0.5, 1.25), (0.72, 0.5, 1.35), ["flash", "punch", "bloom"]),
        ("P2", (0.44, 0.7, 1.6), (0.44, 0.7, 1.75), ["zoomin", "stars"]),
        ("P4", (0.54, 0.8, 2.2), (0.54, 0.8, 2.4), ["punch", "shake"]),
        ("P6", (0.4, 0.47, 2.1), (0.4, 0.47, 2.3), ["bw", "flash"]),
        ("P7", (0.52, 0.4, 1.9), (0.52, 0.4, 2.05), ["punch", "stars", "rgb"]),
        ("P1", (0.2, 0.5, 1.0), (0.45, 0.5, 1.0), ["whip", "bloom"]),
        ("P5", (0.6, 0.32, 2.2), (0.6, 0.32, 2.35), ["flash", "rgb", "shake"]),
        ("P3", (0.83, 0.48, 1.9), (0.83, 0.48, 2.0), ["wipe", "punch"]),
        ("P2", (0.45, 0.5, 1.0), (0.45, 0.55, 1.12), ["stars", "bloom", "flash"]),
        ("P6", (0.38, 0.5, 1.35, 3), (0.38, 0.5, 1.5, 0), ["punch", "shake", "rgb"]),
    ]
    for i, (k, c0, c1, fx) in enumerate(beat_shots):
        S.append(shot(k, B(i), B(i + 1), c0 + (0,) * (4 - len(c0)), c1 + (0,) * (4 - len(c1)), "punch", fx))
    # rapid-fire on 16th notes
    rapid = [("P1", (0.38, 0.43, 2.0)), ("P3", (0.6, 0.55, 2.0)), ("P4", (0.59, 0.25, 2.4)),
             ("P7", (0.5, 0.45, 1.4)), ("P2", (0.44, 0.67, 2.4)), ("P5", (0.6, 0.4, 1.8))]
    t = B(14)
    step = (B(15.5) - B(14)) / len(rapid)
    for k, c in rapid:
        S.append(shot(k, t, t + step, c + (0,), (c[0], c[1], c[2] * 1.06, 0), "punch", ["flash", "punch"]))
        t += step
    S.append(shot("P5", t, B(17.66), (0.6, 0.32, 1.7, 0), (0.6, 0.32, 1.85, 0), "dark", ["caption"]))
    S.append(shot("P7", B(17.66), DUR, (0.52, 0.42, 1.6, 0), (0.52, 0.42, 1.9, 0), "punch",
                  ["flash", "bloom", "fadeout"]))
    return S


TL = None
TXT = {}


def init(paths):
    global TL
    for i, p in enumerate(paths, 1):
        SRC[f"P{i}"] = load_photo(p)
    SRC["POLA"], pc = make_polaroid(SRC["P4"])
    TL = build_timeline()
    # point the polaroid push-in at the photo inside the card
    for s in TL:
        if s["src"] == "POLA":
            s["c0"] = (pc[0], pc[1] - 0.04, s["c0"][2], s["c0"][3])
            s["c1"] = (pc[0], pc[1], s["c1"][2], s["c1"][3])
    TXT["title"] = gold_text(TITLE, SCRIPT_FONT, 230)
    TXT["hero"] = gold_text(HERO_WORD, SCRIPT_FONT, 300)


def render_shot(s, T, frame_no):
    lt = T - s["t0"]
    d = s["t1"] - s["t0"]
    u = ease(lt / d) if "whip" not in s["fx"] else lt / d
    c = [a + (b - a) * u for a, b in zip(s["c0"], s["c1"])]
    if "punch" in s["fx"]:
        c[2] *= 1 + 0.18 * (1 - ease_out(lt / 0.3))
    if "shake" in s["fx"]:
        rng = np.random.default_rng(frame_no)
        amp = math.exp(-lt / 0.25)
        c[0] += rng.normal() * 0.012 * amp
        c[1] += rng.normal() * 0.008 * amp
        c[3] += rng.normal() * 1.5 * amp
        c[2] *= 1 + 0.03 * amp
    if abs(c[3]) > 0.01:
        c[2] *= 1.06  # hide rotated corners
    f = view(SRC[s["src"]], *c)
    f = grade(f, s["grade"])
    fx = s["fx"]
    if "bw" in fx:
        f = bw(f)
    if "bloom" in fx:
        f = bloom(f, 0.4)
    if "stars" in fx:
        f = stars(f, math.sin(math.pi * min(lt / 0.5, 1)) * 0.9)
    if "zoomin" in fx:
        f = radial_blur(f, 0.25 * (1 - ease_out(lt / 0.3)))
    if "zoomout" in fx and lt > d - 0.35:
        f = radial_blur(f, 0.35 * ease((lt - (d - 0.35)) / 0.35))
    if "rgb" in fx:
        f = rgb_split(f, 22 * math.exp(-lt / 0.1))
    if "hero" in fx:
        f = write_on(f, TXT["hero"], (lt - 0.4) / 1.1, W / 2, H * 0.72)
    if "caption" in fx:
        f = pop_text(f, CAPTION[0], BOLD_FONT, 170, H * 0.5, (1, 1, 1), lt)
        f = pop_text(f, CAPTION[1], BOLD_FONT, 300, H * 0.5 + 240, (0.92, 0.12, 0.12), lt - BEAT)
    if "flash" in fx:
        f = 1 - (1 - f) * (1 - 0.9 * math.exp(-lt / 0.07))
    if s["grade"] == "warm":
        f = particles(f, T, n=40, strength=0.6)
    return f


def render(i):
    T = i / FPS
    if T < 2.0:  # intro: particles + gold title
        f = np.zeros((H, W, 3), np.float32) + np.array([0.035, 0.025, 0.012])
        f = particles(f, T + 2)
        alpha = 1 - ease((T - 1.35) / 0.55)
        f = write_on(f, TXT["title"], (T - 0.1) / 1.0, W / 2, H * 0.45, alpha=alpha)
        return to_pil(f).tobytes()

    idx = max(j for j, s in enumerate(TL) if s["t0"] <= T)
    s = TL[idx]
    f = render_shot(s, T, i)
    lt = T - s["t0"]
    prev = TL[idx - 1] if idx > 0 else None
    if "fadein" in s["fx"]:
        f = f * ease(lt / 0.4)
    if "leak" in s["fx"] and lt < 0.5 and prev:
        a = render_shot(prev, T, i)
        p = lt / 0.5
        f = light_leak(a * (1 - ease(p)) + f * ease(p), p)
    if "whip" in s["fx"] and lt < 0.15 and prev:
        f = whip(render_shot(prev, T, i), f, lt / 0.15)
    if "wipe" in s["fx"] and lt < 0.22 and prev:
        f = diag_wipe(render_shot(prev, T, i), f, lt / 0.22)
    if "fadeout" in s["fx"]:
        f = f * (1 - ease((T - (DUR - 0.5)) / 0.5))
    return to_pil(f).tobytes()


def main():
    out, audio, *paths = sys.argv[1:]
    assert len(paths) == 7, __doc__
    n = int(DUR * FPS)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", audio,
        "-vf", "unsharp=5:5:0.5:5:5:0,vignette=PI/5,noise=alls=2:allf=t",
        "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart", out,
    ]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init, initargs=(paths,)) as pool:
        for i, buf in enumerate(pool.imap(render, range(n), chunksize=4)):
            enc.stdin.write(buf)
            if i % 90 == 0:
                print(f"frame {i}/{n}", file=sys.stderr)
    enc.stdin.close()
    sys.exit(enc.wait())


if __name__ == "__main__":
    main()
