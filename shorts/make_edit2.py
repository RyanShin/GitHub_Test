#!/usr/bin/env python3
"""32 s cinematic velocity edit (photos + video clips) on the full reference
track, with the spoken lines replaced (see make_voice.py).

Structure (mirrors the reference edit, beat grid 9.68 + 0.6 n):
  0.0 - 4.6   establishing shots + narration, title banner on top
  4.6 - 7.25  polaroid print, camera pushes in and zoom-blurs into it
  7.25- 9.68  hero shot, gold script word writes on
  9.68-19.25  beat cuts (one per beat, some held two beats) + rapid cuts
  19.25-24.08 dialogue: "IN THIS / LIFE", "you either / DREAM IT",
              gold "Live it" -- captions synced to the spoken words
  24.08-29.0  second beat section + rapid-fire cuts
  29.0 - end  gold logo with particles, fade out

usage: make_edit2.py OUT.mp4 AUDIO.wav TIMES.json MEDIA_DIR
  MEDIA_DIR holds the photos/clips named in MEDIA below.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

import make_edit as E
from make_edit import W, H, FPS, ease, ease_out

DUR = 31.95
BANNER_FONT = "/usr/share/fonts/opentype/inter/Inter-BoldItalic.otf"
BANNER = ("The Best Summer", "to Ever Exist.")
HERO_WORD = "Summer"
LOGO = "Miyakojima"
POLA_CAPTION = "Irabu Bridge"
CAP1 = ("IN THIS", "LIFE")
CAP2 = ("you either", "DREAM IT")
GOLD_WORD = "Live it"

MEDIA = {
    "restaurant": "864d1555-image.jpg", "selfie": "871bd563-image.jpg",
    "boy_arms": "64652f64-image.jpg", "mom_walk": "c92d73b1-image.jpg",
    "mom_sea": "61b7e95d-image.jpg", "food": "2cf15dab-image.jpg",
    "boy_bridge": "53d0fb7e-image.jpg", "bridge_wide": "f38827a6-image.jpg",
    "shop": "75ef2677-image.jpg", "mom_bridge": "a9c581cb-image.jpg",
    "snorkel": "1a386d9d-image.jpg", "coral1": "ea2f66ce-image.jpg",
    "coral_fish": "1155e1e4-image.jpg", "coral_purple": "2b8cf25b-image.jpg",
    "seafan": "8a472a76-image.jpg", "drone": "fe74a14e-image.jpg",
    "V_beach": "eb9313dc-2026_06_10_21_32.mp4",
    "V_bridge": "13eb7158-2026_06_10_21_34.mp4",
    "V_boat": "8d836de8-2026_06_10_21_51.mp4",
}


def B(n):
    return 9.68 + 0.6 * n


# where the subject is in each clip: (clip seconds, cx, cy) as frame fractions
TRACK = {
    "V_beach": [(0, .57, .58), (1, .53, .58), (2, .57, .58), (3, .62, .58), (4, .58, .58),
                (5, .53, .58), (6, .47, .58), (7, .5, .58), (8, .6, .58), (9, .62, .58),
                (10, .6, .58), (11, .6, .58), (12, .62, .58), (12.9, .65, .58)],
    "V_boat": [(0, .55, .45), (.5, .52, .42), (1, .58, .4), (1.5, .66, .36), (2, .68, .38),
               (2.5, .55, .4), (5, .38, .5), (5.5, .4, .45), (6, .4, .5), (6.5, .32, .55),
               (7, .28, .55), (7.5, .3, .5), (8, .48, .5), (8.5, .5, .5), (9, .6, .5), (9.5, .62, .45)],
    "V_bridge": [(0, .3, .4), (1, .3, .4), (2, .27, .4), (3, .22, .45), (4, .25, .5), (4.5, .2, .5),
                 (13, .18, .45), (13.5, .33, .45), (14, .42, .45), (14.5, .55, .45), (15, .7, .45),
                 (15.5, .7, .45), (16, .4, .45), (16.5, .37, .45), (23.5, .37, .45), (24, .18, .45)],
}


def tracked(clip, t):
    k = TRACK[clip]
    ts = [p[0] for p in k]
    return float(np.interp(t, ts, [p[1] for p in k])), float(np.interp(t, ts, [p[2] for p in k]))


def shot(src, t0, t1, c0, c1=None, grade="punch", fx=(), vt=0.0, spd=1.0, track=False):
    c0 = tuple(c0) + (0,) * (4 - len(c0))
    c1 = tuple(c1 or c0) + (0,) * (4 - len(c1 or c0))
    return dict(src=src, t0=t0, t1=t1, c0=c0, c1=c1, grade=grade, fx=set(fx), vt=vt, spd=spd,
                track=track)


def vshot(clip, t0, t1, vt, z0, z1, grade="punch", fx=()):
    """Video segment whose crop follows the subject (see TRACK)."""
    return shot(clip, t0, t1, (0, 0, z0), (0, 0, z1), grade, fx, vt=vt, track=True)


def build_timeline():
    S = []
    # --- story -----------------------------------------------------------
    S.append(shot("drone", 0.0, 2.3, (0.38, 0.5, 1.0), (0.36, 0.52, 1.25), "warm", ["fadein"]))
    S.append(shot("V_bridge", 2.3, 4.6, (0.5, 0.5, 1.0), (0.5, 0.5, 1.08), "warm", ["leak"], vt=9.0))
    S.append(vshot("V_beach", 4.6, 6.2, 4.0, 1.25, 1.4, "warm", ["leak"]))
    S.append(shot("POLA", 6.2, 7.25, (0.5, 0.5, 1.0), (0.5, 0.5, 1.75), "warm", ["leak", "zoomout"]))
    S.append(vshot("V_bridge", 7.25, B(0), 17.0, 1.0, 1.2, "punch", ["zoomin", "bloom"]))
    # --- beat section 1 ----------------------------------------------------
    S.append(shot("boy_arms", B(0), B(1), (0.37, 0.55, 1.15), (0.37, 0.55, 1.25), "punch",
                  ["punch", "flash", "stars"]))
    S.append(vshot("V_boat", B(1), B(3), 0.0, 1.0, 1.1, "punch", ["shake", "rgb"]))
    S.append(shot("snorkel", B(3), B(4), (0.55, 0.3, 1.2), (0.55, 0.3, 1.4), "punch", ["zoomin", "bloom"]))
    S.append(vshot("V_beach", B(4), B(6), 7.0, 1.4, 1.55, "punch", ["whip"]))
    S.append(shot("shop", B(6), B(7), (0.62, 0.55, 1.4), (0.62, 0.55, 1.5), "punch", ["punch", "rgb"]))
    S.append(shot("seafan", B(7), B(8), (0.62, 0.55, 1.1), (0.62, 0.55, 1.2), "punch", ["wipe", "stars"]))
    S.append(vshot("V_bridge", B(8), B(10), 0.5, 1.1, 1.2, "punch", ["flash", "bloom"]))
    S.append(shot("food", B(10), B(11), (0.42, 0.5, 1.4), (0.42, 0.5, 1.55), "punch", ["shake"]))
    S.append(shot("coral_fish", B(11), B(12), (0.5, 0.45, 1.0), (0.5, 0.45, 1.1), "punch", ["whip", "bloom"]))
    S.append(vshot("V_boat", B(12), B(14), 6.0, 1.0, 1.05, "punch", ["wipe", "punch"]))
    rapid = [("restaurant", (0.6, 0.25, 2.0)), ("coral1", (0.5, 0.5, 1.2)),
             ("bridge_wide", (0.18, 0.4, 1.3)), ("coral_purple", (0.55, 0.5, 1.2)),
             ("selfie", (0.66, 0.5, 1.6))]
    ts = [B(14), 18.28, 18.45, 18.70, 19.05, 19.25]
    for (src, c), a, b in zip(rapid, ts, ts[1:]):
        S.append(shot(src, a, b, c, (c[0], c[1], c[2] * 1.06), "punch", ["flash", "punch"]))
    # --- dialogue ----------------------------------------------------------
    S.append(shot("mom_sea", 19.25, 20.30, (0.38, 0.5, 1.5), (0.38, 0.5, 1.65), "dark", ["flash"]))
    S.append(vshot("V_bridge", 20.30, 20.90, 20.0, 1.3, 1.4, "punch", ["bloom"]))
    S.append(vshot("V_beach", 20.90, 22.0, 9.0, 1.3, 1.4, "dark"))
    S.append(shot("mom_bridge", 22.0, 23.27, (0.42, 0.5, 1.05), (0.42, 0.45, 1.2), "punch",
                  ["xfade", "bloom"]))
    S.append(shot("drone", 23.27, B(24), (0.36, 0.5, 1.3), (0.36, 0.5, 1.45), "punch", ["flash"]))
    # --- beat section 2 ----------------------------------------------------
    S.append(vshot("V_bridge", B(24), B(25), 21.5, 1.25, 1.35, "punch", ["punch", "flash"]))
    S.append(shot("drone", B(25), B(26), (0.45, 0.5, 1.0), (0.3, 0.5, 1.0), "punch", ["whip"]))
    S.append(vshot("V_beach", B(26), B(27), 11.0, 1.35, 1.45, "punch", ["shake"]))
    S.append(shot("mom_bridge", B(27), B(28), (0.42, 0.5, 1.0), (0.42, 0.5, 1.1), "punch", ["wipe", "stars"]))
    S.append(shot("boy_arms", B(28), B(29), (0.36, 0.5, 1.0), (0.36, 0.5, 1.08), "punch", ["bloom", "flash"]))
    S.append(vshot("V_boat", B(29), B(30), 8.0, 1.0, 1.05, "punch", ["rgb", "punch"]))
    rapid2 = [("boy_bridge", (0.43, 0.38, 1.9)), ("mom_walk", (0.44, 0.42, 1.8)),
              ("snorkel", (0.55, 0.25, 1.6)), ("shop", (0.62, 0.55, 1.7)),
              ("mom_sea", (0.38, 0.52, 1.7)), ("seafan", (0.6, 0.5, 1.4)),
              ("selfie", (0.86, 0.45, 1.8)), ("coral_fish", (0.6, 0.4, 1.4)),
              ("boy_arms", (0.37, 0.6, 1.9))]
    t0, step = B(30), (29.0 - B(30)) / len(rapid2)
    for src, c in rapid2:
        S.append(shot(src, t0, t0 + step, c, (c[0], c[1], c[2] * 1.06), "punch", ["flash", "punch"]))
        t0 += step
    return S


SRC, TXT, VID = {}, {}, {}
TL, TIMES = None, None


def load_photo(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return im.filter(ImageFilter.UnsharpMask(radius=2.5, percent=140, threshold=2))


def make_polaroid(photo, caption):
    """Polaroid print on a warm blurred backdrop, as a 1.5x canvas."""
    cw, ch = int(W * 1.5), int(H * 1.5)
    s = max(cw / photo.width, ch / photo.height)
    bg = photo.resize((int(photo.width * s) + 1, int(photo.height * s) + 1), Image.BILINEAR)
    bg = bg.crop((0, 0, cw, ch)).filter(ImageFilter.GaussianBlur(40))
    bg = np.asarray(bg, np.float32) / 255 * np.array([1.0, 0.8, 0.55]) * 0.7
    pw = int(cw * 0.78)
    ph = int(pw * photo.height / photo.width)
    m, mb = 40, 230
    card = Image.new("RGB", (pw + 2 * m, ph + m + mb), (246, 243, 236))
    card.paste(photo.resize((pw, ph), Image.LANCZOS), (m, m))
    ImageDraw.Draw(card).text((m + 30, ph + m + 30), caption,
                              font=ImageFont.truetype(E.SCRIPT_FONT, 150), fill=(40, 45, 70))
    cmask = Image.new("L", card.size, 255).rotate(-4, expand=True)
    card = card.rotate(-4, resample=Image.BICUBIC, expand=True)
    canvas = E.to_pil(np.clip(bg, 0, 1))
    x0, y0 = (cw - card.width) // 2, (ch - card.height) // 2
    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(cmask, (x0 + 30, y0 + 40))
    shadow = shadow.filter(ImageFilter.GaussianBlur(35)).point(lambda v: v * 0.7)
    canvas = Image.composite(Image.new("RGB", canvas.size), canvas, shadow)
    canvas.paste(card, (x0, y0), cmask)
    return canvas, ((x0 + card.width / 2) / cw, (y0 + m + ph / 2 + 20) / ch)


def video_ranges(tl):
    """Per clip, the [start, end] seconds the timeline needs."""
    rng = {}
    for s in tl:
        if s["src"].startswith("V_"):
            a, b = s["vt"], s["vt"] + (s["t1"] - s["t0"]) * s["spd"] + 0.1
            lo, hi = rng.get(s["src"], (a, b))
            rng[s["src"]] = (min(lo, a), max(hi, b))
    return rng


def extract_frames(media, cache):
    """Decode (and sharpen) only the clip ranges the timeline uses."""
    out = {}
    for key, (a, b) in video_ranges(build_timeline()).items():
        d = os.path.join(cache, key)
        os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                        "-i", os.path.join(media, MEDIA[key]),
                        "-vf", f"fps={FPS},unsharp=5:5:1.0:5:5:0", "-q:v", "2",
                        os.path.join(d, "%05d.jpg")], check=True)
        out[key] = (a, d, len(os.listdir(d)))
    return out


def init(media, times, frames):
    global TL, TIMES
    for k, f in MEDIA.items():
        if not k.startswith("V_"):
            SRC[k] = load_photo(os.path.join(media, f))
    VID.update(frames)
    still = source({"src": "V_bridge", "vt": 17.0, "spd": 1.0}, 0.0)
    SRC["POLA"], pc = make_polaroid(still, POLA_CAPTION)
    TL = build_timeline()
    for s in TL:
        if s["src"] == "POLA":
            s["c0"] = (pc[0], pc[1] - 0.03) + s["c0"][2:]
            s["c1"] = (pc[0], pc[1]) + s["c1"][2:]
    TIMES = times
    TXT["hero"] = E.gold_text(HERO_WORD, E.SCRIPT_FONT, 300)
    TXT["gold"] = E.gold_text(GOLD_WORD, E.SCRIPT_FONT, 300)
    TXT["logo"] = E.gold_text(LOGO, E.SCRIPT_FONT, 230)


def source(s, lt):
    if not s["src"].startswith("V_"):
        return SRC[s["src"]]
    start, d, n = VID[s["src"]]
    idx = int(round((s["vt"] + lt * s["spd"] - start) * FPS))
    return Image.open(os.path.join(d, f"{min(max(idx, 0), n - 1) + 1:05d}.jpg")).convert("RGB")


def render_shot(s, T, frame_no):
    lt = T - s["t0"]
    d = s["t1"] - s["t0"]
    fx = s["fx"]
    u = lt / d if "whip" in fx else ease(lt / d)
    c = [a + (b - a) * u for a, b in zip(s["c0"], s["c1"])]
    if s["track"]:
        c[0], c[1] = tracked(s["src"], s["vt"] + lt * s["spd"])
    if "punch" in fx:
        c[2] *= 1 + 0.18 * (1 - ease_out(lt / 0.3))
    if "shake" in fx:
        rng = np.random.default_rng(frame_no)
        amp = math.exp(-lt / 0.25)
        c[0] += rng.normal() * 0.012 * amp
        c[1] += rng.normal() * 0.008 * amp
        c[3] += rng.normal() * 1.5 * amp
        c[2] *= 1 + 0.03 * amp
    if abs(c[3]) > 0.01:
        c[2] *= 1.06
    f = E.grade(E.view(source(s, lt), *c), s["grade"])
    if "bw" in fx:
        f = E.bw(f)
    if "bloom" in fx:
        f = E.bloom(f, 0.4)
    if "stars" in fx:
        f = E.stars(f, math.sin(math.pi * min(lt / 0.5, 1)) * 0.9)
    if "zoomin" in fx:
        f = E.radial_blur(f, 0.25 * (1 - ease_out(lt / 0.3)))
    if "zoomout" in fx and lt > d - 0.35:
        f = E.radial_blur(f, 0.35 * ease((lt - (d - 0.35)) / 0.35))
    if "rgb" in fx:
        f = E.rgb_split(f, 22 * math.exp(-lt / 0.1))
    if "flash" in fx:
        f = 1 - (1 - f) * (1 - 0.9 * math.exp(-lt / 0.07))
    if s["grade"] == "warm":
        f = E.particles(f, T, n=40, strength=0.6)
    return f


def banner(f, T):
    """Top title strip, as in the reference's opening."""
    a = min(T / 0.3, 1) * (1 - ease((T - 4.2) / 0.4))
    if a <= 0:
        return f
    bh = 230
    f = f.copy()
    f[:bh] = f[:bh] * (1 - a) + a
    try:
        font = ImageFont.truetype(BANNER_FONT, 74)
    except OSError:
        font = ImageFont.truetype(E.BOLD_FONT, 74)
    m = Image.new("L", (W, bh), 0)
    d = ImageDraw.Draw(m)
    for i, line in enumerate(BANNER):
        w = d.textlength(line, font=font)
        d.text(((W - w) / 2, 40 + i * 80), line, font=font, fill=255)
    t = np.asarray(m, np.float32)[..., None] / 255 * a
    f[:bh] = f[:bh] * (1 - t)
    return f


def overlays(f, T):
    w = TIMES
    if T < 4.6:
        f = banner(f, T)
    if 7.9 <= T < B(0):
        f = E.write_on(f, TXT["hero"], (T - 7.95) / 1.0, W / 2, H * 0.72)
    # dialogue captions, synced to the replacement voice
    if w["d1"]["t0"] - 0.05 <= T < 20.30:
        f = E.pop_text(f, CAP1[0], E.BOLD_FONT, 150, H * 0.47, (1, 1, 1), T - w["d1"]["words"][0])
        f = E.pop_text(f, CAP1[1], E.BOLD_FONT, 300, H * 0.47 + 230, (0.92, 0.12, 0.12),
                       T - w["d1"]["words"][-1])
    if w["d2"]["t0"] - 0.05 <= T < 22.25:
        a = 1 - ease((T - 22.0) / 0.25)
        g = E.pop_text(f, CAP2[0], E.BOLD_FONT, 90, H * 0.47, (1, 1, 1), T - w["d2"]["words"][0])
        g = E.pop_text(g, CAP2[1], E.BOLD_FONT, 230, H * 0.47 + 180, (0.92, 0.12, 0.12),
                       T - w["d2"]["words"][-1])
        f = f * (1 - a) + g * a
    if w["d3"]["t0"] + 0.1 <= T < B(24):
        p = (T - (w["d3"]["t0"] + 0.1)) / 0.9
        f = E.write_on(f, TXT["gold"], p, W / 2, H * 0.3, alpha=1 - ease((T - 23.75) / 0.3))
    return f


def render(i):
    T = i / FPS
    if T >= 29.0:  # end card
        f = np.zeros((H, W, 3), np.float32) + np.array([0.035, 0.025, 0.012])
        f = E.particles(f, T)
        alpha = 1 - ease((T - 30.4) / 1.5)
        f = E.write_on(f, TXT["logo"], (T - 28.95) / 0.45, W / 2, H * 0.47, alpha=alpha)
        return E.to_pil(f).tobytes()

    idx = max(j for j, s in enumerate(TL) if s["t0"] <= T)
    s = TL[idx]
    f = render_shot(s, T, i)
    lt = T - s["t0"]
    prev = TL[idx - 1] if idx > 0 else None
    if "fadein" in s["fx"]:
        f = f * ease(lt / 0.5)
    if prev and "leak" in s["fx"] and lt < 0.5:
        p = lt / 0.5
        f = E.light_leak(render_shot(prev, T, i) * (1 - ease(p)) + f * ease(p), p)
    if prev and "xfade" in s["fx"] and lt < 0.35:
        p = ease(lt / 0.35)
        f = render_shot(prev, T, i) * (1 - p) + f * p
    if prev and "whip" in s["fx"] and lt < 0.15:
        f = E.whip(render_shot(prev, T, i), f, lt / 0.15)
    if prev and "wipe" in s["fx"] and lt < 0.22:
        f = E.diag_wipe(render_shot(prev, T, i), f, lt / 0.22)
    return E.to_pil(overlays(f, T)).tobytes()


def main():
    out, audio, times_path, media = sys.argv[1:5]
    only = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
    times = json.load(open(times_path))
    cache = tempfile.mkdtemp(prefix="edit2_frames_")
    frames = extract_frames(media, cache)
    if only:  # preview: render chosen timestamps into a contact sheet
        init(media, times, frames)
        ims = [Image.frombytes("RGB", (W, H), render(int(t * FPS))).resize((216, 384)) for t in only]
        cols = min(len(ims), 12)
        sheet = Image.new("RGB", (216 * cols, 384 * ((len(ims) + cols - 1) // cols)))
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % cols) * 216, (k // cols) * 384))
        sheet.save(out)
        return
    n = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", audio, "-vf", "unsharp=5:5:0.5:5:5:0,vignette=PI/5,noise=alls=2:allf=t",
           "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init, initargs=(media, times, frames)) as pool:
        for i, buf in enumerate(pool.imap(render, range(n), chunksize=4)):
            enc.stdin.write(buf)
            if i % 120 == 0:
                print(f"frame {i}/{n}", file=sys.stderr)
    enc.stdin.close()
    sys.exit(enc.wait())


if __name__ == "__main__":
    main()
