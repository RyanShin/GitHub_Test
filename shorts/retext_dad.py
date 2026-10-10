#!/usr/bin/env python3
"""Edit a short "toggle list" clip in place, frame by frame, keeping its exact
timing, camera moves, sparks and toggle switches:

  * black out the faded background person in the opening frames
  * replace each row's text with a Korean caption in the same colours,
    tracked to the original word's position / size / angle / blur / opacity
  * swap the closing photo for another one with the original's dissolve,
    zoom-in and shake

usage: retext_dad.py IN.mp4 PHOTO.jpg OUT.mp4 WORKDIR
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "Pretendard-Black.otf")

ORDER = ["good", "loving", "caring", "savage"]          # top -> bottom on screen
# row -> (coloured word, second word, colour gradient 1, colour gradient 2)  [RGB]
ROWS = {
    "good": ("구독자", "10명", ((150, 240, 95), (60, 185, 50)), ((255, 255, 255), (215, 218, 228))),
    "loving": ("구독자", "50명", ((235, 120, 225), (165, 80, 215)), ((255, 255, 255), (215, 218, 228))),
    "caring": ("구독자", "75명", ((110, 235, 255), (20, 165, 230)), ((255, 255, 255), (215, 218, 228))),
    "savage": ("구독자", "100명", ((255, 200, 90), (255, 75, 25)), ((255, 200, 90), (255, 75, 25))),
}
# full caption width / coloured-word width, measured on sharp frames
EXTEND = {"good": 1.98, "loving": 1.75, "caring": 1.68, "savage": 1.85}
HUE = {"good": [(18, 85)], "caring": [(86, 108)], "loving": [(125, 172)], "savage": [(0, 22)]}
PERSON_FRAMES = 25          # frames where the faded background person shows
PERSON_FADE = 4
DISSOLVE = (185, 192)       # photo dissolves in over these frames
NO_NEW_ROWS_AFTER = 176     # only keep tracking existing rows past this frame


# --------------------------------------------------------------------------
# detection + tracking

def norm_rect(rect):
    (cx, cy), (w, h), ang = rect
    if h > w:
        w, h, ang = h, w, ang - 90
    ang = (ang + 90) % 180 - 90
    return [cx, cy, w, h, ang]


def candidates(hsv, rng):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for a, b in rng:
        m |= cv2.inRange(hsv, (a, 90, 90), (b, 255, 255))
    mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mc)
    Hh, Ww = hsv.shape[:2]
    out = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        clip = y <= 2 or y + h >= Hh - 2
        if a < 300 or h < (8 if clip else 18):
            continue
        sel = (lab == j) & (m > 0)
        pts = np.column_stack(np.where(sel))[:, ::-1].astype(np.float32)
        if len(pts) < 40:
            continue
        r = norm_rect(cv2.minAreaRect(pts))
        if abs(r[4]) > 25:
            continue
        # core letter box: percentiles along the row axis drop thin glow / fringe
        a = np.radians(r[4])
        u = pts[:, 0] * np.cos(a) + pts[:, 1] * np.sin(a)
        v = -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)
        u0, u1 = np.percentile(u, [1, 99])
        v0, v1 = np.percentile(v, [6, 94])
        uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
        r = [uc * np.cos(a) - vc * np.sin(a), uc * np.sin(a) + vc * np.cos(a), u1 - u0, (v1 - v0) * 1.08, r[4]]
        if not clip and (r[2] < 1.6 * r[3] or r[2] < 110):
            continue
        if clip and (r[2] < 1.2 * r[3] or r[3] > 140 or r[2] < 80):
            continue
        out.append(dict(r=r, box=(x, y, w, h), a=int(a), clip=clip,
                        v=float(np.percentile(hsv[..., 2][sel], 90))))
    return out


def overlap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return ix * iy / max(1, min(w1 * h1, w2 * h2))


def extent(c):
    """Box covering a detected word, the rest of its caption and its toggle."""
    x, y, w, h = c["box"]
    return (x - 10, y - 10, int(w * 2.2 + 3.5 * h) + 20, h + 20)


def track(frames):
    n = len(frames)
    det = [dict() for _ in range(n)]
    last = {}
    for i, im in enumerate(frames):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        C = {k: candidates(hsv, HUE[k]) for k in ORDER}
        chosen = {}
        for k in ["loving", "caring", "good", "savage"]:
            ki = ORDER.index(k)
            cs = [c for c in C[k]
                  if all(overlap(c["box"], extent(o)) < 0.25 for o in chosen.values())
                  and all((ORDER.index(o) < ki) == (oc["r"][1] < c["r"][1]) for o, oc in chosen.items())]
            pick = None
            if k in last and i - last[k][0] <= 3:
                _, pr = last[k]
                gap = i - last[k][0]
                best = None
                for c in cs:
                    d = np.hypot(c["r"][0] - pr["r"][0], c["r"][1] - pr["r"][1])
                    ok_size = 0.4 < c["r"][3] / pr["r"][3] < 2.5 or c["clip"] or pr["clip"]
                    if ok_size and d < max(1.5 * pr["r"][3], 80) * gap and (best is None or d < best[0]):
                        best = (d, c)
                pick = best[1] if best else None
            lost_recently = k in last and i - last[k][0] <= 6
            if pick is None and i <= NO_NEW_ROWS_AFTER and cs and not lost_recently:
                # the word is the leftmost sizeable blob of its colour (toggles sit right)
                big = max(c["a"] for c in cs)
                pick = min((c for c in cs if c["a"] >= 0.4 * big), key=lambda c: c["box"][0])
            if pick is not None:
                chosen[k] = pick
                last[k] = (i, pick)
        det[i] = chosen
    return det


def to_full(k, c):
    """Coloured-word rect -> whole-caption rect (extend along the row axis)."""
    cx, cy, w, h, ang = c["r"]
    if k == "savage":
        if w > 6.6 * h:          # blob swallowed the red toggle
            w2 = 6.5 * h
        elif w > 5.0 * h:        # SAVAGE + DAD already in one blob
            w2 = w
        else:
            w2 = w * EXTEND[k]
    else:
        w2 = w * EXTEND[k]
    a = np.radians(ang)
    dx = (w2 - w) / 2
    return [cx + dx * np.cos(a), cy + dx * np.sin(a), w2, h, ang]


def smooth(det, n, H):
    """Per row: rebuild clipped / odd-sized boxes from neighbours, fill short gaps."""
    tracks = {}
    for k in ORDER:
        R = [None] * n
        for i in range(n):
            c = det[i].get(k)
            if c:
                R[i] = dict(full=to_full(k, c), v=c["v"], clip=c["clip"])
        clean = [i for i in range(n) if R[i] and not R[i]["clip"]]
        if clean:
            for i in clean:
                cx, cy, w, h, ang = det[i][k]["r"]
                near = sorted(clean, key=lambda j: abs(j - i))[1:7]
                wm = float(np.median([det[j][k]["r"][2] for j in near])) if near else w
                if w > 1.3 * wm:                 # blob merged with a toggle / fringe
                    w2 = wm
                    a = np.radians(ang)
                    shift = (w - w2) / 2
                    c = dict(det[i][k], r=[cx - shift * np.cos(a), cy - shift * np.sin(a), w2, h, ang])
                    R[i]["full"] = to_full(k, c)
        for i in range(n):
            if R[i] is None or not clean:
                continue
            near = sorted(clean, key=lambda j: abs(j - i))[:6]
            hm = float(np.median([R[j]["full"][3] for j in near]))
            f = R[i]["full"]
            if R[i]["clip"] or not (0.6 * hm < f[3] < 1.6 * hm):
                ref = R[near[0]]["full"]
                top, bot = f[1] - f[3] / 2, f[1] + f[3] / 2
                cy = f[1]
                if top <= 4:
                    cy = bot - ref[3] / 2
                elif bot >= H - 4:
                    cy = top + ref[3] / 2
                a = np.radians(ref[4])
                left = f[0] - f[2] / 2 * np.cos(a)
                R[i]["full"] = [left + ref[2] / 2 * np.cos(a), cy, ref[2], ref[3], ref[4]]
        idx = [i for i in range(n) if R[i]]
        for a, b in zip(idx, idx[1:]):
            if 1 < b - a <= 10:
                for i in range(a + 1, b):
                    t = (i - a) / (b - a)
                    fa, fb = np.array(R[a]["full"]), np.array(R[b]["full"])
                    R[i] = dict(full=list(fa + (fb - fa) * t), v=R[a]["v"] + (R[b]["v"] - R[a]["v"]) * t,
                                clip=True)
        if k == "savage":   # over the closing photo the row holds still
            lastgood = max((i for i in range(min(n, DISSOLVE[0])) if R[i]), default=None)
            if lastgood is not None:
                for i in range(DISSOLVE[0], n):
                    R[i] = dict(R[lastgood], clip=True)
        tracks[k] = R
    return tracks


# --------------------------------------------------------------------------
# caption rendering

def caption_layer(k, ref_h=220):
    """Canonical caption as BGRA float image and its content box."""
    w1, w2, c1, c2 = ROWS[k]
    font = ImageFont.truetype(FONT, ref_h)
    gap = int(ref_h * 0.28)
    b1, b2 = font.getbbox(w1), font.getbbox(w2)
    width = (b1[2] - b1[0]) + gap + (b2[2] - b2[0])
    top = min(b1[1], b2[1])
    height = max(b1[3], b2[3]) - top
    pad = int(ref_h * 0.6)
    W, H = width + 2 * pad, height + 2 * pad
    out = np.zeros((H, W, 4), np.float32)
    x = pad
    for word, bb, (ct, cb) in ((w1, b1, c1), (w2, b2, c2)):
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).text((x - bb[0], pad - top), word, font=font, fill=255,
                               stroke_width=max(2, ref_h // 40), stroke_fill=255)
        a = np.asarray(m, np.float32) / 255
        yy = np.clip((np.arange(H) - pad) / max(height, 1), 0, 1)[:, None, None]
        col = (np.array(ct[::-1]) * (1 - yy) + np.array(cb[::-1]) * yy) / 255   # RGB -> BGR
        rim = np.clip(cv2.dilate(a, np.ones((5, 5), np.uint8)) - a, 0, 1)
        al = np.clip(a + rim, 0, 1)
        rgb = col * a[..., None] + col * 0.35 * rim[..., None]
        out[..., :3] = out[..., :3] * (1 - al[..., None]) + rgb
        out[..., 3] = np.maximum(out[..., 3], al)
        x += (bb[2] - bb[0]) + gap
    sh = 0.10  # slight italic like the source font
    M = np.float32([[1, -sh, sh * H / 2], [0, 1, 0]])
    out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR)
    return dict(img=out, box=(pad, pad, width, height), glow=np.array(c1[1][::-1]) / 255)


def draw_caption(f, cap, full, alpha, blur_sigma, motion):
    """f: float BGR 0..1."""
    cx, cy, w, h, ang = full
    img, (bx, by, bw, bh) = cap["img"], cap["box"]
    # detected boxes include glow / fringe: real letters are ~72% of the box height
    s = min(w / bw, TEXT_H * h / bh)
    sx = min(w / bw, s * 1.25)          # widen toward the original caption width
    a = np.radians(ang)
    ca, sa = np.cos(a), np.sin(a)
    # left-align on the original word so the toggle stays clear
    lx = cx - (w / 2) * ca + (bw * sx / 2) * ca
    ly = cy - (w / 2) * sa + (bw * sx / 2) * sa
    M = np.float32([[sx * ca, -s * sa, 0], [sx * sa, s * ca, 0]])
    M[:, 2] = np.array([lx, ly]) - M[:, :2] @ np.array([bx + bw / 2, by + bh / 2])
    H, W = f.shape[:2]
    lay = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    if blur_sigma > 0.4:
        lay = cv2.GaussianBlur(lay, (0, 0), blur_sigma)
    mv = float(np.hypot(*motion))
    if mv > 2:
        L = int(min(mv, 120)) | 1
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1
        R = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5),
                                    -np.degrees(np.arctan2(motion[1], motion[0])), 1)
        k = cv2.warpAffine(k, R, (L, L))
        lay = cv2.filter2D(lay, -1, k / max(k.sum(), 1e-6))
    al = lay[..., 3:4] * alpha
    pre = np.where(lay[..., 3:4] > 1e-4, lay[..., :3] / np.maximum(lay[..., 3:4], 1e-4), 0)
    ab = max(2, int(h * 0.05))             # chromatic fringe like the source
    pre = pre.copy()
    pre[..., 2] = np.roll(pre[..., 2], -ab, axis=1)
    pre[..., 0] = np.roll(pre[..., 0], ab, axis=1)
    glow = cv2.GaussianBlur(lay[..., 3], (0, 0), max(h * 0.22, 3))[..., None] * cap["glow"] * 0.9 * alpha
    f = 1 - (1 - f) * (1 - np.clip(glow, 0, 1))
    return f * (1 - al) + np.clip(pre, 0, 1) * al


def rect_mask(shape, full, grow_w=0.0, grow_h=0.0, feather=0):
    cx, cy, w, h, ang = full
    m = np.zeros(shape[:2], np.float32)
    box = cv2.boxPoints(((cx, cy), (w + grow_w * h * 2, h + grow_h * h * 2), ang))
    cv2.fillConvexPoly(m, np.int32(box), 1.0)
    return cv2.GaussianBlur(m, (0, 0), feather) if feather else m


TEXT_H = 1.0
TOGGLE_GAP = {"good": 1.85, "loving": 1.85, "caring": 1.85, "savage": 1.15}


def toggle_mask(im, full, k, color_only=False):
    """Bright pixels of the toggle switch to the right of a caption."""
    cx, cy, w, h, ang = full
    a = np.radians(ang)
    d = w / 2 + TOGGLE_GAP[k] * h
    r = [cx + d * np.cos(a), cy + d * np.sin(a), 2.9 * h, 1.6 * h, ang]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    if color_only:
        # red pill + white knob only (the background behind it is a bright photo)
        r = [cx + (w / 2 + 1.0 * h) * np.cos(a), cy + (w / 2 + 1.0 * h) * np.sin(a), 4.2 * h, 1.6 * h, ang]
        m = rect_mask(im.shape, r)
        red = cv2.inRange(hsv, (0, 120, 120), (7, 255, 255)) | cv2.inRange(hsv, (172, 120, 120), (180, 255, 255))
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        near_red = cv2.dilate(red, np.ones((int(h * 0.5) | 1, int(h * 0.5) | 1), np.uint8))
        white = cv2.inRange(hsv, (0, 0, 225), (180, 45, 255)) & near_red
        pill = cv2.morphologyEx(red | white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats((pill > 0).astype(np.uint8) * (m > 0).astype(np.uint8))
        keep = np.zeros(pill.shape, np.float32)
        if n > 1:
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            keep[lab == j] = 1
            keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return cv2.GaussianBlur(keep, (0, 0), 1.5)
    m = rect_mask(im.shape, r)
    v = hsv[..., 2].astype(np.float32)
    bright = np.clip((v - 140) / 50, 0, 1)
    # toggles are green / white / grey: never protect orange caption letters
    orange = cv2.inRange(hsv, (5, 90, 0), (28, 255, 255)).astype(np.float32) / 255
    return cv2.GaussianBlur(m * bright * (1 - orange), (0, 0), 1.5)


def erase(f, mask):
    """f float BGR; fill mask with a smooth inpaint of the surroundings."""
    im8 = (f * 255).astype(np.uint8)
    small = cv2.resize(im8, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    ms = (cv2.resize(mask, (small.shape[1], small.shape[0])) > 0.05).astype(np.uint8) * 255
    fill = cv2.inpaint(small, ms, 6, cv2.INPAINT_TELEA)
    fill = cv2.resize(cv2.GaussianBlur(fill, (0, 0), 2), (f.shape[1], f.shape[0])).astype(np.float32) / 255
    m = mask[..., None]
    return f * (1 - m) + fill * m


def sharpness(im, full):
    m = rect_mask(im.shape, full) > 0
    if m.sum() < 50:
        return 0.0
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F)[m].var())


# --------------------------------------------------------------------------
# closing photo

def photo_motion(frames, start):
    """Homographies mapping the final frame onto each closing frame (shake)."""
    last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    m = np.full(last.shape, 255, np.uint8)
    m[int(last.shape[0] * 0.4):int(last.shape[0] * 0.6)] = 0      # skip the caption row
    sift = cv2.SIFT_create(4000)
    kl, dl = sift.detectAndCompute(last, m)
    bf = cv2.BFMatcher()
    Hs = {}
    for i in range(start, len(frames)):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        k, d = sift.detectAndCompute(g, m)
        Hm = None
        if d is not None and len(k) > 8:
            ms = [p[0] for p in bf.knnMatch(dl, d, k=2) if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
            if len(ms) >= 20:
                Hm, inl = cv2.findHomography(np.float32([kl[p.queryIdx].pt for p in ms]),
                                             np.float32([k[p.trainIdx].pt for p in ms]), cv2.RANSAC, 3)
                if Hm is None or inl.sum() < 20:
                    Hm = None
        Hs[i] = Hm if Hm is not None else np.eye(3)
    return Hs


def dissolve_weights(frames, a, b):
    base = frames[a - 1].astype(np.float32)
    full = frames[b + 2].astype(np.float32)
    H, W = base.shape[:2]
    reg = (slice(0, H), slice(int(W * 0.83), W))
    den = (full[reg] - base[reg]).ravel()
    w = {}
    for i in range(a, b):
        num = (frames[i].astype(np.float32)[reg] - base[reg]).ravel()
        w[i] = float(np.clip(num @ den / (den @ den), 0, 1))
    return w


def zoomed(img, s):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------

def main():
    src, photo_path, out, work = sys.argv[1:5]
    fdir, odir = os.path.join(work, "src_frames"), os.path.join(work, "out_frames")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(odir, exist_ok=True)
    for f in os.listdir(odir):
        os.remove(os.path.join(odir, f))
    if not os.listdir(fdir):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-fps_mode", "passthrough",
                        os.path.join(fdir, "%05d.png")], check=True)
    pts = [float(p) for p in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout.split()]
    names = sorted(os.listdir(fdir))
    frames = [cv2.imread(os.path.join(fdir, f)) for f in names]
    n = len(frames)
    H, W = frames[0].shape[:2]

    tracks = smooth(track(frames), n, H)
    caps = {k: caption_layer(k) for k in ORDER}
    vref = {k: np.percentile([t["v"] for t in tracks[k] if t], 90) for k in ORDER}
    sharp = {k: [sharpness(frames[i], tracks[k][i]["full"]) if tracks[k][i] else 0 for i in range(n)]
             for k in ORDER}
    sref = {k: np.percentile([s for s in sharp[k] if s > 0] or [1], 90) for k in ORDER}

    newp = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    newp = ImageOps.fit(newp, (W, H), Image.LANCZOS, centering=(0.5, 0.4))
    newp = cv2.cvtColor(np.asarray(newp), cv2.COLOR_RGB2BGR).astype(np.float32) / 255
    d0, d1 = DISSOLVE
    wts = dissolve_weights(frames, d0, d1)
    Hs = photo_motion(frames, d1)
    clean_before = None

    for i in range(n):
        orig = frames[i]
        f = orig.astype(np.float32) / 255
        rows = {k: tracks[k][i] for k in ORDER if tracks[k][i]}
        toggles = np.zeros((H, W), np.float32)
        for k, t in rows.items():
            toggles = np.maximum(toggles, toggle_mask(orig, t["full"], k,
                                                      color_only=(k == "savage" and i >= DISSOLVE[0] - 15)))

        # black out the faded background person (keep toggles + sparks)
        if i < PERSON_FRAMES + PERSON_FADE:
            hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
            sparks = cv2.inRange(hsv, (0, 120, 150), (25, 255, 255)).astype(np.float32) / 255
            keep = np.maximum(toggles, sparks * 0.9)[..., None]
            t = 1.0 if i < PERSON_FRAMES else 1 - (i - PERSON_FRAMES + 1) / (PERSON_FADE + 1)
            f = f * (1 - t) + f * keep * t

        # erase the original captions
        m = np.zeros((H, W), np.float32)
        for k, t in rows.items():
            cx, cy, w, h, ang = t["full"]
            a = np.radians(ang)
            g = (0.6 if k == "savage" else 0.25) * h   # room for the trailing "D"
            wide = [cx + g / 2 * np.cos(a), cy + g / 2 * np.sin(a), w + g, h, ang]
            m = np.maximum(m, rect_mask(f.shape, wide, 0.1, 0.5, 5))
        if m.max() > 0:
            f = erase(f, np.clip(m * 1.3, 0, 1) * (1 - toggles))
        if i == d0 - 1:
            clean_before = f.copy()

        # closing photo: dissolve (with the original's zoom-in), then hold with shake
        if i >= d0 and clean_before is not None:
            tog = orig.astype(np.float32) / 255
            if i < d1:
                w = wts[i]
                ph = zoomed(newp, 0.88 + 0.12 * w)
                f = clean_before * (1 - w) + ph * w
            else:
                f = cv2.warpPerspective(newp, Hs[i], (W, H), borderMode=cv2.BORDER_REFLECT)
            tm = toggles[..., None]
            f = f * (1 - tm) + tog * tm

        # draw the Korean captions with matched opacity / blur / motion
        for k, t in rows.items():
            alpha = float(np.clip(t["v"] / vref[k], 0.15, 1.0)) ** 1.5
            s = sharp[k][i] / sref[k]
            sigma = float(np.clip((1 - min(s, 1)) * t["full"][3] * 0.06, 0, 12))
            p, q = (tracks[k][i - 1] if i else None), (tracks[k][i + 1] if i + 1 < n else None)
            motion = ((q["full"][0] - p["full"][0]) / 4, (q["full"][1] - p["full"][1]) / 4) if p and q else (0, 0)
            f = draw_caption(f, caps[k], t["full"], alpha, sigma, motion)
        cv2.imwrite(os.path.join(odir, names[i]), (np.clip(f, 0, 1) * 255).astype(np.uint8))

    # 60 fps CFR with the source's per-frame durations (first frame ~3 ticks)
    seq = os.path.join(work, "seq")
    os.makedirs(seq, exist_ok=True)
    for f in os.listdir(seq):
        os.remove(os.path.join(seq, f))
    k = 0
    for i, nm in enumerate(names):
        d = (pts[i + 1] - pts[i]) if i + 1 < n else (pts[-1] - pts[-2])
        for _ in range(max(1, round(d * 60))):
            os.symlink(os.path.join(odir, nm), os.path.join(seq, f"{k:05d}.png"))
            k += 1
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "60", "-i", os.path.join(seq, "%05d.png"),
                    "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", out], check=True)


if __name__ == "__main__":
    main()
