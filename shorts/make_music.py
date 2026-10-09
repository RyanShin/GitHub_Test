#!/usr/bin/env python3
"""Synthesize a royalty-free tropical-house style loop (120 BPM) as a WAV file.

usage: make_music.py OUT.wav [DURATION_SEC]
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 120
BEAT = 60.0 / BPM
BAR = BEAT * 4

# I - V - vi - IV in A major (MIDI note numbers, root first)
CHORDS = [
    [57, 61, 64],  # A
    [52, 56, 59],  # E
    [54, 57, 61],  # F#m
    [50, 54, 57],  # D
]
# 8th-note arpeggio pattern: index into chord tones, +12 per octave step
ARP = [0, 1, 2, 1 + 3, 2, 1, 0 + 3, 2]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def fft_filter(x, lo=None, hi=None):
    """Brick-wall-ish band filter with a short cosine roll-off."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    g = np.ones_like(f)
    if lo is not None:
        g *= np.clip((f - lo * 0.7) / (lo * 0.3), 0, 1)
    if hi is not None:
        g *= np.clip((hi * 1.3 - f) / (hi * 0.3), 0, 1)
    return np.fft.irfft(spec * g, len(x))


def add(buf, start, sig, gain=1.0):
    i = int(start * SR)
    if i >= len(buf):
        return
    n = min(len(sig), len(buf) - i)
    buf[i:i + n] += sig[:n] * gain


def env(n, decay, attack=0.002):
    t = np.arange(n) / SR
    a = np.clip(t / attack, 0, 1)
    return a * np.exp(-t / decay)


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    freq = 45 + 85 * np.exp(-t / 0.04)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * env(n, 0.18, 0.001)


def clap(rng):
    n = int(0.25 * SR)
    noise = fft_filter(rng.standard_normal(n), lo=900, hi=4500)
    e = env(n, 0.07, 0.001)
    # three quick re-triggers give the "clap" smear
    for d in (0.010, 0.020):
        k = int(d * SR)
        e[k:] += env(n - k, 0.012, 0.001) * 0.6
    return noise * e / 3.0


def hat(rng):
    n = int(0.08 * SR)
    noise = fft_filter(rng.standard_normal(n), lo=7000)
    return noise * env(n, 0.018, 0.0005) / 2.0


def pluck(f, dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    # marimba-ish: fundamental + bright 4th partial that dies fast
    s = np.sin(2 * np.pi * f * t) * env(n, 0.22)
    s += 0.35 * np.sin(2 * np.pi * f * 4 * t) * env(n, 0.04)
    s += 0.15 * np.sin(2 * np.pi * f * 2 * t) * env(n, 0.10)
    return s


def bass(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    e = np.clip(t / 0.005, 0, 1) * np.clip((dur - t) / 0.02, 0, 1)
    return s * e


def pad(chord, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for m in chord:
        for det in (-0.12, 0.12):
            f = hz(m + det)
            # band-limited-ish saw from 6 harmonics
            for h in range(1, 7):
                s += np.sin(2 * np.pi * f * h * t) / h
    e = np.clip(t / 0.25, 0, 1) * np.clip((dur - t) / 0.2, 0, 1)
    return s * e / 12.0


def main():
    out = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 14.5
    rng = np.random.default_rng(7)
    n = int((dur + 1.0) * SR)
    drums = np.zeros(n)
    music_l = np.zeros(n)
    music_r = np.zeros(n)
    bassline = np.zeros(n)
    pads = np.zeros(n)

    K, C, H = kick(), clap(rng), hat(rng)
    bars = int(np.ceil(dur / BAR)) + 1
    for b in range(bars):
        t0 = b * BAR
        chord = CHORDS[b % len(CHORDS)]
        intro = b == 0  # first bar: no drums, just the hook
        add(pads, t0, pad(chord, BAR))
        for i, a in enumerate(ARP):
            note = chord[a % 3] + 12 * (a // 3) + 12
            s = pluck(hz(note))
            # alternate pan for width
            add(music_l, t0 + i * BEAT / 2, s, 0.55 if i % 2 else 0.35)
            add(music_r, t0 + i * BEAT / 2, s, 0.35 if i % 2 else 0.55)
        if intro:
            continue
        for beat in range(4):
            tb = t0 + beat * BEAT
            add(drums, tb, K, 0.9)
            add(drums, tb + BEAT / 2, H, 0.5)
            if beat in (1, 3):
                add(drums, tb, C, 0.6)
            # off-beat "pumping" bass
            add(bassline, tb + BEAT / 2, bass(hz(chord[0] - 24), BEAT / 2 * 0.9), 0.5)

    # sidechain duck the pad/bass against the kick
    t = np.arange(n) / SR
    since = np.mod(t, BEAT)
    duck = 1 - 0.65 * np.exp(-since / 0.09)
    duck[t < BAR] = 1.0
    pads = fft_filter(pads, hi=2200) * duck
    bassline *= duck

    left = drums + bassline + 0.5 * pads + music_l
    right = drums + bassline + 0.5 * pads + music_r
    st = np.stack([left, right], axis=1)[: int(dur * SR)]

    # fade out over the last second, normalize to -1 dBFS
    fade = int(1.0 * SR)
    st[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
    st *= 10 ** (-1 / 20) / np.max(np.abs(st))
    pcm = (st * 32767).astype("<i2")

    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
