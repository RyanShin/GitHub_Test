#!/usr/bin/env python3
"""Replace the spoken lines in a reference track with new TTS lines.

1. Split the track into vocals / instrumental with a UVR MDX-Net model
   (sherpa-onnx).
2. Inside the speech regions only, swap the original mix for the
   instrumental (short crossfades), so the music's own vocal chops survive.
3. Synthesize each new line with Kokoro TTS, deepen it, add a touch of room,
   match the loudness of the original speech, and drop it at the same onset.
4. Write the new mix plus a JSON of line onsets for caption syncing.

usage: make_voice.py REF.wav OUT.wav OUT_TIMES.json MODEL_DIR
  MODEL_DIR must hold UVR-MDX-NET-Voc_FT.onnx, kokoro-v1.0.onnx, voices-v1.0.bin
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

VOICE = "am_onyx"
SPEED = 0.88
# regions of the reference where narration / dialogue is spoken
SPEECH = [(0.0, 7.05), (19.15, 23.5)]
# (key, text, onset seconds, latest end seconds)
LINES = [
    ("n1", "Miyako-jima called.", 0.25, 2.20),
    ("n2", "Time stood still.", 2.30, 3.95),
    ("n3", "Nothing but blue sea, white sand... and us.", 4.05, 6.95),
    ("d1", "In this life.", 19.25, 20.45),
    ("d2", "You either dream it,", 20.80, 22.30),
    ("d3", "or you live it.", 22.45, 23.45),
]
XFADE = 0.06


def separate(x, sr, model):
    import sherpa_onnx as so
    cfg = so.OfflineSourceSeparationConfig(model=so.OfflineSourceSeparationModelConfig(
        uvr=so.OfflineSourceSeparationUvrModelConfig(model=model), num_threads=os.cpu_count()))
    out = so.OfflineSourceSeparation(cfg).process(sr, np.ascontiguousarray(x.T))
    vocals, inst = (np.array(s.data).T for s in out.stems)
    return vocals, inst


def tts(kokoro, text, sr_out, tmp):
    a, sr = kokoro.create(text, voice=VOICE, speed=SPEED, lang="en-us")
    a = np.asarray(a, np.float32)
    nz = np.flatnonzero(np.abs(a) > 0.01 * np.abs(a).max())
    a = a[max(nz[0] - 200, 0):nz[-1] + 2400]
    words = word_onsets(a, sr)
    sf.write(tmp, a, sr)
    # deepen ~6% (keep tempo), warm low end, gentle compression, small room
    af = (f"asetrate={sr}*0.94,aresample={sr_out},atempo={1 / 0.94:.4f},highpass=f=60,"
          "equalizer=f=140:t=q:w=1:g=3,equalizer=f=3500:t=q:w=1.5:g=2,"
          "acompressor=threshold=0.08:ratio=3:attack=5:release=90,"
          "aecho=0.85:0.5:40|85:0.16|0.08")
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", tmp, "-af", af, "-ac", "1",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32), words


def word_onsets(a, sr, gap=0.035):
    """Onsets (s) of voiced chunks separated by short silences."""
    hop = int(sr * 0.01)
    env = np.sqrt(np.convolve(a ** 2, np.ones(hop) / hop, "same"))[::hop]
    on = env > 0.08 * env.max()
    onsets, quiet = [], gap / 0.01
    for i, v in enumerate(on):
        if v and quiet >= gap / 0.01:
            onsets.append(round(i * 0.01, 3))
        quiet = 0 if v else quiet + 1
    return onsets


def fit(a, sr, window):
    """Time-compress (pitch-preserving) if the line overruns its window."""
    dur = len(a) / sr
    if dur <= window:
        return a, 1.0
    r = min(dur / window, 1.3)
    tmp = "/tmp/_fit.wav"
    sf.write(tmp, a, sr)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", tmp, "-af", f"atempo={r:.4f}",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32), r


def active_rms(v):
    e = np.sqrt(np.convolve(v ** 2, np.ones(2048) / 2048, "same"))
    return np.sqrt(np.mean(v[e > 0.25 * e.max()] ** 2))


def main():
    ref, out, times_out, mdir = sys.argv[1:5]
    x, sr = sf.read(ref, dtype="float32")
    vocals, inst = separate(x, sr, os.path.join(mdir, "UVR-MDX-NET-Voc_FT.onnx"))

    # swap in the instrumental inside the speech regions
    n = len(x)
    w = np.zeros(n, np.float32)
    t = np.arange(n) / sr
    for a, b in SPEECH:
        w = np.maximum(w, np.clip(np.minimum((t - a) / XFADE + 1, (b - t) / XFADE), 0, 1))
    mix = x * (1 - w[:, None]) + inst[:n] * w[:, None]

    from kokoro_onnx import Kokoro
    kokoro = Kokoro(os.path.join(mdir, "kokoro-v1.0.onnx"), os.path.join(mdir, "voices-v1.0.bin"))
    voc_mono = vocals.mean(axis=1)
    times = {}
    for key, text, onset, end in LINES:
        a, words = tts(kokoro, text, sr, "/tmp/_tts.wav")
        a, r = fit(a, sr, end - onset)
        region = next((r for r in SPEECH if r[0] <= onset < r[1]), SPEECH[0])
        target = active_rms(voc_mono[int(region[0] * sr):int(region[1] * sr)])
        a = a * (target / (active_rms(a) + 1e-9))
        i = int(onset * sr)
        m = min(len(a), n - i)
        mix[i:i + m] += a[:m, None]
        words = [round(onset + w / r, 3) for w in words]
        times[key] = {"text": text, "t0": onset, "t1": onset + len(a) / sr, "words": words}
        print(f"{key} {onset:6.2f}-{onset + len(a) / sr:6.2f}  {text}  {words}", file=sys.stderr)

    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    sf.write(out, mix, sr, subtype="PCM_16")
    with open(times_out, "w") as f:
        json.dump(times, f, indent=1)


if __name__ == "__main__":
    main()
