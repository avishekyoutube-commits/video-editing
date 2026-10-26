"""Synthesise background music bed and SFX track for the reel (all original, royalty-free)."""
import sys, wave
import numpy as np

SR = 48000
DUR = 21.5
N = int(SR * DUR)
rng = np.random.default_rng(7)
out_dir = sys.argv[1]


def write(path, x):
    x = np.clip(x, -1, 1)
    st = np.stack([x, x], 1) if x.ndim == 1 else x
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x); s = 0.0
    for i in range(len(x)):
        s = (1 - a) * x[i] + a * s; y[i] = s
    return y


def hp(x, cutoff):
    return x - lp(x, cutoff)


def add(buf, t, sig):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i]


def env(n, a, d):
    e = np.ones(n)
    na, nd = int(a * SR), int(d * SR)
    if na: e[:na] = np.linspace(0, 1, na)
    if nd: e[-nd:] *= np.linspace(1, 0, nd)
    return e


def note(f):
    return 440 * 2 ** ((f - 69) / 12)


# ---------------- music: 100 BPM, Am - F - C - G ----------------
bpm = 100; beat = 60 / bpm; bar = beat * 4
t = np.arange(N) / SR
chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
roots = [45, 41, 48, 43]
pad = np.zeros(N); bass = np.zeros(N); drums = np.zeros(N); keys = np.zeros(N)
nbars = int(np.ceil(DUR / bar)) + 1
for b in range(nbars):
    t0 = b * bar
    ch = chords[b % 4]
    n = int(bar * SR)
    tt = np.arange(n) / SR
    s = np.zeros(n)
    for m in ch:
        for det in (-0.08, 0.08):
            f = note(m + 12) * 2 ** (det / 12)
            s += 0.5 * np.sin(2 * np.pi * f * tt) + 0.18 * np.sin(4 * np.pi * f * tt)
    add(pad, t0, s * env(n, 0.25, 0.3) * 0.05)
    # plucky arpeggio on 8ths
    arp = ch + [ch[1] + 12]
    for k in range(8):
        m = arp[k % 4] + 12
        nn = int(beat / 2 * SR); tt2 = np.arange(nn) / SR
        p = np.sin(2 * np.pi * note(m) * tt2) * np.exp(-tt2 * 9)
        add(keys, t0 + k * beat / 2, p * 0.06)
    # bass on 1 and 3 (+ syncopation)
    for bt in (0, 2, 2.5):
        nn = int(beat * 0.9 * SR); tt2 = np.arange(nn) / SR
        f = note(roots[b % 4] - 12 + 12)
        sb = (np.sin(2 * np.pi * f * tt2) + 0.3 * np.sin(4 * np.pi * f * tt2)) * np.exp(-tt2 * 3) * env(nn, 0.005, 0.05)
        add(bass, t0 + bt * beat, sb * 0.16)
    for k in range(4):
        tb = t0 + k * beat
        # kick on 1 and 3
        if k in (0, 2):
            nn = int(0.35 * SR); tt2 = np.arange(nn) / SR
            ph = 2 * np.pi * np.cumsum(45 + 90 * np.exp(-tt2 * 30)) / SR
            add(drums, tb, np.sin(ph) * np.exp(-tt2 * 9) * 0.45)
        # clap on 2 and 4
        if k in (1, 3):
            nn = int(0.2 * SR); tt2 = np.arange(nn) / SR
            add(drums, tb, hp(rng.standard_normal(nn), 1200) * np.exp(-tt2 * 22) * 0.12)
        # hats on off-beats
        nn = int(0.06 * SR); tt2 = np.arange(nn) / SR
        add(drums, tb + beat / 2, hp(rng.standard_normal(nn), 7000) * np.exp(-tt2 * 70) * 0.07)

pad = lp(pad, 2500)
music = pad + bass + drums + keys
# intro filter-sweep feel: first 0.6s fade-in; outro fade
fade = np.ones(N)
fade[: int(0.4 * SR)] = np.linspace(0.3, 1, int(0.4 * SR))
fade[-int(0.8 * SR):] = np.linspace(1, 0, int(0.8 * SR))
music *= fade
music /= np.max(np.abs(music)) + 1e-9
write(f"{out_dir}/music.wav", music * 0.8)

# ---------------- SFX ----------------
sfx = np.zeros(N)


def whoosh(d=0.45, up=True, g=0.5):
    n = int(d * SR); tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    # sweeping band: crude via mixing lp at moving cutoff in chunks
    y = np.zeros(n); chunk = 480
    for i in range(0, n, chunk):
        frac = i / n
        c = 300 + 5000 * (frac if up else 1 - frac)
        seg = noise[i:i + chunk]
        y[i:i + chunk] = hp(lp(seg, c), c * 0.3)
    e = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 2
    return y * e * g * 2.2


def pop(f=900, g=0.35):
    n = int(0.12 * SR); tt = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f * (1 + 1.5 * np.exp(-tt * 40))) / SR
    return np.sin(ph) * np.exp(-tt * 35) * g


def click(g=0.3):
    n = int(0.03 * SR); tt = np.arange(n) / SR
    return hp(rng.standard_normal(n), 2000) * np.exp(-tt * 200) * g


def ding(f=1318.5, g=0.22):
    n = int(0.9 * SR); tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.5 * tt) + 0.25 * np.sin(2 * np.pi * f * 2 * tt)) * np.exp(-tt * 5) * g


def thud(g=0.6):
    n = int(0.5 * SR); tt = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(40 + 80 * np.exp(-tt * 18)) / SR
    return (np.sin(ph) * np.exp(-tt * 7) + lp(rng.standard_normal(n), 400) * np.exp(-tt * 25) * 0.5) * g


def buzz(g=0.18):
    n = int(0.35 * SR); tt = np.arange(n) / SR
    s = np.sign(np.sin(2 * np.pi * 140 * tt)) + np.sign(np.sin(2 * np.pi * 147 * tt))
    return lp(s, 1800) * env(n, 0.005, 0.08) * g


def riser(d=0.6, g=0.3):
    n = int(d * SR); tt = np.arange(n) / SR
    f = 200 + 1400 * (tt / d) ** 2
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * (tt / d) ** 2 * g + whoosh(d, True, g * 0.6)


def impact(g=0.7):
    t = np.zeros(int(1.2 * SR)); t[: int(0.5 * SR)] = thud(g)
    return t + lp(rng.standard_normal(int(1.2 * SR)), 1200) * np.exp(-np.arange(int(1.2 * SR)) / SR * 4) * 0.25 * g


add(sfx, 0.00, whoosh(0.35, True, 0.45))
add(sfx, 0.30, pop(700, 0.3))
add(sfx, 0.55, pop(1050, 0.25))
for tc in (1.96, 13.30):          # punch-in cuts
    add(sfx, tc - 0.01, click(0.18))
add(sfx, 4.52, whoosh(0.35, True, 0.5))   # b-roll A in
add(sfx, 6.70, whoosh(0.3, False, 0.35))  # out
add(sfx, 7.05, click(0.35))               # save tap
add(sfx, 7.12, ding(1568, 0.18))
add(sfx, 10.48, whoosh(0.35, True, 0.5))  # b-roll B in
add(sfx, 11.15, thud(0.55))               # bar drops
add(sfx, 12.80, whoosh(0.3, False, 0.35))
add(sfx, 13.45, ding(1318.5, 0.2))        # simple check
add(sfx, 15.62, whoosh(0.35, True, 0.5))  # b-roll C in
for k in range(12):                       # monthly coins
    add(sfx, 16.05 + k * 0.17, pop(1200 + 40 * k, 0.12))
add(sfx, 18.40, whoosh(0.3, False, 0.35))
add(sfx, 18.80, pop(800, 0.3))            # formula card
add(sfx, 20.62, buzz(0.2))                # strike
add(sfx, 21.05, impact(0.55))
write(f"{out_dir}/sfx.wav", sfx)
print("audio ok")
