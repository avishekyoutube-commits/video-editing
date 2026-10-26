"""Compose the reel: graded footage + punch-in jump cuts + motion-graphic B-roll + UI cards + animated captions.
Reads graded RGB frames on stdin-pipe from ffmpeg, writes RGB frames to an ffmpeg encoder."""
import math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SRC, AUDIO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREVIEW = sys.argv[4] if len(sys.argv) > 4 else None   # optional "t1,t2,..." -> write stills only
W, H, FPS = 1080, 1920, 30
NF = int(21.5 * FPS)
FONT = "/usr/share/fonts/opentype/inter/"
_fc = {}


def font(sz, w="Black"):
    k = (sz, w)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(f"{FONT}Inter-{w}.otf", sz)
    return _fc[k]


# palette
WHITE = (255, 255, 255); YEL = (255, 210, 63); ACC = (124, 77, 255); GRN = (46, 229, 157)
RED = (255, 77, 90); NAVY = (12, 16, 38); CARD = (18, 20, 34)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x); return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x); return 3 * x * x - 2 * x * x * x


def back_out(x, s=1.9):
    x = clamp(x); x -= 1; return x * x * ((s + 1) * x + s) + 1


def window(t, a, b, fi=0.2, fo=0.2):
    """0..1 visibility for interval [a,b] with fades."""
    if t < a or t > b: return 0.0
    return min(clamp((t - a) / fi) if fi else 1, clamp((b - t) / fo) if fo else 1)


# ---------------------------------------------------------------- captions
# (start_idx, end_idx_inclusive, text words) using whisper word timings, text corrected by hand (Hinglish)
import json
WT = json.load(open(sys.argv[0].rsplit("/", 1)[0] + "/words.json"))
def ts(i): return WT[i][0]
def te(i): return WT[i][1]

# each word: (text, whisper index range start, end)
WORDS = [
    ("SIP", 0, 2), ("ka", 3, 3), ("RETURN", 4, 4), ("calculate", 5, 5), ("karna", 6, 6), ("ek", 7, 7),
    ("LUMPSUM", 8, 8), ("ke", 9, 9), ("return", 10, 10), ("calculate", 11, 11), ("karne", 12, 12),
    ("jaisa", 13, 13), ("NAHI", 14, 14), ("hota", 15, 15),
    ("Zyaadatar", 16, 17), ("log", 18, 18), ("yahan", 19, 19), ("CONFUSE", 20, 20), ("ho", 21, 21),
    ("jaate", 22, 22), ("hain", 23, 23), ("ise", 24, 25), ("SAVE", 26, 26), ("kar", 27, 27), ("lijiye", 28, 28),
    ("apni", 29, 29), ("SIP", 30, 32), ("ka", 33, 33), ("return", 34, 34), ("calculate", 35, 35),
    ("karne", 36, 36), ("se", 37, 37), ("PEHLE", 38, 38),
    ("LUMPSUM", 39, 39), ("mein", 39, 39), ("EK", 40, 40), ("HI", 41, 41), ("DATE", 42, 42), ("pe", 43, 43),
    ("paisa", 44, 44), ("invest", 45, 45), ("hota", 46, 46), ("hai,", 47, 47), ("isliye", 48, 48),
    ("return", 49, 49), ("calculate", 50, 50), ("karna", 51, 51), ("bahut", 52, 52), ("SIMPLE", 53, 53), ("hai", 54, 54),
    ("SIP", 55, 57), ("mein", 58, 58), ("HAR", 59, 59), ("MAHINE", 60, 60), ("ALAG", 61, 61), ("DATE", 62, 62),
    ("par", 63, 63), ("amount", 64, 64), ("invest", 65, 65), ("hota", 66, 66), ("hai,", 67, 67),
    ("isliye", 68, 68), ("NORMAL", 69, 69), ("RETURN", 70, 70), ("FORMULA", 71, 71), ("yahan", 72, 72),
    ("kaam", 73, 73), ("NAHI", 74, 74), ("karega", 75, 75),
]
# "Lumpsum mein" share whisper word 39 -> split its time
words = []
for k, (txt, a, b) in enumerate(WORDS):
    words.append([txt, ts(a), te(b)])
for k in range(len(words) - 1):
    if words[k][1] == words[k + 1][1]:   # same source word -> split
        s, e = words[k][1], words[k][2]; m = s + (e - s) * 0.62
        words[k][2] = m; words[k + 1][1] = m
CHUNKS = [(0, 2), (3, 4), (5, 7), (8, 10), (11, 13),
          (14, 15), (16, 20), (21, 24), (25, 28), (29, 32),
          (33, 34), (35, 38), (39, 42), (43, 45), (46, 49),
          (50, 53), (54, 56), (57, 60), (61, 62), (63, 64), (65, 68)]
assert CHUNKS[-1][1] == len(words) - 1
KEY = {"SIP", "RETURN", "LUMPSUM", "CONFUSE", "SAVE", "SIMPLE", "HAR", "MAHINE", "ALAG", "DATE", "NORMAL",
       "FORMULA", "PEHLE", "EK", "HI"}
NEG = {"NAHI"}
CAP_Y = 1165
CAP_SZ = 84


def draw_captions(img, t):
    ci = None
    for n, (a, b) in enumerate(CHUNKS):
        s = words[a][1] - 0.05
        e = words[CHUNKS[n + 1][0]][1] - 0.05 if n + 1 < len(CHUNKS) else 21.5
        if s <= t < e: ci = n; break
    if ci is None: return
    a, b = CHUNKS[ci]
    start = words[a][1] - 0.05
    pin = back_out((t - start) / 0.18)
    toks = [w[0].upper().rstrip(",") for w in words[a:b + 1]]
    f = font(CAP_SZ)
    space = f.getlength(" ")
    widths = [f.getlength(x) for x in toks]
    # wrap into lines (max 900px)
    lines, cur, cw = [], [], 0
    for i, wdt in enumerate(widths):
        if cur and cw + space + wdt > 900:
            lines.append(cur); cur, cw = [], 0
        cw += (space if cur else 0) + wdt; cur.append(i)
    lines.append(cur)
    lh = int(CAP_SZ * 1.18)
    layer = Image.new("RGBA", (W, lh * len(lines) + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for li, ln in enumerate(lines):
        lw = sum(widths[i] for i in ln) + space * (len(ln) - 1)
        x = (W - lw) / 2; y = 30 + li * lh
        for i in ln:
            wd = words[a + i]
            active = wd[1] - 0.04 <= t < wd[2] + 0.06
            txt = toks[i]
            col = RED if txt in NEG else (YEL if txt in KEY else WHITE)
            if active:
                pad = 14
                d.rounded_rectangle([x - pad, y - 6, x + widths[i] + pad, y + CAP_SZ + 14], 18,
                                    fill=(RED if txt in NEG else ACC) + (235,))
                col = WHITE
            d.text((x, y), txt, font=f, fill=col + (255,), stroke_width=7 if not active else 0, stroke_fill=(0, 0, 0, 255))
            x += widths[i] + space
    # drop shadow
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sh.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.55)))
    sc = 0.82 + 0.18 * pin
    full = Image.alpha_composite(sh, layer)
    if sc != 1:
        full = full.resize((max(1, int(full.width * sc)), max(1, int(full.height * sc))), Image.BICUBIC)
    alpha = clamp((t - start) / 0.08)
    if alpha < 1:
        full.putalpha(full.getchannel("A").point(lambda v: int(v * alpha)))
    img.alpha_composite(full, (int((W - full.width) / 2), int(CAP_Y - full.height / 2)))


# ---------------------------------------------------------------- jump cuts (punch-in zooms)
CUTS = [(0.0, 1.10), (1.96, 1.26), (4.70, 1.04), (6.86, 1.18), (10.70, 1.06), (13.30, 1.24), (15.42, 1.04), (18.72, 1.20)]
FACE = (540, 600)


def zoom_for(t):
    seg = 0
    for i, (c, z) in enumerate(CUTS):
        if t >= c: seg = i
    c0, z0 = CUTS[seg]
    c1 = CUTS[seg + 1][0] if seg + 1 < len(CUTS) else 21.5
    p = (t - c0) / (c1 - c0)
    # slow push-in within each segment + tiny settle after the cut
    return z0 + 0.035 * ease_in_out(p) + 0.025 * (1 - ease_out((t - c0) / 0.25))


def zoomed(frame, z):
    cx, cy = FACE
    cw, ch = W / z, H / z
    x0 = clamp(cx - cw / 2, 0, W - cw); y0 = clamp(cy - ch * 0.42, 0, H - ch)
    return frame.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


# ---------------------------------------------------------------- B-roll scenes (motion graphics)
def bg_grad(t, c1=(10, 14, 40), c2=(40, 18, 70)):
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    g = clamp(0.5 + 0.5 * math.sin(t * 0.8))
    m = (y * 0.75 + x * 0.25 * g)[..., None]
    arr = np.array(c1)[None, None, :] * (1 - m) + np.array(c2)[None, None, :] * m
    # soft glow blob
    gx, gy = 0.5 + 0.2 * math.sin(t * 0.9), 0.35 + 0.05 * math.cos(t * 1.1)
    r = np.sqrt((x - gx) ** 2 + ((y - gy) * H / W) ** 2)
    arr += (np.exp(-r * 4.5)[..., None] * np.array([90, 60, 200])[None, None, :] * 0.55)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")


def grid_overlay(d, t, alpha=28):
    off = (t * 40) % 90
    for x in range(0, W + 90, 90):
        d.line([(x - off, 0), (x - off, H)], fill=(255, 255, 255, alpha), width=1)
    for y in range(0, H + 90, 90):
        d.line([(0, y - off), (W, y - off)], fill=(255, 255, 255, alpha), width=1)


def label_pill(d, cx, y, text, sz, fg, bg, a=1.0):
    f = font(sz)
    tw = f.getlength(text)
    d.rounded_rectangle([cx - tw / 2 - 30, y, cx + tw / 2 + 30, y + sz + 34], (sz + 34) // 2, fill=bg + (int(255 * a),))
    d.text((cx - tw / 2, y + 12), text, font=f, fill=fg + (int(255 * a),))


def scene_confused(t, lt):
    img = bg_grad(t, (14, 10, 30), (70, 16, 40))
    d = ImageDraw.Draw(img, "RGBA")
    grid_overlay(d, t)
    # chaotic chart line
    pts = []
    for i in range(60):
        x = 120 + i * 14
        y = 700 - i * 4 + 120 * math.sin(i * 0.7 + lt * 6) * math.sin(i * 0.23 + lt * 2)
        pts.append((x, y))
    n = int(len(pts) * ease_out(lt / 0.8))
    if n > 1:
        d.line(pts[:n], fill=(255, 120, 140, 255), width=10, joint="curve")
    # calculator card
    cy = 860 + 40 * (1 - ease_out(lt / 0.3))
    d.rounded_rectangle([190, cy, 890, cy + 190], 36, fill=(0, 0, 0, 170), outline=(255, 255, 255, 60), width=3)
    d.text((240, cy + 30), "RETURN =", font=font(60), fill=WHITE + (255,))
    rnd = np.random.default_rng(int(lt * 12))
    val = f"{rnd.integers(1, 99)}.{rnd.integers(0, 99):02d}%"
    d.text((590, cy + 30), val, font=font(60), fill=YEL + (255,))
    d.text((240, cy + 110), "???  CAGR? XIRR? ABS?", font=font(40, "Bold"), fill=(255, 255, 255, 170))
    # floating question marks
    for k in range(7):
        ph = k * 1.7
        x = 120 + (k * 137) % 840
        y = 330 + 60 * math.sin(lt * 3 + ph) + (k % 3) * 70
        s = 0.6 + 0.4 * back_out((lt - k * 0.08) / 0.25)
        f = font(int(110 * s))
        d.text((x, y), "?", font=f, fill=(255, 255, 255, int(200 * clamp((lt - k * 0.08) / 0.2))), stroke_width=4, stroke_fill=(0, 0, 0, 200))
    label_pill(d, W / 2, 180, "LOG YAHAN CONFUSE HOTE HAIN", 46, (20, 20, 20), YEL, clamp(lt / 0.25))
    return img


def scene_lumpsum(t, lt):
    img = bg_grad(t, (8, 22, 30), (10, 60, 70))
    d = ImageDraw.Draw(img, "RGBA")
    grid_overlay(d, t)
    label_pill(d, W / 2, 180, "LUMPSUM  =  1 DATE, 1 BAAR", 50, (10, 10, 10), GRN, clamp(lt / 0.25))
    base = 950; x0, x1 = 110, 970
    d.line([(x0, base), (x1, base)], fill=(255, 255, 255, 200), width=5)
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for i, m in enumerate(months):
        x = x0 + 30 + i * 70
        d.text((x - 16, base + 16), m, font=font(22, "Bold"), fill=(255, 255, 255, 150))
    # single big bar dropping in on JAN with growth curve after
    p = back_out((lt - 0.25) / 0.45, 1.4)
    bh = 420 * p
    bx = x0 + 30 - 26
    d.rounded_rectangle([bx, base - bh, bx + 56, base], 12, fill=GRN + (255,))
    if lt > 0.5:
        d.text((bx - 10, base - bh - 70), "₹1,00,000", font=font(48), fill=WHITE + (int(255 * clamp((lt - 0.5) / 0.2)),))
    # growth line from bar top to end
    q = ease_out((lt - 0.8) / 1.0)
    if q > 0:
        pts = []
        for i in range(int(60 * q) + 1):
            u = i / 60
            pts.append((bx + 28 + u * (x1 - bx - 60), base - 420 - 230 * (u ** 1.6)))
        if len(pts) > 1: d.line(pts, fill=YEL + (255,), width=8, joint="curve")
        if q > 0.95:
            d.text((x1 - 330, base - 720), "CAGR ✓", font=font(64), fill=YEL + (255,))
    # calendar chip
    cy = 1350 - 1000  # off: keep lower area for captions
    return img


def scene_sip(t, lt):
    img = bg_grad(t, (10, 12, 36), (36, 26, 90))
    d = ImageDraw.Draw(img, "RGBA")
    grid_overlay(d, t)
    label_pill(d, W / 2, 180, "SIP  =  HAR MAHINE, ALAG DATE", 50, (255, 255, 255), ACC, clamp(lt / 0.25))
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    dates = [5, 3, 7, 5, 2, 6, 4, 5, 3, 7, 5, 4]
    # 4x3 calendar tiles
    for i, m in enumerate(months):
        r, c = divmod(i, 4)
        x = 95 + c * 228; y = 330 + r * 230
        appear = ease_out((lt - 0.05 * i) / 0.25)
        if appear <= 0: continue
        yy = y + 30 * (1 - appear)
        d.rounded_rectangle([x, yy, x + 200, yy + 200], 26, fill=(255, 255, 255, int(28 * appear) + 10), outline=(255, 255, 255, int(90 * appear)), width=2)
        d.rounded_rectangle([x, yy, x + 200, yy + 52], 26, fill=ACC + (int(220 * appear),))
        d.rectangle([x, yy + 30, x + 200, yy + 52], fill=ACC + (int(220 * appear),))
        d.text((x + 100 - font(28).getlength(m) / 2, yy + 10), m, font=font(28), fill=WHITE + (int(255 * appear),))
        # coin pops in on its date
        cp = back_out((lt - 0.45 - 0.17 * i) / 0.22)
        if cp > 0:
            rr = 44 * cp
            cx, cyy = x + 100, yy + 128
            d.ellipse([cx - rr, cyy - rr, cx + rr, cyy + rr], fill=YEL + (255,), outline=(200, 140, 20, 255), width=5)
            txt = f"{dates[i]}"
            ft = font(int(40 * max(cp, 0.1)))
            d.text((cx - ft.getlength(txt) / 2, cyy - 26 * cp), txt, font=ft, fill=(60, 40, 0, 255))
    return img


BROLL = [  # (start, end, scene)
    (4.70, 6.86, scene_confused),
    (10.66, 12.98, scene_lumpsum),
    (15.80, 18.55, scene_sip),
]
TR = 0.22  # transition length


def apply_broll(base, t):
    for a, b, fn in BROLL:
        if a - 0.01 <= t <= b:
            lt = t - a
            sc = fn(t, lt).convert("RGB")
            # Ken-burns slow zoom on the scene
            z = 1.0 + 0.04 * (lt / (b - a))
            if z != 1.0: sc = zoomed_center(sc, z)
            pin = ease_out(lt / TR); pout = ease_out((b - t) / TR)
            p = min(pin, pout)
            if p >= 1: return sc, True
            # zoom-blur style transition: incoming scene scales from 1.12 -> 1, crossfade, flash
            s = 1 + 0.12 * (1 - p)
            sc2 = zoomed_center(sc, s).filter(ImageFilter.GaussianBlur(10 * (1 - p)))
            out = Image.blend(base, sc2, p)
            flash = 1 - abs(p - 0.5) * 2
            if flash > 0:
                out = Image.blend(out, Image.new("RGB", (W, H), (255, 255, 255)), 0.18 * flash)
            return out, True
    return base, False


def zoomed_center(im, z):
    cw, ch = W / z, H / z
    x0, y0 = (W - cw) / 2, (H - ch) / 2
    return im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


# ---------------------------------------------------------------- UI overlays
def progress_bar(d, t):
    p = t / 21.5
    d.rounded_rectangle([40, 70, W - 40, 80], 5, fill=(255, 255, 255, 70))
    d.rounded_rectangle([40, 70, 40 + (W - 80) * p, 80], 5, fill=YEL + (255,))


def hook(img, t):
    a = window(t, 0.0, 4.62, 0.01, 0.25)
    if a <= 0: return
    layer = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    p1 = back_out((t - 0.05) / 0.35)
    f = font(118)
    parts = [("SIP", YEL), (" ≠ ", RED), ("LUMPSUM", WHITE)]
    tw = sum(f.getlength(x) for x, _ in parts)
    # backing card
    d.rounded_rectangle([W / 2 - tw / 2 - 40, 40, W / 2 + tw / 2 + 40, 210], 40, fill=(10, 10, 20, 200))
    x = W / 2 - tw / 2
    for i, (s, c) in enumerate(parts):
        pi = back_out((t - 0.05 - 0.22 * i) / 0.3)
        if pi > 0:
            fi = font(max(10, int(118 * (0.6 + 0.4 * pi))))
            dy = (118 - fi.size) / 2
            d.text((x + (f.getlength(s) - fi.getlength(s)) / 2, 58 + dy), s, font=fi, fill=c + (255,), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        x += f.getlength(s)
    # subtitle pill slides in
    p2 = ease_out((t - 0.7) / 0.35)
    if p2 > 0:
        txt = "RETURN CALCULATION"
        fs = font(44); w2 = fs.getlength(txt) + 70
        x2 = W / 2 - w2 / 2 + 300 * (1 - p2)
        d.rounded_rectangle([x2, 240, x2 + w2, 312], 36, fill=ACC + (int(255 * p2),))
        d.text((x2 + 35, 252), txt, font=fs, fill=WHITE + (int(255 * p2),))
    if a < 1: layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
    img.alpha_composite(layer, (0, int(60 + 30 * (1 - p1))))


def card(img, t, a, b, y, h, drawfn):
    vis = window(t, a, b, 0.0001, 0.2)
    if vis <= 0: return
    pin = back_out((t - a) / 0.35, 1.5)
    layer = Image.new("RGBA", (W, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.rounded_rectangle([90, 20, 930, 20 + h], 34, fill=CARD + (225,), outline=(255, 255, 255, 50), width=2)
    drawfn(d, t - a, layer)
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sh.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.5)))
    layer = Image.alpha_composite(sh, layer)
    s = 0.7 + 0.3 * pin
    lay = layer.resize((max(1, int(W * s)), max(1, int(layer.height * s))), Image.BICUBIC)
    av = vis * clamp((t - a) / 0.12)
    if av < 1: lay.putalpha(lay.getchannel("A").point(lambda v: int(v * av)))
    img.alpha_composite(lay, (int((W - lay.width) / 2), int(y + (layer.height - lay.height) / 2 + 60 * (1 - ease_out((t - a) / 0.3)))))


def draw_check(d, cx, cy, r, col, p=1.0):
    pts = [(cx - r * 0.5, cy), (cx - r * 0.12, cy + r * 0.38), (cx + r * 0.55, cy - r * 0.4)]
    if p < 0.5:
        q = p / 0.5; pts = [pts[0], (pts[0][0] + (pts[1][0] - pts[0][0]) * q, pts[0][1] + (pts[1][1] - pts[0][1]) * q)]
    else:
        q = (p - 0.5) / 0.5; pts = [pts[0], pts[1], (pts[1][0] + (pts[2][0] - pts[1][0]) * q, pts[1][1] + (pts[2][1] - pts[1][1]) * q)]
    d.line(pts, fill=col + (255,), width=int(r * 0.22), joint="curve")


def save_ui(d, lt, layer):
    # bookmark button + text, tap at lt~0.2, toast after
    tap = 0.2
    filled = lt > tap
    bx, by = 150, 70
    # ripple
    if tap < lt < tap + 0.5:
        r = 40 + 120 * ease_out((lt - tap) / 0.5)
        d.ellipse([bx + 40 - r, by + 60 - r, bx + 40 + r, by + 60 + r], outline=(255, 255, 255, int(200 * (1 - (lt - tap) / 0.5))), width=6)
    s = 1 + 0.25 * math.sin(math.pi * clamp((lt - tap) / 0.25)) if filled else 1
    w_, h_ = 70 * s, 92 * s
    cx, cy = bx + 40, by + 60
    poly = [(cx - w_ / 2, cy - h_ / 2), (cx + w_ / 2, cy - h_ / 2), (cx + w_ / 2, cy + h_ / 2), (cx, cy + h_ / 2 - 26 * s), (cx - w_ / 2, cy + h_ / 2)]
    d.polygon(poly, fill=(YEL + (255,)) if filled else None, outline=WHITE + (255,), width=7)
    d.text((300, 62), "ISE SAVE KAR LO", font=font(62), fill=WHITE + (255,))
    sub = "Saved to collection ✓" if filled else "Tap to save"
    d.text((302, 140), sub, font=font(36, "SemiBold"), fill=(GRN if filled else (200, 200, 210)) + (255,))
    # finger cursor
    if lt < tap + 0.35:
        fx = cx + 30 + 140 * (1 - ease_out(lt / tap)) if lt < tap else cx + 30
        fy = cy + 40 + 120 * (1 - ease_out(lt / tap)) if lt < tap else cy + 40
        a = int(255 * clamp(1 - (lt - tap) / 0.35)) if lt > tap else 255
        d.ellipse([fx - 26, fy - 26, fx + 26, fy + 26], fill=(255, 255, 255, int(a * 0.85)), outline=(0, 0, 0, int(a * 0.6)), width=3)


def simple_ui(d, lt, layer):
    d.text((150, 58), "LUMPSUM RETURN", font=font(40, "Bold"), fill=(190, 190, 205, 255))
    d.text((150, 112), "CAGR = (FV / PV)", font=font(56), fill=WHITE + (255,))
    d.text((150 + font(56).getlength("CAGR = (FV / PV)"), 104), "1/n", font=font(32), fill=WHITE + (255,))
    d.text((150 + font(56).getlength("CAGR = (FV / PV)") + font(32).getlength("1/n") + 8, 112), "− 1", font=font(56), fill=WHITE + (255,))
    p = ease_out((lt - 0.15) / 0.4)
    if p > 0:
        d.ellipse([790, 70, 890, 170], fill=GRN + (int(255 * p),))
        draw_check(d, 840, 120, 46, (10, 40, 20), p)
    pl = ease_out((lt - 0.4) / 0.3)
    if pl > 0:
        d.rounded_rectangle([150, 196, 150 + 330 * pl, 250], 27, fill=GRN + (60,))
        if pl > 0.9: d.text((172, 202), "BAHUT SIMPLE", font=font(34), fill=GRN + (255,))


def formula_ui(d, lt, layer):
    d.text((150, 58), "SIP RETURN", font=font(40, "Bold"), fill=(190, 190, 205, 255))
    f = font(56)
    x = 150
    d.text((x, 112), "CAGR = (FV / PV)", font=f, fill=WHITE + (255,))
    x2 = x + f.getlength("CAGR = (FV / PV)")
    d.text((x2, 104), "1/n", font=font(32), fill=WHITE + (255,))
    x3 = x2 + font(32).getlength("1/n") + 8
    d.text((x3, 112), "− 1", font=f, fill=WHITE + (255,))
    xe = x3 + f.getlength("− 1")
    strike = ease_out((lt - 1.85) / 0.25)   # ~20.6s ("kaam nahi")
    if strike > 0:
        d.line([(x - 10, 148), (x - 10 + (xe - x + 20) * strike, 148)], fill=RED + (255,), width=10)
    if lt > 2.0:
        p = back_out((lt - 2.0) / 0.3)
        r = 50 * p
        d.ellipse([840 - r, 120 - r, 840 + r, 120 + r], fill=RED + (255,))
        if p > 0.5:
            d.line([(818, 98), (862, 142)], fill=WHITE + (255,), width=12)
            d.line([(862, 98), (818, 142)], fill=WHITE + (255,), width=12)
        d.rounded_rectangle([150, 196, 150 + 470 * min(p, 1), 250], 27, fill=RED + (70,))
        if p > 0.9: d.text((172, 202), "SIP PE KAAM NAHI", font=font(34), fill=(255, 140, 150, 255))


def draw_frame(base_rgb, t):
    fr = zoomed(base_rgb, zoom_for(t))
    fr, is_broll = apply_broll(fr, t)
    img = fr.convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov, "RGBA")
    progress_bar(d, t)
    img.alpha_composite(ov)
    hook(img, t)
    card(img, t, 6.95, 10.62, 1330, 220, save_ui)
    card(img, t, 13.05, 15.78, 1330, 270, simple_ui)
    card(img, t, 18.75, 21.5, 1330, 270, formula_ui)
    draw_captions(img, t)
    return img.convert("RGB")


GRADE = ("eq=contrast=1.07:saturation=1.10:gamma=0.98,"
         "colorbalance=rs=-0.03:gs=0.0:bs=0.05:rh=0.04:gh=0.01:bh=-0.03,"
         "curves=master='0/0.02 0.25/0.22 0.5/0.5 0.75/0.79 1/0.98',"
         "unsharp=5:5:0.4,vignette=angle=PI/5")

if __name__ == "__main__":
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-vf", GRADE + ",scale=1080:1920,fps=30", "-frames:v", str(NF),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    want = None
    if PREVIEW:
        want = {int(round(float(x) * FPS)): x for x in PREVIEW.split(",")}
        enc = None
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                                "-i", "-", "-i", AUDIO, "-map", "0:v", "-map", "1:a",
                                "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-level", "4.1",
                                "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                                "-g", "60", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", "21.5", OUT],
                               stdin=subprocess.PIPE)
    fsz = W * H * 3
    i = 0
    while i < NF:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz: break
        t = i / FPS
        if want is not None:
            if i in want:
                draw_frame(Image.frombuffer("RGB", (W, H), buf), t).save(f"{OUT}_{want[i]}.jpg", quality=88)
        else:
            enc.stdin.write(draw_frame(Image.frombuffer("RGB", (W, H), buf), t).tobytes())
        if i % 60 == 0: print("frame", i, file=sys.stderr, flush=True)
        i += 1
    if enc:
        # pad if source short
        enc.stdin.close(); enc.wait()
    dec.wait()
    print("frames", i)
