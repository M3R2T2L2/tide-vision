#!/usr/bin/env python3
"""tide-vision: render a video through three computer vision views.

Views
  1. Optical flow   - Farneback dense flow, camera shake removed (median flow subtracted)
  2. Motion heatmap - exponentially decaying accumulation of flow magnitude
  3. Segmentation   - k-means (k=5) in Lab color space, centroids fit once and frozen

Outputs (any combination)
  vertical : 1080x1920, full frame, wipes through the four views (YouTube Shorts)
  panels   : 1920x1080, four views side by side with a motion graph underneath

Usage
  python tide_vision.py input.mp4 --out-dir out
  python tide_vision.py input.mp4 --layouts vertical --label "High tide, La Jolla CA"

Requires ffmpeg on PATH for encoding. Audio from the input is kept.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- config
AW, AH = 360, 640            # analysis resolution (portrait). Flow thresholds below assume this scale.
FLOW_ALPHA_LO, FLOW_ALPHA_SPAN = 0.07, 0.45   # px/frame: flow below LO is transparent
HEAT_DECAY = 0.96            # per-frame decay of the motion accumulator (~0.8 s memory at 30 fps)
HEAT_NORM = 22.0             # accumulator value mapped to full heat (set by eye)
FOAM_S_MAX, FOAM_V_MIN = 55, 205              # OpenCV 8-bit HSV: low saturation, high value
K = 5                        # k-means clusters
PALETTE_HEX = ["#2b1d6e", "#1f8a9e", "#7fbf3b", "#f2a33a", "#f6f1e6"]   # dark -> bright cluster
VIEW_NAMES = ["PERSON'S VIEW", "OPTICAL FLOW", "MOTION HEATMAP", "SEGMENTATION k=5"]
VIEW_COLORS = [(240, 240, 240), (90, 235, 255), (255, 170, 80), (170, 255, 140)]   # RGB, for text
WIPE_FRAMES = 30

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- analysis
def to_analysis(frame):
    return cv2.resize(frame, (AW, AH), interpolation=cv2.INTER_AREA)


def flow_step(prev_gray, small):
    """Dense flow between consecutive frames with global (camera) motion removed."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mx, my = np.median(fl[..., 0]), np.median(fl[..., 1])
    fl[..., 0] -= mx
    fl[..., 1] -= my
    mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1])
    return gray, mag, ang, float(np.hypot(mx, my))


def foam_mask(small, horizon_row):
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < FOAM_S_MAX) & (hsv[..., 2] > FOAM_V_MIN)).astype(np.uint8)
    m[:horizon_row] = 0
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fit_centroids(cap, n_frames):
    """k-means in Lab on pixels sampled from five evenly spaced frames. Returns centroids sorted dark->bright."""
    pts = []
    for fi in np.linspace(0, max(n_frames - 2, 0), 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            sm = cv2.resize(fr, (AW // 2, AH // 2), interpolation=cv2.INTER_AREA)
            pts.append(cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).reshape(-1, 3))
    pts = np.concatenate(pts).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, _, cent = cv2.kmeans(pts, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cent[np.argsort(cent[:, 0])]


def segment(small, centroids, palette_bgr):
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.2), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    d = ((lab[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    lbl = cv2.medianBlur(d.argmin(1).astype(np.uint8).reshape(AH, AW), 5)
    return palette_bgr[lbl]


def pass_one(path, horizon_frac):
    """Time series for the graph: mean sea motion and foam/glare share, below the horizon."""
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    prev = cv2.cvtColor(to_analysis(fr), cv2.COLOR_BGR2GRAY)
    row = int(horizon_frac * AH)
    motion, foam, shake = [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = to_analysis(fr)
        prev, mag, _, sh = flow_step(prev, small)
        shake.append(sh)
        motion.append(float(mag[row:].mean()))
        foam.append(100.0 * float(foam_mask(small, row)[row:].mean()))
    cap.release()
    motion = [motion[0]] + motion      # frame 0 has no predecessor: repeat the first value
    foam = [foam[0]] + foam
    return np.array(motion), np.array(foam), np.array(shake)


# --------------------------------------------------------------------------- views
def blend(base, col, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return (base.astype(np.float32) * (1 - a) + col.astype(np.float32) * a).astype(np.uint8)


def resize(a, w, h, interp=cv2.INTER_CUBIC):
    return cv2.resize(a, (w, h), interpolation=interp)


def dim_base(frame, w, h, edges):
    f = resize(frame, w, h, cv2.INTER_AREA)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor((g * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if edges:
        e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 2.4 * w / 360 / 2.2), 70, 160)
        base[e > 0] = (150, 110, 60)
    return base


def make_views(frame, ang, mag, energy, seg_bgr, w, h):
    hsv = np.zeros((AH, AW, 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8) % 180
    hsv[..., 1] = hsv[..., 2] = 255
    flow_col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_a = np.clip((cv2.GaussianBlur(mag, (0, 0), 4) - FLOW_ALPHA_LO) / FLOW_ALPHA_SPAN, 0, 0.88)

    heat_v = np.clip(energy / HEAT_NORM, 0, 1) ** 0.7
    heat_col = cv2.applyColorMap((heat_v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat_a = np.clip(heat_v * 1.6, 0, 0.92)

    person = resize(frame, w, h, cv2.INTER_AREA)
    flow = blend(dim_base(frame, w, h, True), resize(flow_col, w, h), resize(flow_a, w, h))
    heat = blend(dim_base(frame, w, h, False), resize(heat_col, w, h), resize(heat_a, w, h))
    seg = blend(person, resize(seg_bgr, w, h), np.full((h, w), 0.82, np.float32))
    return [person, flow, heat, seg]


# --------------------------------------------------------------------------- graph + layouts
class Series:
    def __init__(self, motion, foam):
        k = np.ones(5) / 5
        self.m = np.convolve(motion, k, "same")
        self.f = np.convolve(foam, k, "same")
        for a in (self.m, self.f):
            a[:3], a[-3:] = a[3], a[-4]
        self.m_lo, self.m_hi = np.percentile(self.m, [1, 99])
        self.f_lo, self.f_hi = np.percentile(self.f, [1, 99])
        self.n = len(self.m)

    def draw(self, d, x0, y0, x1, y1, idx):
        d.rectangle([x0, y0, x1, y1], outline=(70, 85, 100), width=2)
        w = x1 - x0

        def pts(a, lo, hi):
            xs = x0 + np.arange(len(a)) / (self.n - 1) * w
            ys = y1 - 5 - np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * (y1 - y0 - 10)
            return list(zip(xs.tolist(), ys.tolist()))

        d.line(pts(self.f, self.f_lo, self.f_hi), fill=(60, 200, 210), width=3)
        d.line(pts(self.m, self.m_lo, self.m_hi), fill=(255, 170, 60), width=3)
        cx = x0 + idx / (self.n - 1) * w
        d.line([cx, y0, cx, y1], fill=(255, 255, 255), width=3)


def vertical_frame(views, idx, fps, series, label, fonts):
    fw, fh = 1080, 1920
    n = len(views)
    seg_len = max(1, round(series.n / n))
    bounds = [i * seg_len for i in range(n)]
    mode = max(i for i, b in enumerate(bounds) if idx >= b)
    img = views[mode].copy()
    wx = None
    if mode > 0 and idx - bounds[mode] < WIPE_FRAMES:
        t = (idx - bounds[mode]) / WIPE_FRAMES
        wx = int(t * t * (3 - 2 * t) * fw)
        img = views[mode - 1].copy()
        img[:, :wx] = views[mode][:, :wx]
        if 0 < wx < fw:
            img[:, max(wx - 2, 0):wx + 2] = 255
    img[90:230] = (img[90:230] * 0.45).astype(np.uint8)
    img[1500:1640] = (img[1500:1640] * 0.45).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    show = mode if wx is None or wx > fw // 2 else mode - 1
    d.text((48, 110), VIEW_NAMES[show], font=fonts["chip"], fill=VIEW_COLORS[show])
    d.text((48, 172), f"t = {idx / fps:05.2f}s   {label}", font=fonts["hud"], fill=(215, 225, 235))
    series.draw(d, 48, 1530, fw - 48, 1620, idx)
    d.text((48, 1498), "sea motion", font=fonts["sm"], fill=(255, 170, 60))
    d.text((190, 1498), "foam/glare", font=fonts["sm"], fill=(60, 200, 210))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class PanelLayout:
    W, H, BAR, GRAPH = 1920, 1080, 56, 96

    def __init__(self, fonts):
        self.pw = self.W // 4
        self.ph = self.H - self.BAR - self.GRAPH
        self.crop_w = int(1920 * self.pw / self.ph)         # source crop width at full res (1080x1920 input)
        img = Image.new("RGB", (self.W, self.H), (10, 12, 16))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, self.W, self.BAR], fill=(14, 17, 23))
        for i, (name, col) in enumerate(zip(VIEW_NAMES, VIEW_COLORS)):
            d.text((i * self.pw + 18, 15), name, font=fonts["big"], fill=col)
            if i:
                d.line([i * self.pw, 0, i * self.pw, self.H - self.GRAPH], fill=(60, 70, 80), width=2)
        self.static = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.fonts = fonts

    def frame(self, views, idx, fps, n_frames, series, label):
        out = self.static.copy()
        for i, v in enumerate(views):
            x0 = max((v.shape[1] - self.crop_w) // 2, 0)
            crop = v[:, x0:x0 + self.crop_w]
            out[self.BAR:self.BAR + self.ph, i * self.pw:(i + 1) * self.pw] = cv2.resize(
                crop, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        gy = self.H - self.GRAPH + 10
        d.text((18, gy - 8), f"t = {idx / fps:05.2f}s  frame {idx + 1:04d}/{n_frames}",
               font=self.fonts["sm"], fill=(215, 225, 235))
        legend = f"orange: sea motion   cyan: foam/glare   {label}".strip()
        d.text((self.W - 18 - d.textlength(legend, font=self.fonts["sm"]), gy - 8), legend,
               font=self.fonts["sm"], fill=(140, 150, 160))
        series.draw(d, 18, gy + 16, self.W - 18, self.H - 10, idx)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- encoding
def open_encoder(out_path, w, h, fps, src, crf):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input video (portrait 1080x1920 expected for the panel crop)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--layouts", nargs="+", choices=["vertical", "panels"], default=["vertical", "panels"])
    ap.add_argument("--label", default="", help="text shown in the HUD, e.g. 'High tide, La Jolla CA'")
    ap.add_argument("--horizon", type=float, default=0.31,
                    help="horizon height as a fraction of frame height from the top; foam/glare and sea-motion "
                         "metrics ignore everything above it (default 0.31)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: stop after N frames")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(3)), int(cap.get(4))
    if (fw, fh) != (1080, 1920):
        print(f"note: input is {fw}x{fh}; layouts are designed for 1080x1920 portrait and frames will be resized.")

    print("pass 1/2: motion and foam/glare time series")
    motion, foam, shake = pass_one(args.input, args.horizon)
    print(f"  camera shake (median flow removed) median {np.median(shake):.3f} px/frame at {AW}x{AH}")
    print(f"  sea motion median {np.median(motion):.3f} px/frame, 98th pct {np.percentile(motion, 98):.3f}")
    print(f"  foam/glare median {np.median(foam):.1f}%, 98th pct {np.percentile(foam, 98):.1f}%")
    series = Series(motion, foam)

    centroids = fit_centroids(cap, n_frames)
    palette = np.array([[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)] for h in PALETTE_HEX], np.uint8)
    fonts = {"chip": load_font(FONT_BOLD, 44), "hud": load_font(FONT_REG, 30), "big": load_font(FONT_BOLD, 26),
             "sm": load_font(FONT_REG, 17)}
    panels = PanelLayout(fonts)

    encoders = {}
    if "vertical" in args.layouts:
        encoders["vertical"] = open_encoder(os.path.join(args.out_dir, f"{stem}_vertical.mp4"), 1080, 1920, fps, args.input, args.crf)
    if "panels" in args.layouts:
        encoders["panels"] = open_encoder(os.path.join(args.out_dir, f"{stem}_panels.mp4"), 1920, 1080, fps, args.input, args.crf)

    print("pass 2/2: rendering")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    prev = cv2.cvtColor(to_analysis(frame), cv2.COLOR_BGR2GRAY)
    energy = np.zeros((AH, AW), np.float32)
    horizon_row = int(args.horizon * AH)
    idx = 0
    while ok:
        small = to_analysis(frame)
        if idx == 0:
            mag = np.zeros((AH, AW), np.float32)
            ang = np.zeros((AH, AW), np.float32)
        else:
            prev, mag, ang, _ = flow_step(prev, small)
            mag[:horizon_row] = 0          # sky has no texture to track; keep it out of flow and heat
        energy = energy * HEAT_DECAY + cv2.GaussianBlur(mag, (0, 0), 3)
        views = make_views(frame if (fw, fh) == (1080, 1920) else cv2.resize(frame, (1080, 1920)),
                           ang, mag, energy, segment(small, centroids, palette), 1080, 1920)
        if "vertical" in encoders:
            encoders["vertical"].stdin.write(vertical_frame(views, idx, fps, series, args.label, fonts).tobytes())
        if "panels" in encoders:
            encoders["panels"].stdin.write(panels.frame(views, idx, fps, n_frames, series, args.label).tobytes())
        idx += 1
        if args.max_frames and idx >= args.max_frames:
            break
        ok, frame = cap.read()
    for p in encoders.values():
        p.stdin.close()
        p.wait()
    print(f"done: {idx} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
