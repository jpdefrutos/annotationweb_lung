"""
Training-free lumen box proposals (detection + IoU tracking) for the
bronchoscopy bounding box task, adapted from the lumen scripts (lumen_core.py,
detect_roi.py, preview_boxes.py). With min_children=1 and min_track_len=1 it
gives the same boxes and box_ids as lumen_annotate.py.

Per frame: K-means on intensity inside the scope image, the darkest cluster
split into blobs gives the main (level-1) boxes; optionally K-means again
inside each blob gives nested (level-2) boxes. An IoU tracker with Hungarian
matching links boxes over frames. Additions for the web tool:

  - nested boxes are kept only when a parent has >= min_children of them
  - main boxes on tracks shorter than min_track_len frames are dropped as
    noise (see Runner.process for chunk edges)
  - the tracker state is JSON, so the browser can send it back with the next
    chunk and the tracks continue across requests

box_id is only unique within one run; identify a track by (run_id, box_id).
"""

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from common.metaimage import MetaImage

L1_COLOR, L2_COLOR = "#ff9900", "#30e0e0"

BLACK_THRESHOLD = 15     # pixel brightness below this = black (ROI detection)
BLACK_FRAC = 0.05        # row/col is "black" if < this fraction is bright
N_SAMPLES = 15           # frames sampled over the video for ROI + scope mask


class LumenError(Exception):
    pass


# ── parameters ────────────────────────────────────────────────────────────────

@dataclass
class Params:
    # level 1
    k: int = 5                  # K-means clusters on the whole scope image
    min_area: float = 0.02      # min blob area, fraction of the scope image
    max_area: float = 1.0       # max blob area, fraction of the scope image
    blur: float = 0.0           # Gaussian sigma before clustering, px
    open: int = 0               # morphological opening on the dark mask
    erode: int = 5              # shrink the scope mask by this many px
    # level 2
    levels: int = 1             # 1 = level 1 only, 2 = level 1 + children
    k2: int = 3                 # K-means clusters inside each level-1 blob
    min_gap: float = 15.0       # darkest sub-centre this much below the next
    child_min: float = 0.03     # min child area, fraction of its parent blob
    child_max: float = 0.7      # max child area, fraction of its parent blob
    min_children: int = 2       # keep a parent's children only if it has this many
    open2: int = 0              # opening on the child mask
    # tracking
    track: bool = True
    track_iou: float = 0.3      # min IoU to continue a level-1 track
    track_iou2: float = None    # same for level 2 (None = track_iou)
    max_lost: int = 5           # frames a track survives without a match
    min_track_len: int = 2      # with tracking: drop main boxes whose track is in fewer frames

    @property
    def iou2(self):
        return self.track_iou if self.track_iou2 is None else self.track_iou2


# Fields the UI may set, with (type, min, max). Everything else stays default.
PARAM_LIMITS = {
    "levels": (int, 1, 2),
    "min_area": (float, 0.0, 1.0),
    "min_gap": (float, 0.0, 255.0),
    "child_min": (float, 0.0, 1.0),
    "child_max": (float, 0.0, 1.0),
    "min_children": (int, 1, 20),
    "track": (bool, None, None),
    "track_iou": (float, 0.0, 1.0),
    "track_iou2": (float, 0.0, 1.0),
    # max_lost and min_track_len together set the look-ahead past each chunk,
    # (min_track_len - 1) * (max_lost + 1) frames: at most 44 with these limits.
    "max_lost": (int, 0, 10),
    "min_track_len": (int, 1, 5),
}


def params_from_dict(d):
    """Params from untrusted input: only known fields, type-checked, in range."""
    kw = {}
    for name, (typ, lo, hi) in PARAM_LIMITS.items():
        if name not in d or d[name] is None or d[name] == "":
            continue
        v = d[name]
        try:
            if typ is bool:
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on")
            else:
                v = typ(v)
        except (TypeError, ValueError):
            raise LumenError(f"Invalid value for {name}: {d[name]!r}")
        if lo is not None and not (lo <= v <= hi):
            raise LumenError(f"{name} must be between {lo} and {hi}")
        kw[name] = v
    return Params(**kw)


# ── frames ────────────────────────────────────────────────────────────────────

def read_mhd(path):
    """(H, W, C) uint8 array of an 8-bit .mhd frame, read with common.metaimage."""
    name = os.path.basename(path)
    try:
        image = MetaImage(filename=path)
    except Exception as e:      # MetaImage raises plain Exception and parse errors
        raise LumenError(f"Could not read {name}: {e}")
    etype = image.get_attribute("ElementType")
    if etype != "MET_UCHAR":
        raise LumenError(f"{name}: unsupported pixel type {etype} (only 8-bit frames)")
    img = image.get_pixel_data()
    return img if img.ndim == 3 else img[..., None]


def sequence_frames(image_sequence):
    """{frame_nr: .mhd path} for every frame of the ImageSequence."""
    fmt = image_sequence.format
    if not fmt.lower().endswith(".mhd"):
        raise LumenError("Box suggestions only support .mhd image sequences")
    start = image_sequence.start_frame_nr
    return {f: fmt.replace("#", str(f))
            for f in range(start, start + image_sequence.nr_of_frames)}


def crop(img, roi):
    return img[roi["top"]:roi["bottom"] + 1, roi["left"]:roi["right"] + 1]


# ── ROI and scope mask ────────────────────────────────────────────────────────

def detect_roi(img):
    """Bounding rectangle of the bright scope image in one frame (detect_roi.py)."""
    H, W = img.shape[:2]
    gray = img.mean(axis=2)
    bright = gray > BLACK_THRESHOLD

    def row_is_black(r):
        return bright[r, :].mean() < BLACK_FRAC

    def col_is_black(c):
        return bright[:, c].mean() < BLACK_FRAC

    mid = bright[H // 4: 3 * H // 4, :]
    col_idx = np.where(mid.any(axis=0))[0]
    if len(col_idx) == 0:
        return None
    right_edge = int(col_idx[-1])

    seed_col = max(0, right_edge - H // 2)
    seed_row = H // 2
    if col_is_black(seed_col):
        seed_col = max(0, right_edge - H // 4)
    if col_is_black(seed_col):
        return None

    top = seed_row
    for r in range(seed_row, -1, -1):
        if row_is_black(r):
            break
        top = r
    bottom = seed_row
    for r in range(seed_row, H):
        if row_is_black(r):
            break
        bottom = r
    right = seed_col
    for c in range(seed_col, W):
        if col_is_black(c):
            break
        right = c
    left = seed_col
    for c in range(seed_col, -1, -1):
        if col_is_black(c):
            break
        left = c

    if right <= left:
        return None
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def estimate_roi(frame_paths, n_samples=10):
    """Conservative percentiles of detect_roi over a few frames (preview_boxes.py)."""
    idx = np.linspace(0, len(frame_paths) - 1,
                      min(n_samples, len(frame_paths)), dtype=int)
    rois = []
    for i in idx:
        try:
            r = detect_roi(read_mhd(frame_paths[i]))
        except Exception:
            r = None
        if r:
            rois.append(r)
    if not rois:
        return None
    return {
        "top": int(np.percentile([r["top"] for r in rois], 75)),
        "bottom": int(np.percentile([r["bottom"] for r in rois], 25)),
        "left": int(np.percentile([r["left"] for r in rois], 75)),
        "right": int(np.percentile([r["right"] for r in rois], 25)),
    }


def _scope_raw(paths, roi, black=15):
    """
    Pixels inside the octagonal scope image, before erosion: bright in at least
    one sampled frame (the scope moves, so every image point is bright sometime).
    """
    acc = None
    for p in paths:
        g = crop(read_mhd(p), roi).mean(axis=2) > black
        acc = g if acc is None else (acc | g)
    lab, n = ndimage.label(acc)
    if n > 1:                                   # keep the largest region only
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        acc = lab == sizes.argmax()
    return ndimage.binary_fill_holes(acc)


def roi_and_mask(frames, cache_dir, cache_key, erode=5):
    """
    (roi, mask) for a video, mask in ROI-crop coordinates. Estimated once from
    N_SAMPLES frames spread over the whole video and cached as
    <cache_dir>/<cache_key>_scope.npz.
    """
    nrs = sorted(frames)
    idx = np.linspace(0, len(nrs) - 1, min(N_SAMPLES, len(nrs)), dtype=int)
    paths = [frames[nrs[i]] for i in idx]

    cache = os.path.join(cache_dir, f"{cache_key}_scope.npz")
    if os.path.isfile(cache):
        with np.load(cache) as z:
            roi = dict(zip(("left", "right", "top", "bottom"),
                           (int(v) for v in z["roi"])))
            raw = z["mask"]
    else:
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise LumenError(f"Frame file not found: {missing[0]}")
        roi = estimate_roi(paths, n_samples=len(paths))
        if roi is None:
            raise LumenError("Could not find the scope image (ROI) in this video")
        raw = _scope_raw(paths, roi)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache}.{secrets.token_hex(4)}.tmp.npz"
        np.savez_compressed(tmp, mask=raw, roi=np.array(
            [roi["left"], roi["right"], roi["top"], roi["bottom"]]))
        os.replace(tmp, cache)                  # atomic: no half-written cache
    mask = ndimage.binary_erosion(raw, iterations=erode) if erode > 0 else raw
    return roi, mask


def cache_key_for(image_sequence):
    """Cache name that changes if the sequence is re-pointed to other files."""
    h = hashlib.sha1(image_sequence.format.encode("utf-8")).hexdigest()[:10]
    return f"seq{image_sequence.pk}_{h}"


# ── level 1 ───────────────────────────────────────────────────────────────────

def kmeans_1d(hist, k, iters=100):
    """
    Exact Lloyd's K-means on 8-bit intensities, run on the 256-bin histogram.
    Initialised at intensity quantiles, so it is deterministic.
    """
    levels = np.arange(256, dtype=float)
    cdf = np.cumsum(hist) / hist.sum()
    centers = np.array([np.searchsorted(cdf, q)
                        for q in (np.arange(k) + 0.5) / k], dtype=float)
    for _ in range(iters):
        assign = np.abs(levels[:, None] - centers[None, :]).argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            w = hist[assign == j]
            if w.sum() > 0:
                new[j] = (w * levels[assign == j]).sum() / w.sum()
        if np.allclose(new, centers):
            break
        centers = new
    return np.sort(centers)


def detect_level1(img, roi, mask, p):
    """Level-1 boxes (x, y, w, h) in source-frame pixels, plus blob labels."""
    gray = crop(img, roi).mean(axis=2)
    if p.blur > 0:
        gray = ndimage.gaussian_filter(gray, p.blur)
    gray = np.clip(np.rint(gray), 0, 255).astype(np.uint8)

    hist = np.bincount(gray[mask], minlength=256).astype(float)
    if hist.sum() == 0:
        return {"gray": gray, "labels": None, "kept": [], "boxes": []}
    centers = kmeans_1d(hist, p.k)
    cut = (centers[0] + centers[1]) / 2.0      # darkest cluster boundary
    dark = (gray < cut) & mask
    if p.open > 0:
        dark = ndimage.binary_opening(dark, iterations=p.open)

    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))   # 8-connected
    scope_px = mask.sum()
    min_px, max_px = p.min_area * scope_px, p.max_area * scope_px
    areas = np.bincount(lab.ravel()) if n else np.zeros(1, dtype=int)

    kept, boxes = [], []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        if min_px < areas[i] <= max_px:
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            kept.append(i)
            boxes.append((roi["left"] + x0, roi["top"] + y0, x1 - x0, y1 - y0))
    return {"gray": gray, "labels": lab, "kept": kept, "boxes": boxes}


# ── level 2 ───────────────────────────────────────────────────────────────────

def split_blob(gray, blob, roi, p):
    """K-means inside one level-1 blob. Returns child boxes (x, y, w, h)."""
    hist = np.bincount(gray[blob], minlength=256).astype(float)
    centers = kmeans_1d(hist, p.k2)
    if centers[1] - centers[0] < p.min_gap:
        return []
    cut = (centers[0] + centers[1]) / 2.0

    dark = blob & (gray < cut)
    if p.open2 > 0:
        dark = ndimage.binary_opening(dark, iterations=p.open2)
    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
    parent_area = blob.sum()
    ys, xs = np.nonzero(blob)
    pbox = (roi["left"] + xs.min(), roi["top"] + ys.min(),
            xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)

    children = []
    areas = np.bincount(lab.ravel()) if n else [0]
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        frac = areas[i] / parent_area
        if frac < p.child_min or frac > p.child_max:
            continue
        box = (roi["left"] + sl[1].start, roi["top"] + sl[0].start,
               sl[1].stop - sl[1].start, sl[0].stop - sl[0].start)
        if iou_matrix([box], [pbox])[0, 0] >= 0.8:      # ≈ parent box
            continue
        children.append(box)
    return children


def detect(img, roi, mask, p):
    """{"level1": [box], "level2": [(box, parent_index)]}."""
    s1 = detect_level1(img, roi, mask, p)
    out = {"level1": list(s1["boxes"]), "level2": []}
    if p.levels < 2:
        return out
    for pi, comp_id in enumerate(s1["kept"]):
        kids = split_blob(s1["gray"], s1["labels"] == comp_id, roi, p)
        if len(kids) < p.min_children:
            continue
        for box in kids:
            out["level2"].append((box, pi))
    return out


# ── matching and tracking ─────────────────────────────────────────────────────

def iou_matrix(a, b):
    """IoU between two lists of (x, y, w, h) boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) -
                 np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) -
                 np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def match(a, b, thr):
    """Hungarian matching on IoU. Returns [(i, j, iou)] with iou >= thr."""
    m = iou_matrix(a, b)
    if m.size == 0:
        return []
    r, c = linear_sum_assignment(-m)
    return [(i, j, m[i, j]) for i, j in zip(r, c) if m[i, j] >= thr]


class _IdCounter:
    """Shared ID source for both trackers; its value is part of the run state."""

    def __init__(self, next_id=1):
        self.next_id = next_id

    def __call__(self):
        v = self.next_id
        self.next_id += 1
        return v


class IoUTracker:
    """
    The BronchoTop tracker: match new boxes to each track's last box by IoU
    (Hungarian, threshold iou_thr). Unmatched boxes start new tracks; a track
    unmatched for more than max_lost frames is removed. No motion model.
    """

    def __init__(self, iou_thr, max_lost, new_id, tracks=None):
        self.iou_thr, self.max_lost, self.new_id = iou_thr, max_lost, new_id
        # id -> {"box": last box, "lost": frames unmatched, "n": frames seen}
        self.tracks = tracks if tracks is not None else {}

    def reset(self):
        self.tracks = {}

    def update(self, boxes):
        """Returns a track ID for each box, in order."""
        ids = list(self.tracks)
        prev = [self.tracks[t]["box"] for t in ids]
        out = [None] * len(boxes)
        matched = set()
        for i, j, _ in match(prev, boxes, self.iou_thr):
            t = ids[i]
            self.tracks[t] = {"box": boxes[j], "lost": 0, "n": self.tracks[t]["n"] + 1}
            out[j] = t
            matched.add(t)
        for t in ids:
            if t not in matched:
                self.tracks[t]["lost"] += 1
                if self.tracks[t]["lost"] > self.max_lost:
                    del self.tracks[t]
        for j, b in enumerate(boxes):
            if out[j] is None:
                out[j] = self.new_id()
                self.tracks[out[j]] = {"box": b, "lost": 0, "n": 1}
        return out

    def dump(self):
        return {str(t): {"box": [int(v) for v in d["box"]], "lost": d["lost"], "n": d["n"]}
                for t, d in self.tracks.items()}

    @staticmethod
    def load_tracks(d):
        out = {}
        for t, x in (d or {}).items():
            box = tuple(int(v) for v in x["box"])
            if len(box) != 4:
                raise ValueError("track box must have 4 values")
            out[int(t)] = {"box": box, "lost": int(x["lost"]), "n": int(x.get("n", 1))}
        return out


# ── runner ────────────────────────────────────────────────────────────────────

def new_run_id():
    """Time + random part, e.g. 20261113-132401-a3f9."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


class Runner:
    """
    Detection (+ tracking) over frames in increasing order. Equivalent to
    lumen_core.Runner, but the tracker state can be saved with state() and
    restored by passing it back in, so a run can span several HTTP requests.
    A state from a run with other params is ignored and a new run starts.
    """

    def __init__(self, frames, roi, mask, params, state=None):
        self.frames, self.roi, self.mask, self.p = frames, roi, mask, params
        self.dropped = 0        # boxes dropped by min_track_len in the last process()
        try:
            self._load(state if self._compatible(state, params) else None)
        except (KeyError, TypeError, ValueError, AttributeError):
            self._load(None)        # malformed state from the browser: new run

    def _load(self, state):
        params = self.p
        if state:
            self.run_id = state["run_id"]
            prev = state.get("prev")
            self.prev = None if prev is None else int(prev)
            self._ids = _IdCounter(int(state["next_id"]))
            t1 = IoUTracker.load_tracks(state["tracks"].get("1"))
            t2 = IoUTracker.load_tracks(state["tracks"].get("2"))
        else:
            self.run_id, self.prev = new_run_id(), None
            self._ids = _IdCounter(1)
            t1, t2 = {}, {}
        self.trackers = {1: IoUTracker(params.track_iou, params.max_lost, self._ids, t1),
                         2: IoUTracker(params.iou2, params.max_lost, self._ids, t2)}

    @staticmethod
    def _compatible(state, params):
        try:
            return bool(state) and state["params"] == asdict(params) \
                and isinstance(state["run_id"], str) and int(state["next_id"]) >= 1
        except (KeyError, TypeError, ValueError):
            return False

    def state(self):
        return {"run_id": self.run_id, "params": asdict(self.p), "prev": self.prev,
                "next_id": self._ids.next_id,
                "tracks": {"1": self.trackers[1].dump(), "2": self.trackers[2].dump()}}

    def _check_size(self, img, f):
        """The frame must fit the scope mask estimated from the rest of the video."""
        if crop(img, self.roi).shape[:2] != self.mask.shape:
            raise LumenError(f"Frame {f} has a different size than the rest of the video "
                             f"({img.shape[1]}x{img.shape[0]})")

    def frame(self, f):
        """Rows (box dicts) for one frame. Call in increasing frame order."""
        p = self.p
        path = self.frames.get(f)
        if path is None or not os.path.isfile(path):
            raise LumenError(f"Frame {f} not found")
        img = read_mhd(path)
        self._check_size(img, f)
        det = detect(img, self.roi, self.mask, p)
        l1, l2 = det["level1"], det["level2"]
        if p.track:
            if self.prev is None or f != self.prev + 1:
                for t in self.trackers.values():
                    t.reset()                       # never link across a gap
            id1 = self.trackers[1].update(l1)
            id2 = self.trackers[2].update([b for b, _ in l2]) if p.levels > 1 else []
            prefix = "T"
        else:
            id1 = list(range(1, len(l1) + 1))
            id2 = list(range(len(l1) + 1, len(l1) + len(l2) + 1))
            prefix = "D"
        self.prev = f

        rows = [_row(f, prefix, box, 1, bid, None, self.run_id)
                for box, bid in zip(l1, id1)]
        rows += [_row(f, prefix, box, 2, bid, id1[pi], self.run_id)
                 for (box, pi), bid in zip(l2, id2)]
        return rows

    def process(self, frame_nrs):
        """
        Rows for consecutive frames. With tracking and min_track_len > 1, main
        (level-1) boxes whose track appears in fewer than min_track_len frames
        are dropped as noise, together with their children. Nested (level-2)
        tracks are not filtered: small boxes overlap too little between frames
        for their tracks to be reliable.

        Whether a box on the last frames of the chunk is kept can depend on
        later frames, so the tracker also looks ahead past the chunk: a track
        that reaches min_track_len frames does so within
        (min_track_len - 1) * (max_lost + 1) frames. The look-ahead rows are
        not returned and the saved state is the one after the chunk, so the
        next chunk (starting at the next frame) gives exactly the same result
        as one long run. Frames seen in earlier chunks count via the state.
        """
        frame_nrs = list(frame_nrs)
        p = self.p
        self.dropped = 0
        if not frame_nrs:
            return []
        filtering = p.track and p.min_track_len > 1
        continues = self.prev is not None and frame_nrs[0] == self.prev + 1
        # level-1 track id -> frames seen before this chunk
        before = ({tid: d["n"] for tid, d in self.trackers[1].tracks.items()}
                  if filtering and continues else {})

        rows = []
        for f in frame_nrs:
            rows += self.frame(f)
        if not filtering:
            return rows

        snapshot = self.state()
        ahead = []
        f = frame_nrs[-1]
        for _ in range((p.min_track_len - 1) * (p.max_lost + 1)):
            f += 1
            if f not in self.frames:
                break                     # end of the video
            ahead += self.frame(f)
        self._load(snapshot)

        seen = dict(before)
        for r in rows + ahead:
            if r["level"] == 1:
                seen[r["box_id"]] = seen.get(r["box_id"], 0) + 1
        parents = {(r["frame_nr"], r["box_id"]) for r in rows
                   if r["level"] == 1 and seen[r["box_id"]] >= p.min_track_len}
        out = [r for r in rows if (r["frame_nr"], r["box_id"] if r["level"] == 1
                                   else r["parent_id"]) in parents]
        self.dropped = len(rows) - len(out)
        return out


def _row(f, prefix, box, level, box_id, parent_id, run_id):
    x, y, w, h = (int(v) for v in box)
    return {"frame_nr": f, "label": f"{prefix}{box_id}", "x": x, "y": y,
            "width": w, "height": h,
            "color": L1_COLOR if level == 1 else L2_COLOR, "level": level,
            "box_id": int(box_id), "parent_id": parent_id, "run_id": run_id}
