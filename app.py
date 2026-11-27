import html
import math
import re
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_extractor import FeatureExtractor
from sorters import ALGORITHMS

MODEL_FILE = "rf_sorting_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
MAX_ARRAY = 20000
MAX_BENCHMARK = 5000

st.set_page_config(page_title="SmartSort.ai", page_icon="📐", layout="wide")

ALGO_NOTES = {
    "InsertionSort": "Shifts each element into place. Very low overhead, best on tiny or nearly sorted data, slow on large shuffled data.",
    "QuickSort": "Partitions in place around a pivot. Very fast on shuffled data, but slows badly on sorted or heavily repeated values.",
    "MergeSort": "Splits the array in halves, sorts each, and merges them. Predictable O(n log n) on any input.",
    "HeapSort": "Builds a heap and extracts the largest value repeatedly. Steady O(n log n), and it copes well with heavy repetition.",
    "CountingSort": "Counts how often each value occurs. Very fast when the range of values is small.",
    "TimSort": "Finds runs that are already ordered and merges them. The algorithm behind Python's built-in sort.",
}

# ------------------------------------------------------------------ helpers
def H(s):
    """Collapse HTML to one line so Markdown never mistakes it for code."""
    return " ".join(line.strip() for line in s.strip().splitlines())


def md(s):
    st.markdown(H(s), unsafe_allow_html=True)


def disp(name):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", str(name))


def fmt_num(v):
    v = float(v)
    return f"{int(v):,}" if v.is_integer() else f"{v:,.2f}"


def fmt_t(t):
    ms = t * 1000
    return f"{ms * 1000:.0f} µs" if ms < 1 else f"{ms:,.2f} ms"


def sec(n, title, note=""):
    md(
        f'<div class="sec"><span class="n">{n}</span><span class="t">{title}</span>'
        f'<span class="rule"></span><span class="note">{note}</span></div>'
    )


# ------------------------------------------------------------------ styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
:root{
  --bg:#141B1F; --card:#1B2429; --panel:rgba(27,36,41,.82);
  --ink:#DAE3E1; --mute:#8DA2A8; --line:#344850; --faint:rgba(140,180,175,.075);
  --hi:#6CC4AC; --cy:#5F8799; --cyd:#8DB4C7; --warn:#D9A65E;
}
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stButton button { font-family:'Barlow', sans-serif; }
.stApp{
  background-color:var(--bg);
  background-image:
    linear-gradient(var(--faint) 1px, transparent 1px), linear-gradient(90deg, var(--faint) 1px, transparent 1px),
    linear-gradient(rgba(140,180,175,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(140,180,175,.035) 1px, transparent 1px);
  background-size:120px 120px,120px 120px,24px 24px,24px 24px;
  background-attachment:fixed;
  color:var(--ink);
}
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ max-width:1120px; padding-top:2.4rem; padding-bottom:4rem; }
label p, [data-testid="stWidgetLabel"] p{ color:var(--mute)!important; font-size:.95rem; }

/* masthead */
.top{ display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:2rem; align-items:end; margin-bottom:2.2rem; }
.ss-h1{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:clamp(3rem,7vw,4.8rem); line-height:1.02; letter-spacing:.005em; color:var(--ink); }
.lede{ color:var(--mute); font-size:1.08rem; line-height:1.55; max-width:58ch; margin:1rem 0 0 0; }
.tb{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); border:1px solid var(--line); }
.tb div{ background:var(--card); padding:.55rem .8rem; }
.tb span{ display:block; color:var(--mute); font-size:.82rem; }
.tb b{ font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:1.35rem; letter-spacing:.01em; color:var(--ink); }

/* section headings */
.sec{ display:flex; align-items:center; gap:.8rem; margin:2.4rem 0 1rem 0; }
.sec .n{ width:1.7rem; height:1.7rem; border:1.5px solid var(--hi); color:var(--hi); border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:1.05rem; flex:none; }
.sec .t{ font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:1.7rem; letter-spacing:.01em; white-space:nowrap; color:var(--ink); }
.sec .rule{ flex:1; height:9px; border-top:1px solid var(--line); background:repeating-linear-gradient(90deg,var(--line) 0 1px,transparent 1px 12px) top/100% 5px no-repeat; }
.sec .note{ color:var(--mute); font-size:.9rem; white-space:nowrap; }

/* sheets */
.sheet{ position:relative; background:var(--panel); border:1px solid var(--line); border-radius:2px; padding:1.1rem 1.3rem; box-shadow:0 2px 10px rgba(0,0,0,.18); }
.sheet::before,.sheet::after{ content:""; position:absolute; width:11px; height:11px; border:2px solid var(--hi); }
.sheet::before{ top:-2px; left:-2px; border-right:0; border-bottom:0; }
.sheet::after{ bottom:-2px; right:-2px; border-left:0; border-top:0; }
.sheet .cap{ color:var(--mute); font-size:.9rem; margin-bottom:.4rem; }
svg.prof{ width:100%; height:auto; display:block; }
svg.prof rect{ fill:var(--cy); }
svg.prof .base{ stroke:var(--ink); stroke-width:1; }
svg.prof .dim{ stroke:var(--mute); stroke-width:1; fill:none; }
svg.prof text{ fill:var(--mute); font-family:'Barlow',sans-serif; font-size:10.5px; font-variant-numeric:tabular-nums; }
svg.prof text.v{ fill:var(--ink); font-size:10px; text-anchor:middle; }
svg.prof text.ix{ text-anchor:middle; }

/* hero */
.duo{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:1.2rem; }
.hero{ display:flex; gap:1rem; align-items:center; justify-content:space-between; min-height:210px; }
.hero .k{ color:var(--mute); font-size:.95rem; }
.hero .algo{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:clamp(2.8rem,5.5vw,4.2rem); line-height:1; color:var(--hi); margin:.25rem 0 .55rem 0; }
.hero .why{ color:var(--ink); font-size:.98rem; line-height:1.45; max-width:36ch; }
.hero .alt{ color:var(--mute); font-size:.9rem; margin-top:.7rem; }
svg.gauge{ width:170px; flex:none; }
svg.gauge .tr{ fill:none; stroke:#2A3A41; stroke-width:9; }
svg.gauge .vl{ fill:none; stroke:var(--hi); stroke-width:9; }
svg.gauge .tk{ stroke:var(--mute); stroke-width:1; }
svg.gauge .gn{ fill:var(--ink); font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:25px; text-anchor:middle; }
svg.gauge .gl{ fill:var(--mute); font-family:'Barlow',sans-serif; font-size:8.5px; text-anchor:middle; }
.votes .row{ display:grid; grid-template-columns:118px 1fr 40px; gap:.7rem; align-items:center; margin:.55rem 0; }
.votes .nm{ font-size:.98rem; }
.votes .tr{ height:12px; background:repeating-linear-gradient(90deg,var(--faint) 0 1px,transparent 1px 10%); border-right:1px solid var(--line); border-left:1px solid var(--line); position:relative; }
.votes .fl{ position:absolute; left:0; top:2px; bottom:2px; background:var(--cy); }
.votes .row.top .fl{ background:var(--hi); }
.votes .pc{ text-align:right; color:var(--mute); font-variant-numeric:tabular-nums; font-size:.92rem; }

/* measurements */
.grid3{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }
.m{ background:var(--card); padding:.9rem 1.05rem 1rem 1.05rem; }
.m-top{ display:flex; justify-content:space-between; align-items:baseline; gap:.5rem; }
.m-l{ color:var(--mute); font-size:.95rem; }
.m-v{ font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:1.9rem; font-variant-numeric:tabular-nums; line-height:1; color:var(--ink); }
.scale{ position:relative; height:9px; margin:.7rem 0 .55rem 0; border-bottom:1px solid var(--ink); background:repeating-linear-gradient(90deg,var(--line) 0 1px,transparent 1px 25%); border-right:1px solid var(--line); }
.scale.plain{ background:none; }
.scale i{ position:absolute; top:-3px; width:9px; height:9px; margin-left:-4.5px; background:var(--hi); transform:rotate(45deg); }
.m-n{ font-size:.9rem; line-height:1.35; color:var(--ink); }
.m .two{ display:flex; justify-content:space-between; font-size:.95rem; margin-top:.35rem; color:var(--mute); }
.m .two b{ color:var(--ink); font-weight:500; font-variant-numeric:tabular-nums; }

/* lanes */
.lane{ display:grid; grid-template-columns:2rem 190px minmax(0,1fr) 84px; gap:.7rem; align-items:center; margin:.62rem 0; }
.lane .rk{ color:var(--mute); font-family:'Barlow Condensed',sans-serif; font-size:1.15rem; text-align:right; }
.lane .nm{ font-size:1rem; white-space:nowrap; color:var(--ink); }
.tag{ font-size:.72rem; border:1px solid; padding:0 .35rem; margin-left:.45rem; border-radius:2px; vertical-align:2px; }
.tag.pick{ color:var(--hi); border-color:var(--hi); }
.tag.fast{ color:var(--cyd); border-color:var(--cyd); }
.l-track{ height:24px; position:relative; background:repeating-linear-gradient(90deg,var(--faint) 0 1px,transparent 1px 10%); border-left:1px solid var(--line); border-right:2px solid var(--line); }
.l-bar{ height:100%; background:var(--cy); transform-origin:left center; animation:run cubic-bezier(.2,.7,.3,1) both; }
.lane.pk .l-bar{ background:var(--hi); }
.lane .tm{ text-align:right; font-variant-numeric:tabular-nums; font-size:.98rem; color:var(--ink); }
.lane .dnf{ color:var(--mute); font-size:.9rem; padding-left:.6rem; line-height:24px; }
@keyframes run{ from{ transform:scaleX(0); } to{ transform:scaleX(1); } }
.axis{ margin:.9rem 0 0 calc(2rem + 190px + 1.4rem); color:var(--mute); font-size:.85rem; }
.verdict{ margin-top:1.1rem; padding:.8rem 1rem; border:1px solid var(--line); border-left:4px solid var(--warn); background:var(--card); font-size:1rem; line-height:1.45; color:var(--ink); }
.verdict.match{ border-left-color:var(--hi); }
.exlbl{ color:var(--mute); font-size:.95rem; margin:1.1rem 0 .4rem 0; }
.fine{ color:var(--mute); font-size:.88rem; margin-top:.6rem; }

/* title block footer */
.foot{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin-top:3rem; }
.foot div{ background:var(--card); padding:.6rem .85rem; }
.foot span{ display:block; color:var(--mute); font-size:.82rem; }
.foot b{ font-weight:500; font-size:.98rem; color:var(--ink); }

/* streamlit widgets */
div[data-baseweb="textarea"], div[data-baseweb="base-input"]{ background:#0F1518!important; border:1px solid var(--line)!important; border-radius:2px!important; }
div[data-baseweb="textarea"]:focus-within{ border-color:var(--hi)!important; box-shadow:0 0 0 1px var(--hi)!important; }
.stTextArea textarea{ background:transparent!important; color:var(--ink)!important; font-size:1.05rem; font-variant-numeric:tabular-nums; }
.stButton>button{ border-radius:2px; border:1px solid var(--line); background:var(--card); color:var(--ink); font-weight:500; padding:.3rem .5rem; transition:border-color .15s, color .15s; }
.stButton>button p{ white-space:nowrap; font-size:.95rem; }
.stButton>button:hover{ border-color:var(--hi); color:var(--hi); }
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"]{ background:var(--hi); border-color:var(--hi); color:#0E1719; font-weight:600; }
.stButton>button[kind="primary"]:hover, .stButton>button[data-testid="stBaseButton-primary"]:hover{ color:#0E1719; filter:brightness(1.08); }
[data-testid="stExpander"] details{ background:var(--card); border:1px solid var(--line); border-radius:2px; }
[data-testid="stExpander"] summary p{ color:var(--ink); }

@media (max-width: 820px){
  .top, .duo{ grid-template-columns:1fr; }
  .grid3{ grid-template-columns:1fr; }
  .foot{ grid-template-columns:1fr 1fr; }
  .sec .note{ display:none; }
  .lane{ grid-template-columns:1.6rem 120px minmax(0,1fr) 70px; gap:.4rem; }
  .axis{ margin-left:0; }
  .hero{ flex-direction:column; align-items:flex-start; }
}
@media (prefers-reduced-motion: reduce){ .l-bar{ animation:none; } }
</style>
"""
md(CSS)


# ------------------------------------------------------------------ data + model
@st.cache_resource(show_spinner=False)
def load_brain():
    return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE)


def parse_array(text):
    out = []
    for tok in [t for t in re.split(r"[,\s;]+", text.strip()) if t]:
        v = float(tok)
        out.append(int(v) if v.is_integer() else v)
    return out


def make_preset(kind, n):
    rng = np.random.default_rng()
    if kind == "Random":
        return rng.integers(0, n * 10, n).tolist()
    if kind == "Nearly sorted":
        arr = list(range(n))
        for _ in range(max(1, int(n * 0.05))):
            i, j = rng.choice(n, 2, replace=False)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    if kind == "Reversed":
        return list(range(n, 0, -1))
    if kind == "Duplicates":
        return rng.choice(rng.integers(0, 50, 8), n).tolist()
    return np.round(rng.normal(0, 1000, n)).astype(int).tolist()


def _ex_insertion():
    rng = np.random.default_rng()
    n = int(rng.integers(10, 17))
    a = list(range(1, n + 1))
    for _ in range(int(rng.integers(1, 3))):
        i = int(rng.integers(0, n - 1))
        a[i], a[i + 1] = a[i + 1], a[i]
    return a


EXAMPLES = {
    "InsertionSort": ("Tiny, nearly sorted", _ex_insertion),
    "QuickSort": ("400 random numbers", lambda: np.random.default_rng().integers(0, 10**9, 400).tolist()),
    "HeapSort": ("Repeats and outliers", lambda: np.minimum(np.random.default_rng().zipf(1.5, 1500), 10**9).tolist()),
    "CountingSort": ("800 ratings, 0 to 9", lambda: np.random.default_rng().integers(0, 10, 800).tolist()),
    "TimSort": ("1,000 reversed", lambda: list(range(1000, 0, -1))),
}


def set_example(key):
    st.session_state.arr_input = ", ".join(map(str, EXAMPLES[key][1]()))
    st.session_state.expect = key
    st.session_state.go = True


def set_preset(kind):
    st.session_state.expect = None
    n = st.session_state.get("n_size", 40)
    st.session_state.arr_input = ", ".join(map(str, make_preset(kind, n)))
    st.session_state.go = True


def benchmark(arr):
    times = {}
    reps = 3 if len(arr) <= 2000 else 1
    span = max(arr) - min(arr)
    for name, fn in ALGORITHMS.items():
        best = float("inf")
        if name == "CountingSort" and span > 5_000_000:
            times[name] = best
            continue
        for _ in range(reps):
            data = list(arr)
            t0 = time.perf_counter()
            try:
                fn(data)
            except Exception:
                best = float("inf")
                break
            best = min(best, time.perf_counter() - t0)
        times[name] = best
    return times


def analyse(arr):
    feats = FeatureExtractor.extract(arr)
    cols = list(getattr(model, "feature_names_in_", feats.keys()))
    X = pd.DataFrame([feats])[cols]
    probs = model.predict_proba(X)[0]
    names = encoder.inverse_transform(model.classes_)
    ranked = sorted(zip(map(str, names), map(float, probs)), key=lambda t: -t[1])
    times = benchmark(arr) if len(arr) <= MAX_BENCHMARK else None
    return {"arr": arr, "feats": feats, "ranked": ranked, "times": times}


# ------------------------------------------------------------------ svg + html builders
def profile_svg(arr, w=720, h=160):
    n = len(arr)
    vals = np.asarray(arr, dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    if n > 96:
        vb = np.array([vals[i].mean() for i in np.array_split(np.arange(n), 96)])
    else:
        vb = vals
    m = len(vb)
    pl, pr, pt, pb = 8, 92, 18, 22
    pw, ph = w - pl - pr, h - pt - pb
    bw = pw / m
    gap = 2 if m <= 40 else 1
    base = pt + ph
    parts = []
    for i, v in enumerate(vb):
        bh = 6 + (v - lo) / span * (ph - 6)
        x, y = pl + i * bw, base - bh
        parts.append(f'<rect x="{x + gap / 2:.2f}" y="{y:.2f}" width="{max(bw - gap, .8):.2f}" height="{bh:.2f}"/>')
        if m <= 16:
            parts.append(f'<text class="v" x="{x + bw / 2:.2f}" y="{y - 4:.2f}">{fmt_num(arr[i])}</text>')
            parts.append(f'<text class="ix" x="{x + bw / 2:.2f}" y="{h - 7}">{i + 1}</text>')
    if m > 16:
        parts.append(f'<text x="{pl}" y="{h - 7}">1</text>')
        parts.append(f'<text x="{pl + pw}" y="{h - 7}" text-anchor="end">{n:,}</text>')
    parts.append(f'<line class="base" x1="{pl}" y1="{base}" x2="{pl + pw}" y2="{base}"/>')
    dx = w - pr + 22
    parts.append(f'<path class="dim" d="M{dx} {pt} V{base} M{dx - 4} {pt} H{dx + 4} M{dx - 4} {base} H{dx + 4}"/>')
    parts.append(f'<text x="{dx + 9}" y="{pt + 4}">max {fmt_num(hi)}</text>')
    parts.append(f'<text x="{dx + 9}" y="{base + 1}">min {fmt_num(lo)}</text>')
    parts.append(f'<text x="{dx + 9}" y="{(pt + base) / 2 + 4:.0f}">n = {n:,}</text>')
    return f'<svg class="prof" viewBox="0 0 {w} {h}" role="img" aria-label="Bar profile of the array, {n} elements">{"".join(parts)}</svg>'


def gauge_svg(p):
    L = math.pi * 50
    ticks = ""
    for k in range(5):
        a = math.pi * (1 - k / 4)
        x1, y1 = 60 + 55 * math.cos(a), 62 - 55 * math.sin(a)
        x2, y2 = 60 + 60 * math.cos(a), 62 - 60 * math.sin(a)
        ticks += f'<line class="tk" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
    return (
        '<svg class="gauge" viewBox="0 0 120 76" role="img" aria-label="Model confidence">'
        '<path class="tr" d="M10 62 A50 50 0 0 1 110 62"/>'
        f'<path class="vl" d="M10 62 A50 50 0 0 1 110 62" style="stroke-dasharray:{p * L:.2f} {L:.2f}"/>'
        f'{ticks}<text class="gn" x="60" y="56">{p:.0%}</text><text class="gl" x="60" y="71">model confidence</text></svg>'
    )


def meter(label, value, frac, note, ticks=True):
    pos = max(0.0, min(1.0, frac)) * 100
    cls = "scale" if ticks else "scale plain"
    return (
        f'<div class="m"><div class="m-top"><span class="m-l">{label}</span><span class="m-v">{value}</span></div>'
        f'<div class="{cls}"><i style="left:{pos:.1f}%"></i></div><div class="m-n">{note}</div></div>'
    )


def measurements_html(f):
    n = int(f["size"])
    ent_max = math.log(n) if n > 1 else 1.0
    ent_norm = f["entropy"] / ent_max if ent_max else 0.0
    runs, inv, dup = f["normalized_runs"], f["inversion_ratio"], f["duplicate_ratio"]

    size_note = (
        "Tiny array. Simple sorts carry almost no overhead." if n < 32 else
        "Small array." if n < 500 else
        "Medium array." if n < 5000 else
        "Large array. The algorithm choice matters most here."
    )
    ent_note = (
        "Nearly every value is distinct." if ent_norm >= 0.95 else
        "Some values repeat." if ent_norm >= 0.6 else
        "Heavy repetition of a few values."
    )
    run_note = (
        "Long ordered stretches." if runs <= 0.1 else
        "Some ordered stretches." if runs <= 0.4 else
        "Direction changes often, like shuffled data."
    )
    inv_note = (
        "Almost sorted." if inv < 0.05 else
        "Mostly in order." if inv < 0.35 else
        "Typical of shuffled data." if inv < 0.65 else
        "Mostly reversed."
    )
    dup_note = (
        "No notable duplicates." if dup < 0.05 else
        "Some duplicates." if dup < 0.4 else
        "Duplicate-heavy."
    )
    var = f["variance"]
    var_txt = f"{var:,.0f}" if var >= 100 else f"{var:.2f}"
    cells = [
        meter("Size", f"{n:,}", math.log10(max(n, 1)) / math.log10(MAX_ARRAY), size_note, ticks=False),
        meter("Entropy", f"{f['entropy']:.2f}", ent_norm, f"{ent_note} ({ent_norm:.0%} of the maximum for this size)"),
        meter("Run density", f"{runs:.2f}", runs, run_note),
        meter("Inversions", f"{inv:.2f}", inv, inv_note),
        meter("Duplicates", f"{dup:.0%}", dup, dup_note),
        f'<div class="m"><div class="m-top"><span class="m-l">Value spread</span></div>'
        f'<div class="two"><span>Variance</span><b>{var_txt}</b></div>'
        f'<div class="two"><span>Key range per element</span><b>{f["key_range_ratio"]:,.2f}</b></div>'
        f'<div class="m-n" style="margin-top:.7rem">How far apart the values sit.</div></div>',
    ]
    return f'<div class="grid3">{"".join(cells)}</div>'


def lanes_html(times, pick):
    finite = {k: v for k, v in times.items() if math.isfinite(v)}
    order = sorted(finite, key=finite.get) + [k for k in times if k not in finite]
    tmax = max(finite.values()) if finite else 1.0
    fastest = order[0] if finite else None
    rows = []
    for i, k in enumerate(order, 1):
        tags = ""
        if k == pick:
            tags += '<span class="tag pick">AI pick</span>'
        if k == fastest:
            tags += '<span class="tag fast">Fastest</span>'
        if k in finite:
            frac = max(finite[k] / tmax, 0.012)
            dur = 0.5 + 1.7 * frac
            track = f'<div class="l-track"><div class="l-bar" style="width:{frac * 100:.2f}%;animation-duration:{dur:.2f}s"></div></div>'
            tm = fmt_t(finite[k])
        else:
            track = '<div class="l-track"><span class="dnf">Did not finish</span></div>'
            tm = "n/a"
        rows.append(
            f'<div class="lane {"pk" if k == pick else ""}"><span class="rk">{i}</span>'
            f'<span class="nm">{html.escape(disp(k))}{tags}</span>{track}<span class="tm">{tm}</span></div>'
        )
    return "".join(rows), finite, order, fastest


# ------------------------------------------------------------------ load model
try:
    model, encoder = load_brain()
except FileNotFoundError:
    md(
        f'<div class="verdict">Model files not found. Run <b>python ml_pipeline.py</b> so that '
        f"{MODEL_FILE} and {ENCODER_FILE} sit next to app.py, then reload.</div>"
    )
    st.stop()

model_name = type(model).__name__
model_label = "Random Forest" if "RandomForest" in model_name else model_name
n_trees = len(getattr(model, "estimators_", []))
n_feats = getattr(model, "n_features_in_", 7)
n_classes = len(model.classes_)

# ------------------------------------------------------------------ masthead
md(
    f"""
<div class="top">
  <div>
    <div class="ss-h1">SmartSort.ai</div>
    <p class="lede">Paste an array. A trained model reads its shape and picks the sorting algorithm most likely to run fastest. Then every algorithm is timed on your data, so you can check the call.</p>
  </div>
  <div class="tb">
    <div><span>Model</span><b>{model_label}</b></div>
    <div><span>Trees</span><b>{n_trees}</b></div>
    <div><span>Features read</span><b>{n_feats}</b></div>
    <div><span>Algorithms</span><b>{n_classes}</b></div>
  </div>
</div>
"""
)

# ------------------------------------------------------------------ 1 input
st.session_state.setdefault("arr_input", "84, 12, 99, 3, 45, 76, 22, 9, 58, 1")
st.session_state.setdefault("n_size", 40)

sec(1, "Specimen array", "numbers separated by commas or spaces")
left, right = st.columns([3, 2], gap="large")
with left:
    st.text_area("Array", key="arr_input", height=120, label_visibility="collapsed")
    run_btn = st.button("Analyse array", type="primary", use_container_width=True)
with right:
    st.slider("Size for generated arrays", 10, 500, key="n_size")
    presets = ["Random", "Nearly sorted", "Reversed", "Duplicates", "Gaussian"]
    c1 = st.columns(3)
    c2 = st.columns(3)
    for slot, name in zip(c1 + c2[:2], presets):
        slot.button(name, on_click=set_preset, args=(name,), use_container_width=True)

md('<div class="exlbl">Known cases. Each one is built to favour a different algorithm.</div>')
for col, (algo, (label, _)) in zip(st.columns(5), EXAMPLES.items()):
    col.button(label, key=f"ex_{algo}", on_click=set_example, args=(algo,), use_container_width=True)

try:
    live = parse_array(st.session_state.arr_input)
except ValueError:
    live = None

if live and len(live) >= 2:
    md(f'<div class="sheet"><div class="cap">Profile of the current input</div>{profile_svg(live[:MAX_ARRAY])}</div>')
else:
    md('<div class="sheet"><div class="cap">Enter at least two numbers to see the array profile.</div></div>')

# ------------------------------------------------------------------ run
first = "res" not in st.session_state
if run_btn or st.session_state.pop("go", False) or first:
    if live is None:
        md('<div class="verdict">That input has something other than numbers. Use digits separated by commas or spaces.</div>')
    elif len(live) < 2:
        md('<div class="verdict">Enter at least two numbers.</div>')
    elif len(live) > MAX_ARRAY:
        md(f'<div class="verdict">Keep the array under {MAX_ARRAY:,} elements.</div>')
    else:
        if run_btn:
            st.session_state.expect = None
        with st.spinner("Measuring the array and timing every algorithm"):
            st.session_state.res = analyse(live)
            st.session_state.res["expect"] = st.session_state.get("expect")

res = st.session_state.get("res")
if res is None:
    st.stop()

arr, feats, ranked, times = res["arr"], res["feats"], res["ranked"], res["times"]
pick, pick_p = ranked[0]
n = len(arr)

# ------------------------------------------------------------------ 2 recommendation
sec(2, "Recommendation", f"for the {n:,}-element array analysed")
alt = f'Runner-up: {html.escape(disp(ranked[1][0]))} at {ranked[1][1]:.0%}' if len(ranked) > 1 else ""
expect = res.get("expect")
if expect:
    agree = "The model agreed." if pick == expect else "The model chose differently."
    alt += f'<br>This case was built to favour {html.escape(disp(expect))}. {agree}'
vote_rows = "".join(
    f'<div class="row {"top" if i == 0 else ""}"><span class="nm">{html.escape(disp(nm))}</span>'
    f'<div class="tr"><div class="fl" style="width:{p * 100:.1f}%"></div></div><span class="pc">{p * 100:.0f}%</span></div>'
    for i, (nm, p) in enumerate(ranked)
)
md(
    f"""
<div class="duo">
  <div class="sheet hero">
    <div>
      <div class="k">Recommended algorithm</div>
      <div class="algo">{html.escape(disp(pick))}</div>
      <div class="why">{html.escape(ALGO_NOTES.get(pick, ""))}</div>
      <div class="alt">{alt}</div>
    </div>
    {gauge_svg(pick_p)}
  </div>
  <div class="sheet votes">
    <div class="cap">How the {n_trees} trees voted</div>
    {vote_rows}
  </div>
</div>
"""
)

# ------------------------------------------------------------------ 3 measurements
sec(3, "What the model measured", "marker position shows where the array sits on each scale")
md(measurements_html(feats))

# ------------------------------------------------------------------ 4 verification
sec(4, "Check against real timings", "every algorithm run on your array")
if times is None:
    md(f'<div class="sheet">Timing is skipped above {MAX_BENCHMARK:,} elements so the page stays responsive.</div>')
else:
    rows, finite, order, fastest = lanes_html(times, pick)
    if not finite:
        verdict = '<div class="verdict">No algorithm finished on this array.</div>'
    elif pick == fastest:
        verdict = f'<div class="verdict match"><b>Match.</b> {html.escape(disp(pick))} was the fastest algorithm on this array.</div>'
    elif pick in finite:
        rank = order.index(pick) + 1
        ratio = finite[pick] / finite[fastest]
        verdict = (
            f'<div class="verdict"><b>Close, not exact.</b> The model picked {html.escape(disp(pick))}, which ranked '
            f"{rank} of {len(order)} and ran {ratio:.1f}x slower than the fastest, {html.escape(disp(fastest))}.</div>"
        )
    else:
        verdict = (
            f'<div class="verdict"><b>Miss.</b> {html.escape(disp(pick))} could not sort this array. '
            f"The fastest was {html.escape(disp(fastest))}.</div>"
        )
    noise = '<div class="fine">Arrays under 100 elements sort in microseconds, so timer noise can reorder close results.</div>' if n < 100 else ""
    md(
        f'<div class="sheet">{rows}<div class="axis">Bar length is proportional to run time. Shorter is faster.</div>{verdict}{noise}</div>'
    )

with st.expander("Raw feature values"):
    st.dataframe(pd.DataFrame([feats]), hide_index=True, use_container_width=True)

with st.expander("How SmartSort.ai decides"):
    st.markdown(
        "1. The array is reduced to a few numbers describing its shape: size, entropy, run density, inversion ratio, variance, duplicate ratio and key range.\n"
        "2. A Random Forest trained on arrays that were each timed against every algorithm votes on which one wins for that shape.\n"
        "3. The check at the bottom re-times every algorithm on your actual input, so a wrong call is visible."
    )

# ------------------------------------------------------------------ footer title block
reps = "best of 3 runs (1 above 2,000 elements)"
md(
    f"""
<div class="foot">
  <div><span>Project</span><b>SmartSort.ai</b></div>
  <div><span>Classifier</span><b>{model_label}, {n_trees} trees</b></div>
  <div><span>Inputs</span><b>{n_feats} array features</b></div>
  <div><span>Timing method</span><b>{reps}</b></div>
</div>
"""
)
