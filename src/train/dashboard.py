"""
Live dashboard for a metrics.jsonl.

    python -m src.train.dashboard                                  # runs/nanospeaker, port 8000
    python -m src.train.dashboard runs/other/metrics.jsonl --port 8080
    python -m src.train.dashboard runs/other/metrics.jsonl --snapshot report.html

The positional argument is what a bare URL shows. Any run can be opened on the same server
by naming it in the URL instead:

    http://127.0.0.1:8000/?path=runs/other/metrics.jsonl

so comparing two runs is two tabs, not two processes. The page names the file it is showing
in its header, and carries its own ?path= through to the polling endpoint. Paths from the
URL must live under the directory the server was started in.

Stdlib only, and it reads the log rather than talking to the trainer: the training
process should never wait on, or crash because of, something that draws pictures. The
page polls for rows appended since the last one it holds, so watching a live run costs
one small request every few seconds no matter how long the run has been going.

The panels are the ones metrics.py argues are worth logging, grouped so that each chart
has a single y-scale -- two measures of different magnitude get two charts, never two
axes, because a dual axis lets you place any correlation you like by choosing the scales.
Learning rates share a chart on a log scale: same unit, so it is still one axis.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #dcdad4;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #86847d;
    --grid: #e7e5e0;
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
    --s5: #e87ba4; --s6: #008300; --s7: #4a3aa7; --s8: #e34948;
    --good: #008300; --warn: #eda100; --crit: #e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #111110; --surface-1: #1a1a19; --border: #34332f;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e85;
      --grid: #2a2a27;
      --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
      --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
      --good: #199e70; --warn: #c98500; --crit: #e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #34332f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e85;
    --grid: #2a2a27;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
    --good: #199e70; --warn: #c98500; --crit: #e66767;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 24px 20px 64px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
  h1 { font-size: 19px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
  .sub { color: var(--text-secondary); font-size: 13px; }
  .path {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
    color: var(--text-muted); background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 6px; padding: 2px 7px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); display: inline-block; }
  .dot.live { background: var(--good); }
  .dot.idle { background: var(--warn); }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0 22px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 11px 13px; }
  .tile .k { color: var(--text-secondary); font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; }
  .tile .v { font-size: 22px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
  .tile .n { color: var(--text-muted); font-size: 11.5px; font-variant-numeric: tabular-nums; }
  .tile .v.good { color: var(--good); } .tile .v.warn { color: var(--warn); } .tile .v.crit { color: var(--crit); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 14px; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 6px; position: relative; }
  .card h2 { font-size: 13.5px; font-weight: 600; margin: 0; }
  /* Titles carry the long explanation. CSS-only so it works before the first poll lands,
     and on :focus as well as :hover so it is reachable from the keyboard. */
  .t { cursor: help; border-bottom: 1px dotted var(--text-muted); outline: none; }
  .t:hover, .t:focus { border-bottom-color: var(--text-primary); }
  .pop {
    position: absolute; z-index: 20; left: 14px; top: 30px; width: min(400px, calc(100% - 28px));
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px; font-size: 12.5px; line-height: 1.55; color: var(--text-secondary);
    box-shadow: 0 8px 28px rgba(0,0,0,.22); opacity: 0; visibility: hidden;
    transition: opacity .12s .25s, visibility .12s .25s;   /* the delay is the grace period
       that lets the pointer travel from the title into the panel without it closing */
    text-transform: none; letter-spacing: normal;
  }
  .pop b { color: var(--text-primary); font-weight: 600; }
  .pop p { margin: 0 0 7px; }
  .pop p:last-child { margin: 0; }
  h2:has(.t:hover) ~ .pop, h2:has(.t:focus) ~ .pop,
  .k:has(.t:hover) ~ .pop, .k:has(.t:focus) ~ .pop,
  .pop:hover { opacity: 1; visibility: visible; transition-delay: 0s; }
  .tile { position: relative; }
  .tile .pop { left: 10px; top: 34px; width: 340px; }
  .tiles > .tile:nth-last-child(-n+3) .pop { left: auto; right: 10px; }
  .card .why { color: var(--text-muted); font-size: 11.5px; margin: 2px 0 8px; }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 2px 0 6px; font-size: 11.5px; color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .legend i { width: 11px; height: 3px; border-radius: 2px; display: inline-block; }
  .plot { position: relative; }
  .plot svg { display: block; width: 100%; height: auto; overflow: visible; }
  .tip {
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 7px 9px; font-size: 11.5px; white-space: nowrap; z-index: 5;
    box-shadow: 0 4px 14px rgba(0,0,0,.14); font-variant-numeric: tabular-nums;
  }
  .tip b { font-weight: 600; }
  .tip .ts { color: var(--text-muted); margin-left: 10px; }
  .tip div { display: flex; gap: 10px; justify-content: space-between; }
  .tip i { width: 9px; height: 3px; border-radius: 2px; display: inline-block; margin-right: 5px; vertical-align: middle; }
  .bar { display: flex; gap: 8px; align-items: center; margin-left: auto; }
  button {
    font: inherit; font-size: 12.5px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 7px; padding: 4px 10px; cursor: pointer;
  }
  button:hover { color: var(--text-primary); }
  table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; width: 100%; }
  th, td { text-align: right; padding: 4px 9px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th { color: var(--text-secondary); font-weight: 500; text-align: right; position: sticky; top: 0; background: var(--surface-1); }
  th:first-child, td:first-child { text-align: left; }
  .scroll { overflow: auto; max-height: 340px; }
  details { margin-top: 16px; background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  summary { cursor: pointer; font-size: 13px; font-weight: 600; }
</style>
<div class="wrap">
  <header>
    <h1>__TITLE__</h1>
    <code class="path" title="__PATH__">__PATH__</code>
    <span class="sub"><span class="dot" id="pulse"></span> <span id="status">connecting…</span></span>
    <span class="bar">
      <button id="theme">theme</button>
      <button id="logloss">log loss</button>
      <button id="ghosts">hide earlier runs</button>
    </span>
  </header>
  <div class="tiles" id="tiles"></div>
  <div class="sub" id="note" style="margin:-8px 0 14px"></div>
  <div class="grid" id="charts"></div>
  <details>
    <summary>Table view — last 40 steps</summary>
    <div class="scroll" id="table"></div>
  </details>
</div>
<script>
// Axis and tooltip numbers are read at a glance, so they get significant digits rather
// than a fixed width: 0 prints as "0", 1e-5 stays exponential, 7043 becomes 7.0K.
const F = {
  big: v => v == null ? "—" : Math.abs(v) >= 1e9 ? (v/1e9).toFixed(2)+"B" : Math.abs(v) >= 1e6 ? (v/1e6).toFixed(1)+"M"
        : Math.abs(v) >= 1e4 ? (v/1e3).toFixed(1)+"K" : String(Math.round(v)),
  n: v => {
    if (v == null || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e4) return F.big(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a >= 0.001) return v.toFixed(String(a).length > 6 ? 3 : 4).replace(/0+$/, "").replace(/\.$/, "");
    return v.toExponential(0);
  },
  pct: v => v == null ? "—" : (Math.abs(v) < 0.1 && v !== 0 ? (100*v).toFixed(1) : (100*v).toFixed(0)) + "%",
};

// Runs do not all log the same fields. Pretraining has a router, two optimisers and a source
// mix; SFT has none of those, times its steps as one number, and validates per task domain
// instead. So the panels are built from the keys the file actually contains: a metric the
// trainer never wrote is an absent panel rather than an empty frame, and a new valid_* or
// mix_* source shows up without anyone editing this file.
let KEYS = new Set(), VGROUPS = [], MIXGROUPS = [], SPECS = [];
const DERIVED = /^(b_|_)|^(compute_s|tokens_est)$/;   // fields this page adds, not the log
const has = k => KEYS.has(k);
const nice = g => g.replace(/_/g, " ");
const SLOT = i => `var(--s${(i % 8) + 1})`;

function discover() {
  KEYS = new Set();
  for (const r of raw)
    for (const k in r)
      if (!DERIVED.test(k) && r[k] != null && isFinite(+r[k])) KEYS.add(k);
  const groups = p => [...KEYS].filter(k => k.startsWith(p) && k.length > p.length)
                               .map(k => k.slice(p.length)).sort();
  VGROUPS = groups("valid_").filter(g => g !== "batches" && g !== "loss");
  MIXGROUPS = groups("mix_");
}

/** Seconds for one step, under whichever name the trainer wrote it. */
const stepSec = r => { const v = +r.step_time_s; return isFinite(v) ? v : (+r.step_s || 0); };
const dataSec = r => { const v = +r.data_wait_s; return isFinite(v) ? v : (+r.t_data_s || 0); };
const DATA_K = () => has("data_wait_s") ? "data_wait_s" : has("t_data_s") ? "t_data_s" : null;

/**
 * Two shapes, because two trainers. Where the loop times its own phases the panel is the
 * stacked breakdown; where it reports one number per step there is nothing to stack, and a
 * seven-band chart with everything in "unattributed" would be a lie with a legend.
 */
function timeSpec() {
  const parts = ["t_fwd_s", "t_bwd_s", "t_optim_s", "data_wait_s", "t_eval_s", "t_ckpt_s"];
  if (parts.some(has))
    return {id:"time", title:"Step time", why:"seconds per step and where they went — everything the wall clock covers, including the periodic validations and checkpoint writes", fmt:F.n, stack:1, smooth:2, y:[0,null],
      series:[{k:"b_fwd", label:"forward", c:"var(--s1)", derived:1},
              {k:"b_bwd", label:"backward", c:"var(--s2)", derived:1},
              {k:"b_optim", label:"optimiser", c:"var(--s3)", derived:1},
              {k:"b_data", label:"waiting on the loader", c:"var(--s4)", derived:1},
              {k:"b_eval", label:"validation", c:"var(--s5)", derived:1},
              {k:"b_ckpt", label:"checkpoint", c:"var(--s6)", derived:1},
              {k:"b_other", label:"unattributed", c:"var(--s7)", derived:1}]};
  if (!has("step_time_s") && !has("step_s")) return null;
  return {id:"time", title:"Step time", help:"time_flat", fmt:F.n, smooth:5, y:[0,null],
    why:"seconds per step, whole — this trainer does not say where they went",
    series:[{k:"b_step", label:"step time", c:"var(--s1)", derived:1}]};
}

// Each panel is one y-scale. `why` is the reason the metric is on screen at all -- a chart
// nobody can act on is a chart nobody should read. `smooth` is a half-window in steps: the
// per-step value is drawn faint underneath and the rolling mean on top, because for the
// noisy metrics (grad norm, throughput, mix) the trend is the readable part and the spikes
// still need to be visible when they happen. An entry that depends on what the log holds is
// a function, evaluated after every fetch.
const CATALOG = [
  () => ({id:"loss", title:"Loss", fmt:F.n,
    why: VGROUPS.length ? "the objective, and whether the held-out splits follow it" : "the objective",
    series:[{k:"loss", label:"train", c:"var(--s1)", smooth:5},
            ...VGROUPS.map((g, i) => ({k:"valid_" + g, label:"valid " + nice(g), c:SLOT(i + 1), dots:1})),
            ...(has("valid_loss") ? [{k:"valid_loss", label:"valid (all)", c:SLOT(VGROUPS.length + 1), dots:1}] : [])]}),
  {id:"unr", title:"Update / weight norm", why:"‖dW‖/‖W‖ — moves before the loss does; a spike is divergence", log:1, fmt:F.n,
   series:[{k:"update_norm_ratio", label:"update ratio", c:"var(--s1)", smooth:3}]},
  {id:"grad", title:"Gradient norm", why:"pre-clip; read together with the clip fraction", log:1, fmt:F.n,
   series:[{k:"grad_norm", label:"grad norm", c:"var(--s1)", smooth:5}]},
  {id:"clip", title:"Clip fraction", why:"if this sits at 1.0 the clip is the optimiser and the LR is wrong", fmt:F.pct, y:[0,1],
   series:[{k:"clip_frac", label:"clipped", c:"var(--s2)", smooth:5}]},
  {id:"router", title:"Router health", why:"entropy → 0 or CV climbing means the router is collapsing", fmt:F.n, y:[0,null],
   series:[{k:"router_entropy", label:"entropy", c:"var(--s1)"},
           {k:"expert_load_cv", label:"load CV", c:"var(--s2)", smooth:3},
           {k:"tokens_dropped_frac", label:"dropped", c:"var(--s3)"}]},
  {id:"aux", title:"Aux (balance) loss", why:"the balancing pressure the router is under", fmt:F.n,
   series:[{k:"aux_loss", label:"aux loss", c:"var(--s1)", smooth:3}]},
  // Two ways to write down a schedule: the rates themselves, or the multiplier on rates the
  // log never states. They are different units, so they are never the same panel.
  () => has("lr_muon") || has("lr_adamw")
    ? {id:"lr", title:"Learning rate", why:"WSD schedule, log scale — same unit, so still one axis", log:1, fmt:F.n,
       series:[{k:"lr_muon", label:"Muon", c:"var(--s1)"}, {k:"lr_adamw", label:"AdamW", c:"var(--s2)"}]}
    : {id:"lr", title:"Learning-rate schedule", help:"lrs", fmt:F.n, y:[0,null],
       why:"the multiplier on both optimiser rates — warm-up, then cosine to zero",
       series:[{k:"lr_scale", label:"schedule ×", c:"var(--s1)"}]},
  {id:"noise", title:"Noise σ", why:"the injected-noise schedule, for lining up against loss kinks", fmt:F.n,
   series:[{k:"noise_std", label:"noise σ", c:"var(--s7)"}]},
  {id:"tps", title:"Throughput", why:"tokens/s — the number the run's length is made of", fmt:F.big,
   series:[{k:"tok_per_s", label:"tokens/s", c:"var(--s1)", smooth:5}]},
  {id:"sup", title:"Supervised fraction", why:"share of the packed window that carries a target — the rest is context the loss ignores", fmt:F.pct, y:[0,1], smooth:5,
   series:[{k:"sup_frac", label:"supervised", c:"var(--s3)"}]},
  timeSpec,
  () => MIXGROUPS.length ? {id:"mix", title:"Realized data mix", why:"the ratios that happened, not the ones configured", fmt:F.pct, stack:1, smooth:8, y:[0,1],
    series: MIXGROUPS.map((g, i) => ({k:"mix_" + g, label:nice(g), c:SLOT(i)}))} : null,
  {id:"mem", title:"GPU memory", why:"reserved is the number that OOMs, not allocated", fmt:F.n, y:[0,null],
   series:[{k:"mem_reserved_gb", label:"reserved GB", c:"var(--s1)"}, {k:"mem_allocated_gb", label:"allocated GB", c:"var(--s2)"}]},
];

/** The catalog minus everything this log has no numbers for, series by series. */
function activeCharts() {
  const out = [];
  for (const item of CATALOG) {
    const c = typeof item === "function" ? item() : item;
    if (!c) continue;
    const series = c.series.filter(S => S.derived || has(S.k));
    if (series.length) out.push({...c, series});
  }
  return out;
}

// Plain-language explanations, shown by hovering a title. Written for someone who has never
// trained a model: what the number is, what good looks like, and what would be bad news.
const HELP = {
  loss: `<p><b>How wrong the model is when it guesses the next piece of text.</b> It reads text with the
    next word hidden, guesses, and is scored on how surprised it was by the real answer. Lower is better.</p>
    <p>The <b>train</b> line is measured on the text it is studying right now. The <b>valid</b> lines are
    text it has never seen — one for each held-out split, which is a source of text while the model is
    reading the internet and a kind of task once it is being taught to answer — so they are the honest
    test of whether it is learning rather than memorising.</p>
    <p>Good: everything drifting down, with the valid lines following the train line. Bad: the valid lines
    flatten out or turn upward while train keeps falling — that is memorising the training text instead of
    learning the language in it.</p>`,
  unr: `<p><b>How much the model changed this step, relative to its own size.</b> 0.01 means the internals
    moved by about 1%.</p>
    <p>It is the earliest warning sign there is, because it reacts before the loss does. A sudden jump to
    many times its usual value means training is about to fall apart; a collapse toward zero means learning
    has effectively stopped.</p>`,
  grad: `<p><b>How big a correction this batch of text is asking for.</b> After each guess, the model works
    out which direction to nudge every internal number; this is the overall size of that nudge.</p>
    <p>It normally starts large and settles as the model gets less badly wrong. Occasional spikes are an
    unusual batch of text. A sustained climb is trouble.</p>`,
  clip: `<p><b>How often the safety limit had to step in.</b> Corrections larger than a set size get scaled
    back so that one strange batch cannot wreck the model — this is the fraction of the step that was
    scaled back.</p>
    <p>100% early on is normal and expected. Bad: it never comes down. Then the limit, rather than the
    learning rate, is deciding how fast the model moves, and the learning rate is set too high.</p>`,
  router: `<p><b>Whether the model's specialists are being used evenly.</b> This model is a mixture of
    experts: for each piece of text a small router picks a couple of specialist sub-networks to do the
    work, so it can be large without being slow.</p>
    <p><b>Entropy</b> is how spread out those choices are — 1.0 is using everything, and a slide toward 0
    means the router has collapsed onto a favourite few. <b>Load CV</b> is how uneven the workload is —
    0 is a perfect split, and above about 0.5 a handful of experts are doing all the work while the rest
    are dead weight you paid for. <b>Dropped</b> is text that found no room; it should stay at 0.</p>`,
  aux: `<p><b>The gentle pressure keeping the specialists evenly used.</b> Left alone, the router would
    send everything to whichever experts got good first. A small extra penalty pushes back on that.</p>
    <p>It should sit low and roughly flat. A steady climb means the router keeps trying to concentrate and
    is being fought the whole way — check the router health panel next to it.</p>`,
  lr: `<p><b>How big a step the model takes each time it learns something.</b> Too small and it takes
    forever; too large and it overshoots and destabilises.</p>
    <p>The shape is deliberate: a gentle <b>warm-up</b> from near zero, a long <b>stable</b> stretch at
    full speed, then a <b>decay</b> to near zero at the end so the model settles rather than bounces. The
    two lines are the two optimisers, one for each kind of parameter; the scale is logarithmic so both
    fit in one frame.</p>`,
  noise: `<p><b>Deliberate noise mixed into training.</b> A little randomness stops the model settling into
    the first easy answer it finds.</p>
    <p>It is scheduled to fade as training goes on: rough exploration early, careful refinement later. This
    line is that schedule, and it is here so a bend in the loss can be checked against it.</p>`,
  tps: `<p><b>Speed: pieces of text processed per second.</b> Multiply by the hours you plan to run to see
    how much the model will read in total.</p>
    <p>It should be a flat line. Dips mean the GPU was starved or sharing the machine; a step change up or
    down usually means a setting changed between restarts.</p>`,
  time: `<p><b>Where each second of a step went.</b> <b>Forward</b> is reading the text and making
    predictions; <b>backward</b> is working out how to correct them, and normally costs about twice as
    much; <b>optimiser</b> is applying those corrections to the model.</p>
    <p><b>Waiting on the loader</b> is the graphics card sitting idle while the next batch of text is
    prepared — pure waste, since the card is powered on and doing nothing. <b>Validation</b> and
    <b>checkpoint</b> are the periodic scoring and saving, which is why some steps take visibly longer
    than their neighbours.</p>
    <p><b>Unattributed</b> is the rest of the wall clock: bookkeeping the trainer does not measure. If
    that band grows, time is going somewhere nobody is looking.</p>`,
  mix: `<p><b>Where the text actually came from.</b> The run is configured to blend sources in set
    proportions; this is the blend that really arrived, measured rather than assumed.</p>
    <p>It should sit close to the intended recipe and change only where the recipe was meant to change.
    Drift means one source is running dry or being over-drawn, which quietly changes what the model
    becomes good at.</p>`,
  sup: `<p><b>How much of the text the model is actually being graded on.</b> Fine-tuning packs whole
    conversations into each window, but only the assistant's replies are scored — the question, the system
    prompt and any tool output are there to be read, not to be imitated, so they are masked out.</p>
    <p>This is the share of the window that survived that mask. Higher means more of each expensive step
    is teaching something. It moves with the shape of the data, not with the model, so a sudden drop means
    the packing changed — long tool schemas crowding out the replies, for instance — rather than anything
    going wrong with training.</p>`,
  lrs: `<p><b>How big a step the model takes each time it learns something, as a dial from 0 to 1.</b>
    The run sets two actual rates, one per optimiser; this line is the multiplier applied to both, which
    is the part that changes over time.</p>
    <p>The shape is deliberate: a short <b>warm-up</b> from zero, so the first few batches cannot wreck a
    model that was fine when it arrived, then a long <b>cosine</b> fall back to zero so the model settles
    into its answer rather than bouncing around it.</p>`,
  time_flat: `<p><b>How many seconds one step took, start to finish.</b> Every step reads a batch of text,
    makes its predictions, works out the corrections and applies them; this is the whole of that.</p>
    <p>This trainer reports one number rather than a breakdown, so there is no split into reading,
    correcting and saving here. Steps that stick up are the ones that also ran a validation or wrote a
    checkpoint.</p>
    <p>It should be a flat line. A steady climb means something is slowing down; a sawtooth usually means
    the card is being shared.</p>`,
  mem: `<p><b>How much of the graphics card's memory is in use.</b> <b>Allocated</b> is what the model is
    actively holding; <b>reserved</b> is what it has claimed from the card and may reuse.</p>
    <p>Reserved is the one that matters: when it reaches the card's capacity the run crashes with an
    out-of-memory error. Flat is healthy; a steady climb means something is accumulating and the run will
    eventually hit the ceiling.</p>`,
};

const TILE_HELP = {
  "step": `<p><b>Which training step the run is on.</b> One step is one batch of text read, one correction
    applied. Everything on this page is plotted against this number.</p>
    <p>The phase underneath is where the run is in its learning-rate schedule: warm-up, stable, or the
    final decay.</p>`,
  "tokens seen": `<p><b>How much text the model has read so far</b>, counted in tokens — a token is roughly
    three quarters of a word.</p><p>The percentage is how far that goes through one full pass of the
    available text.</p>`,
  "loss (EMA)": `<p><b>How wrong the model's guesses are, smoothed.</b> Individual steps bounce around
    depending on which text they drew, so this is a running average; the raw figure underneath is the
    latest single step. Lower is better.</p>`,
  "valid perplexity": `<p><b>Roughly how many equally-likely options the model is choosing between
    at each guess</b>, on text it has never been trained on. The vocabulary is 32,768 tokens, so knowing
    nothing at all would score 32,768; lower means it has narrowed the field.</p>
    <p>Measured on the held-out split, not on the text being learned from — that is the point of it, and
    it is the figure worth comparing between runs. The breakdown underneath is each source separately:
    they differ enormously because prose is far less predictable than code.</p>
    <p>The headline number weights those sources by how much of each the run actually trains on.</p>
    <p>Validation happens at two sizes. A quick peek every few steps reads one batch per source — enough
    to spot a disaster, too little to trust to a decimal place. A much larger evaluation runs
    periodically over twenty times as much text, and that is the one to quote.</p>
    <p>This tile shows the most recent measurement of either kind, and says underneath which one it was.
    On the loss chart the large evaluations are the big ringed dots and the peeks are the small ones, so
    a surprising move can be checked against the next solid reading before anyone acts on it.</p>`,
  "tokens/s": `<p><b>How fast the run is going</b>, in pieces of text per second, with the time for one
    step underneath and how much of that was spent waiting for data rather than computing.</p>`,
  "reserved": `<p><b>Graphics card memory claimed by this run.</b> If it reaches the card's capacity the
    run crashes, so this is the number to watch, not the smaller allocated figure beside it.</p>`,
  "time left": `<p><b>How long until the run finishes</b>, at the speed it is going right now.</p>
    <p>It is a projection, not a promise: it moves whenever the step time does, and it does not know about
    anything that stops the machine.</p>`,
  "supervised": `<p><b>How much of each batch the model is actually graded on.</b> Only the assistant's
    replies count towards the score; the questions and any tool output are read for context but masked out
    of the loss, so this is the share of the window doing the teaching.</p>`,
  "grad norm": `<p><b>How big a correction this batch of text is asking for.</b> After each guess the
    model works out which direction to nudge every internal number; this is the overall size of that
    nudge, before any safety limit is applied.</p>
    <p>It should wander around a roughly steady level. A sustained climb, or a jump to many times the
    usual value, means a batch the model finds shocking — during fine-tuning that is often a formatting
    mistake in the training data rather than anything the model did wrong.</p>`,
  "expert load CV": `<p><b>How evenly work is shared between the model's specialists.</b> 0 would be a
    perfectly even split.</p>
    <p>Green is healthy. Amber past about 0.35 and red past 0.5, where a few specialists are doing most of
    the work and the rest are capacity you are paying for and not using.</p>`,
  "router entropy": `<p><b>How freely the model is choosing between its specialists.</b> 1.0 means it is
    using the full set; a fall toward 0 means it has collapsed onto a favourite few and the extra
    capacity is wasted.</p><p>Green is healthy, amber below 0.8, red below 0.5.</p>`,
};

let raw = [], rows = [], dead = [], resumes = [], logLoss = false, ghosts = true, lastTs = 0;
let domSig = "";                  // the panel set the DOM was last built for
let mtime = __MTIME__;
// Validation runs at two sizes: a cheap peek every few steps and a periodic one over many
// more batches. They measure the same thing at very different confidence, so the page keeps
// them apart rather than averaging a solid number together with a noisy one.
let maxVB = 0;
const isFull = r => maxVB > 1 && (+r.valid_batches || 0) === maxVB;

const TZ = "America/Los_Angeles";
const today = () => new Date().toLocaleDateString("en-US", {timeZone: TZ});

/**
 * Unix seconds for a row, whichever way the trainer wrote the stamp: the pretrainer logs a
 * float, SFT logs ISO 8601. Parsed once at ingest so nothing downstream has to care -- an
 * unparsed string multiplied by 1000 is NaN, and every hover silently loses its clock.
 */
function normTs(r) {
  if (typeof r.ts === "string") {
    const t = Date.parse(r.ts);
    r.ts = isFinite(t) ? t / 1000 : null;
  }
}

/** Pacific clock time for a step. Bare time when it happened today, dated when it did not. */
function when(r) {
  if (r.ts == null) return "";
  const d = new Date(r.ts * 1000);
  const day = d.toLocaleDateString("en-US", {timeZone: TZ});
  const clock = d.toLocaleTimeString("en-US", {timeZone: TZ, hour: "numeric", minute: "2-digit", timeZoneName: "short"});
  return (r._est ? "~" : "") + (day === today() ? clock
        : d.toLocaleDateString("en-US", {timeZone: TZ, month: "short", day: "numeric"}) + ", " + clock);
}

/**
 * Rows written before the logger recorded `ts` still deserve a clock. The last line of the
 * file was written at the file's mtime, so walk backwards from there subtracting each step's
 * own duration. It is exact at the tail and drifts as it goes back -- an estimate, marked as
 * one with a leading "~", because time the process spent stopped between attempts is time
 * this cannot see.
 */
function backfillTs() {
  let t = mtime;
  if (t == null) return;
  for (let i = raw.length - 1; i >= 0; i--) {
    const r = raw[i];
    if (r.ts != null && !r._est) { t = r.ts; continue; }
    r.ts = t; r._est = true;
    t -= stepSec(r);
  }
}

/**
 * The log is append-only across restarts, so one file holds several attempts writing the
 * same step numbers. Plotted raw, every line doubles back over itself -- which is what made
 * the first version of this page unreadable.
 *
 * Later writes win. `--resume-step 250` continues from the weights checkpointed at 250, so
 * gluing the earlier attempt's steps 0-249 to the resumed 250-onwards is the real history
 * of the weights that exist now; the attempt's own 250-onwards is a branch that was thrown
 * away. Those superseded stretches are kept as `dead` and drawn faint, because "we rolled
 * back to 250 and tried again" is exactly the kind of thing you want to see on the chart.
 */
function rebuild() {
  const winner = new Map(), starts = [];
  let seg = 0, prev = -Infinity;
  for (const r of raw) {
    normTs(r);
    r.compute_s = stepSec(r) - dataSec(r);
    bands(r);
    if (r.step <= prev) { seg++; starts.push(r.step); }
    prev = r.step;
    r._seg = seg;
    winner.set(r.step, r);
  }
  rows = [...winner.values()].sort((a, b) => a.step - b.step);
  // A marker means "the process went backwards and picked up here", which is only the step
  // a restart resumed at. Seams where a superseded stretch merely runs out are not restarts
  // and drawing them as such put rules on the chart that nothing happened at.
  resumes = [...new Set(starts)].filter(s => s > 0 && winner.has(s));

  dead = [];                              // contiguous stretches that a later attempt replaced
  let run = [];
  for (const r of raw) {
    if (winner.get(r.step) === r) { if (run.length > 1) dead.push({rows: run}); run = []; }
    else run.push(r);
  }
  if (run.length > 1) dead.push({rows: run});
  maxVB = Math.max(0, ...rows.map(r => +r.valid_batches || 0));
  discover();
  SPECS = activeCharts();
  // SFT logs no running token count. tok_per_s is the step's own tokens divided by its own
  // seconds, so multiplying them back gives that step's tokens exactly and the running sum
  // is the number the pretrainer writes down -- an estimate only in that a step missing
  // either field contributes nothing.
  let acc = 0;
  for (const r of rows) { acc += (+r.tok_per_s || 0) * stepSec(r); r.tokens_est = acc; }
  backfillTs();
}

const ghostRuns = () => (ghosts ? dead : []);

function seriesFor(spec, rws) {
  return spec.series.map(S => {
    const rawv = rws.map(r => { const v = +r[S.k]; return isFinite(v) ? v : null; });
    return {...S, rawv, val: smoothed(rawv, S.smooth ?? spec.smooth)};
  });
}

/**
 * Split one step's wall clock into bands that add up to it.
 *
 * Forward and backward are timed on the GPU while the loader and optimiser are timed on the
 * CPU, so they are not the same kind of second and cannot simply be added. The GPU pair is
 * used as a ratio to divide the wall-clock compute mark between them, which keeps the stack
 * honest about the total while still answering "how much of it is backward".
 *
 * Rows written before the trainer measured any of this keep their compute in `unattributed`
 * rather than being dressed up as forward -- the old log genuinely does not know.
 */
function bands(r) {
  r.b_step = stepSec(r);
  r.b_optim = +r.t_optim_s || 0;
  r.b_data = dataSec(r);
  r.b_eval = +r.t_eval_s || 0;
  r.b_ckpt = +r.t_ckpt_s || 0;
  // What is left of the wall clock once the separately-measured phases are taken out. The
  // pretrainer marks this directly; where it does not, subtracting is the only figure that
  // keeps the stack equal to the step. Trusting the GPU spans as seconds instead overshot
  // the step time -- forward and backward overlap the optimiser on the timeline.
  const rest = Math.max(0, r.b_step - (r.b_optim + r.b_data + r.b_eval + r.b_ckpt));
  const compute = isFinite(+r.t_compute_s) ? +r.t_compute_s : rest;
  const f = +r.t_fwd_s, b = +r.t_bwd_s;
  const split = isFinite(f) && isFinite(b) && f + b > 0;
  r.b_fwd = split ? compute * f / (f + b) : 0;
  r.b_bwd = split ? compute * b / (f + b) : 0;
  r.b_other = Math.max(0, r.b_step
    - (r.b_fwd + r.b_bwd + r.b_optim + r.b_data + r.b_eval + r.b_ckpt));
}

function smoothed(vals, half) {
  if (!half) return vals;
  const out = vals.slice();
  for (let i = 0; i < vals.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(vals.length - 1, i + half); j++)
      if (vals[j] != null) { s += vals[j]; n++; }
    out[i] = n ? s / n : null;
  }
  return out;
}

function tick(lo, hi, n) {           // ~n round steps spanning [lo, hi]
  const raw = (hi - lo) / Math.max(n, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 2.5, 5, 10].find(m => m * mag >= raw) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of (Math.log10(hi / lo) > 2.5 ? [1] : [1, 3])) {
      const v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  return out.length > 1 ? out : [lo, hi];
}

function draw(spec, el) {
  const W = Math.max(el.clientWidth || 560, 320), H = 230;
  const L = 52, R = 60, T = 12, B = 28, iw = W - L - R, ih = H - T - B;
  const useLog = spec.id === "loss" ? logLoss : (spec.log && !spec.stack);
  // Earlier attempts sit underneath the current run as ghosts -- comparable, but never
  // mistakable for it. Stacked panels skip them: two stacks in one frame read as neither.
  const past = spec.stack ? [] : ghostRuns().map(g => ({rows: g.rows, S: seriesFor(spec, g.rows)}));
  const S_ = seriesFor(spec, rows);       // raw values, then the drawn (possibly smoothed) ones

  const allRows = past.flatMap(p => p.rows).concat(rows);
  const x0 = Math.min(...allRows.map(r => r.step)), x1 = Math.max(...allRows.map(r => r.step));
  const X = v => L + (x1 === x0 ? iw / 2 : (v - x0) / (x1 - x0) * iw);

  // The scale is fitted to everything drawn, so no line can leave the panel.
  let vals = [];
  if (spec.stack) vals = rows.map((_, i) => S_.reduce((t, S) => t + (S.val[i] || 0), 0));
  else {
    for (const S of S_) vals = vals.concat(S.val, S.smooth || spec.smooth ? S.rawv : []);
    for (const p of past) for (const S of p.S) vals = vals.concat(S.val);
  }
  vals = vals.filter(v => v != null && isFinite(v) && (!useLog || v > 0));
  if (!vals.length) { el.innerHTML = '<div class="why" style="padding:32px 0">no data yet</div>'; return; }
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (spec.y) { if (spec.y[0] != null) lo = spec.y[0]; if (spec.y[1] != null) hi = spec.y[1]; }
  if (useLog) lo = Math.max(lo, 1e-12);
  else if (hi === lo) { hi = lo + 1; }
  else { const p = (hi - lo) * 0.08; hi += p; if (!spec.y || spec.y[0] == null) lo -= p; }
  const ly = v => Math.log10(Math.max(v, 1e-12));
  const Y = v => useLog ? T + ih - (ly(v) - ly(lo)) / (ly(hi) - ly(lo)) * ih
                        : T + ih - (v - lo) / (hi - lo) * ih;

  let s = "";
  for (const v of (useLog ? logTicks(lo, hi) : tick(lo, hi, 4)))
    s += `<line x1="${L}" x2="${W-R}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="var(--grid)" stroke-width="1"/>`
      + `<text x="${L-8}" y="${(Y(v)+3.5).toFixed(1)}" text-anchor="end" font-size="10.5" fill="var(--text-muted)">${spec.fmt(v)}</text>`;
  for (const v of tick(x0, x1, 5)) if (v >= x0 && v <= x1)
    s += `<text x="${X(v).toFixed(1)}" y="${H-9}" text-anchor="middle" font-size="10.5" fill="var(--text-muted)">${F.big(v)}</text>`;
  // Where the run was restarted. Without these, a discontinuity looks like a training event.
  for (const v of resumes)
    s += `<line x1="${X(v).toFixed(1)}" x2="${X(v).toFixed(1)}" y1="${T}" y2="${T+ih}" stroke="var(--text-muted)"`
      + ` stroke-width="1" stroke-dasharray="3 3" opacity=".45"/>`;

  const pts = (val, keepGaps, rws = rows) => {
    let seg = [], paths = [];
    for (let i = 0; i < rws.length; i++) {
      const v = val[i];
      if (v != null && (!useLog || v > 0)) seg.push(`${X(rws[i].step).toFixed(1)},${Y(v).toFixed(1)}`);
      // A validation series is logged every N steps; connecting across those gaps is the
      // curve people mean. A gap in a per-step series is a real break.
      else if (seg.length && keepGaps) { paths.push(seg); seg = []; }
    }
    if (seg.length) paths.push(seg);
    return paths;
  };

  const ends = [];                          // last value of each series, for direct labels
  const stackNow = spec.stack ? S_.reduce((t, S) => t + (S.val[S.val.length - 1] || 0), 0) : 0;
  if (spec.stack) {
    // 2px surface gap between bands, so adjacent fills never bleed into one another.
    const base = rows.map(() => 0);
    for (const S of S_) {
      const top = rows.map((_, i) => base[i] + (S.val[i] || 0));
      const up = rows.map((r, i) => `${X(r.step).toFixed(1)},${Y(top[i]).toFixed(1)}`);
      const dn = rows.map((r, i) => `${X(r.step).toFixed(1)},${Y(base[i]).toFixed(1)}`).reverse();
      s += `<polygon points="${up.concat(dn).join(" ")}" fill="${S.c}" fill-opacity=".8"/>`
        + `<polyline points="${up.join(" ")}" fill="none" stroke="var(--surface-1)" stroke-width="2"/>`;
      // Only bands worth a line of the reader's attention get a label -- 2% of the stack.
      // Otherwise a seven-band chart spends five labels on 6e-7 and crowds out the two
      // numbers anyone came for.
      const last = S.val[S.val.length - 1];
      if (last > 0.02 * stackNow) ends.push({y: Y((top[top.length-1] + base[base.length-1]) / 2), c: S.c, t: spec.fmt(last)});
      for (let i = 0; i < rows.length; i++) base[i] = top[i];
    }
  } else {
    for (const p of past)
      for (const S of p.S)
        for (const path of pts(S.val, !S.dots, p.rows))
          if (path.length > 1)
            s += `<polyline points="${path.join(" ")}" fill="none" stroke="${S.c}" stroke-width="1.5" opacity=".16"/>`;
    for (const S of S_) {
      if (S.smooth ?? spec.smooth)          // the per-step value, underneath and quiet
        for (const p of pts(S.rawv, !S.dots))
          if (p.length > 1) s += `<polyline points="${p.join(" ")}" fill="none" stroke="${S.c}" stroke-width="1" opacity=".28"/>`;
      for (const p of pts(S.val, !S.dots)) {
        if (p.length === 1) s += `<circle cx="${p[0].split(",")[0]}" cy="${p[0].split(",")[1]}" r="3" fill="${S.c}"/>`;
        else s += `<polyline points="${p.join(" ")}" fill="none" stroke="${S.c}" stroke-width="2"`
               + ` stroke-linejoin="round" stroke-linecap="round"/>`;
      }
      // Markers on the sparse series say "measured here", so they stay small and unringed:
      // a surface-coloured ring chops the line into something that reads as a dashed style.
      // The exception is the big periodic evaluation, which is measured over many times more
      // text and earns a mark you can pick out and trust.
      if (S.dots) for (let i = 0; i < rows.length; i++) if (S.val[i] != null) {
        const full = isFull(rows[i]);
        s += `<circle cx="${X(rows[i].step).toFixed(1)}" cy="${Y(S.val[i]).toFixed(1)}"`
          + ` r="${full ? 4 : 1.9}" fill="${S.c}"`
          + (full ? ` stroke="var(--surface-1)" stroke-width="2"` : "") + `/>`;
      }
      for (let i = rows.length - 1; i >= 0; i--)
        if (S.val[i] != null) { ends.push({y: Y(S.val[i]), c: S.c, t: spec.fmt(S.val[i])}); break; }
    }
  }
  // Direct labels at the right edge: the current value of every series, without a number
  // on every point. Pushed apart so two close series stay two readable labels.
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i-1].y + 12);
  for (const e of ends)
    s += `<rect x="${W-R+4}" y="${(e.y-1.5).toFixed(1)}" width="9" height="3" rx="1.5" fill="${e.c}"/>`
      + `<text x="${W-R+17}" y="${(e.y+3.5).toFixed(1)}" font-size="10.5" fill="var(--text-secondary)">${e.t}</text>`;
  s += `<line id="cx" x1="0" x2="0" y1="${T}" y2="${T+ih}" stroke="var(--text-muted)" stroke-width="1" opacity="0"/>`;
  s += `<rect x="${L}" y="${T}" width="${iw}" height="${ih}" fill="transparent"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${spec.title}">${s}</svg>`;
  el._hit = {L, iw, x0, x1, W, H, X};
}

function render() {
  const root = document.getElementById("charts");
  // Panels come and go with the log: a resumed run that starts reporting a metric should
  // grow a panel for it without a reload, so the frame is rebuilt whenever the set changes.
  const sig = SPECS.map(c => c.id + ":" + c.series.map(S => S.k).join(",")).join("|");
  if (sig !== domSig) {
    domSig = sig;
    root.innerHTML = SPECS.map(c => `<div class="card" id="card-${c.id}">
      <h2><span class="t" tabindex="0" aria-describedby="help-${c.id}">${c.title}</span></h2>
      <div class="pop" id="help-${c.id}" role="tooltip">${HELP[c.help || c.id] || ""}</div>
      <div class="why">${c.why}</div>
      <div class="legend">${c.series.length > 1 ? c.series.map(S =>
        `<span><i style="background:${S.c}"></i>${S.label}</span>`).join("") : ""}</div>
      <div class="plot"><div class="body"></div><div class="tip"></div></div></div>`).join("");
    for (const c of SPECS) {
      const card = document.getElementById("card-" + c.id), plot = card.querySelector(".plot");
      const body = plot.querySelector(".body"), tip = plot.querySelector(".tip");
      plot.addEventListener("pointermove", e => {
        const h = body._hit; if (!h || !rows.length) return;
        const rect = plot.getBoundingClientRect(), sx = rect.width / h.W;
        const px = (e.clientX - rect.left) / sx;
        const frac = Math.min(1, Math.max(0, (px - h.L) / h.iw));
        const target = h.x0 + frac * (h.x1 - h.x0);
        let bi = 0, bd = Infinity;
        rows.forEach((r, i) => { const d = Math.abs(r.step - target); if (d < bd) { bd = d; bi = i; } });
        const r = rows[bi];
        const line = body.querySelector("#cx");
        if (line) { const x = h.X(r.step); line.setAttribute("x1", x); line.setAttribute("x2", x); line.setAttribute("opacity", ".5"); }
        tip.innerHTML = `<b>step ${r.step.toLocaleString()}</b><span class="ts">${when(r)}</span>` + c.series.map(S => {
          const v = +r[S.k]; if (!isFinite(v)) return "";
          return `<div><span><i style="background:${S.c}"></i>${S.label}</span><span>${c.fmt(v)}</span></div>`;
        }).join("");
        const tx = h.X(r.step) * sx;
        tip.style.left = Math.min(rect.width - tip.offsetWidth - 4, Math.max(0, tx + 12)) + "px";
        tip.style.top = "6px"; tip.style.opacity = 1;
      });
      plot.addEventListener("pointerleave", () => {
        tip.style.opacity = 0;
        const line = body.querySelector("#cx"); if (line) line.setAttribute("opacity", 0);
      });
    }
  }
  for (const c of SPECS) draw(c, document.getElementById("card-" + c.id).querySelector(".body"));

  /*
   * Tiles read the newest value of each field, not the newest row. A run ends on a big
   * validation, and that line carries only the valid_* losses -- reading the last row alone
   * printed "NaN tokens" and a row of dashes over a run that had just finished perfectly
   * well. Sparse fields are normal in this log; a tile that says "where the run is now"
   * should answer with the last time anyone measured it.
   */
  const latest = k => {
    for (let i = rows.length - 1; i >= 0; i--) {
      const v = rows[i][k];
      if (v != null && v !== "") return v;
    }
    return null;
  };
  // null, not 0, for a field this log does not have: +null is 0, and a missing epoch_frac
  // silently printed "0.00% of an epoch" over a run that tracks no epochs at all.
  const num = k => { const v = latest(k); return v != null && isFinite(+v) ? +v : null; };
  const step = rows.length ? rows[rows.length - 1].step : null;
  const cv = num("expert_load_cv"), ent = num("router_entropy");
  const secs = num("step_time_s") ?? num("step_s");

  /*
   * Held-out perplexity. The per-step `perplexity` field in the log is exp() of the training
   * batch's own loss -- the text the model is being fitted to right now -- so it is not the
   * number "perplexity" usually means and it flattered the run. This uses the valid_* losses
   * instead, weighted by the realized data mix wherever the run reports one: the splits are
   * measured on equal token counts but pretraining is ~80% python, and an unweighted mean
   * would let the small, hardest slice dominate a headline figure. A run that logs no mix
   * (SFT) gets a flat mean over its domains, and the tile says so rather than implying a
   * weighting nobody supplied.
   */
  const vrow = [...rows].reverse().find(x => VGROUPS.some(g => x["valid_" + g] != null)) || {};
  const weighted = MIXGROUPS.length > 0;
  const vparts = [];
  let wsum = 0, lsum = 0, fsum = 0, n = 0, ppl = null;
  for (const g of VGROUPS) {
    const l = +vrow["valid_" + g];
    if (!isFinite(l)) continue;
    // A run's closing validation is written as a line of its own, with no mix on it, so the
    // weights come from the last step that reported one: the recipe does not change between
    // two adjacent lines, and falling back to a flat mean there would move the headline
    // number for a reason that has nothing to do with the model.
    const w = weighted ? (+vrow["mix_" + g] || +latest("mix_" + g) || 0) : 1;
    wsum += w; lsum += w * l; fsum += l; n++;
    vparts.push(nice(g) + " " + F.n(Math.exp(l)));
  }
  const flat = !weighted || wsum <= 0;
  if (n) ppl = Math.exp(flat ? fsum / n : lsum / wsum);
  // The most recent measurement, whichever size it was. A tile on a live dashboard is read
  // as "where the run is now", so a value hundreds of steps old belongs here only if nothing
  // newer exists. Which kind it came from is spelled out underneath, and the loss chart
  // marks the big evaluations so the trustworthy points stay identifiable there.
  const vnote = vparts.length
    ? vparts.join(" · ")
      + (maxVB > 1 ? " · " + (isFull(vrow) ? maxVB + "-batch eval" : "1-batch peek") : "")
      + ", step " + vrow.step + (flat ? " · flat mean" : "")
    : "no validation yet";

  const tiles = [["step", step == null ? "—" : step.toLocaleString(),
                  latest("phase") ? latest("phase") + " phase" : ""]];
  const seen = has("tokens_seen") ? num("tokens_seen") : num("tokens_est");
  if (seen != null)
    tiles.push(["tokens seen", F.big(seen),
      num("epoch_frac") != null ? (100 * num("epoch_frac")).toFixed(2) + "% of an epoch"
        : has("tokens_seen") ? "" : "estimated from throughput"]);
  tiles.push(["loss (EMA)", F.n(num("loss_ema")), "raw " + F.n(num("loss"))]);
  if (num("eta_h") != null)
    tiles.push(["time left", F.n(num("eta_h")) + " h", "at the current rate"]);
  if (num("sup_frac") != null)
    tiles.push(["supervised", F.pct(num("sup_frac")),
      F.big(num("sup_tokens")) + " tokens scored per step"]);
  if (VGROUPS.length) tiles.push(["valid perplexity", F.n(ppl), vnote]);
  tiles.push(["tokens/s", F.big(num("tok_per_s")), F.n(secs) + " s/step"
    + (DATA_K() ? ", " + F.n(num(DATA_K())) + " s of it on the loader" : "")]);
  if (has("mem_reserved_gb"))
    tiles.push(["reserved", F.n(num("mem_reserved_gb")) + " GB",
      has("mem_allocated_gb") ? F.n(num("mem_allocated_gb")) + " GB allocated" : "peak so far"]);
  if (has("expert_load_cv"))
    tiles.push(["expert load CV", F.n(cv),
      num("experts_unused") ? F.n(num("experts_unused")) + " experts unused" : "all experts used",
      cv > 0.5 ? "crit" : cv > 0.35 ? "warn" : "good"]);
  if (has("router_entropy"))
    tiles.push(["router entropy", F.n(ent), F.pct(num("tokens_dropped_frac")) + " tokens dropped",
      ent < 0.5 ? "crit" : ent < 0.8 ? "warn" : "good"]);
  // Without a router to watch, the gradient norm is the run's health check, so it comes out
  // of the table and onto the front row.
  if (has("grad_norm") && !has("router_entropy"))
    tiles.push(["grad norm", F.n(num("grad_norm")),
      has("clip_frac") ? F.pct(num("clip_frac")) + " clipped" : "pre-clip"]);

  const parts = ["hover any title for a plain-English explanation",
                 "bold lines are rolling means over the faint per-step values"];
  if (maxVB > 1) parts.push(`large dots on the valid lines are the ${maxVB}-batch evaluations, the small ones single-batch peeks`);
  else if (VGROUPS.length) parts.push("dots on the valid lines are the steps validation actually ran on");
  if (dead.length) parts.push(`${dead.length} superseded stretch${dead.length > 1 ? "es" : ""} drawn faint — attempts a restart rolled back over`);
  if (resumes.length) parts.push("dashed rules mark a restart");
  document.getElementById("note").textContent = parts.join(" · ");
  document.getElementById("ghosts").style.display = dead.length ? "" : "none";

  document.getElementById("tiles").innerHTML = tiles.map(([k, v, n, cls]) =>
    `<div class="tile"><div class="k"><span class="t" tabindex="0">${k}</span></div>`
    + `<div class="v ${cls||""}">${v}</div><div class="n">${n||"&nbsp;"}</div>`
    + `<div class="pop" role="tooltip">${TILE_HELP[k] || ""}</div></div>`).join("");

  const cols = ["step","loss","loss_ema","perplexity","aux_loss","grad_norm","clip_frac","update_norm_ratio",
                "expert_load_cv","router_entropy","lr_scale","sup_frac","tok_per_s","step_time_s","step_s",
                "mem_reserved_gb","eta_h"]
                .filter(c => c === "step" || has(c));
  const last = rows.slice(-40).reverse();
  document.getElementById("table").innerHTML =
    `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
    last.map(r => `<tr>${cols.map(c => `<td>${r[c] == null ? "—" : (c === "step" ? r[c] : F.n(+r[c]))}</td>`).join("")}</tr>`).join("") +
    "</tbody></table>";
}

const PRELOAD = __PRELOAD__;      // non-null in a snapshot: no server to poll

/**
 * The page's own ?path= is carried through to the API, so one server can show any run and a
 * tab keeps polling the file it was opened for.
 *
 * `from` counts lines in the file, so it comes from `raw.length` and never `rows.length` --
 * `rows` is deduplicated by step, and asking from the smaller number re-fetches every line a
 * later attempt superseded, appends them again, and turns each duplicate into a fake restart.
 */
function rowsQuery() {
  const q = new URLSearchParams(location.search);
  q.set("from", raw.length);
  return q.toString();
}

async function poll() {
  try {
    // `raw.length`, never `rows.length`: the server indexes lines in the file, and `rows` is
    // deduplicated by step. Asking from the smaller number re-fetches every line an earlier
    // attempt superseded, appends them again, and each duplicate reads as another restart.
    const d = PRELOAD ? {rows: raw.length ? [] : PRELOAD, reset: false}
                      : await (await fetch("api/rows?" + rowsQuery())).json();
    if (d.reset) raw = [];
    if (d.rows.length) {
      raw.push(...d.rows);
      rebuild();
      lastTs = Date.now();
      render();
    } else if (!document.getElementById("charts").children.length) render();
    const age = (Date.now() - lastTs) / 1000;
    const live = rows.length && age < 180;
    document.getElementById("pulse").className = "dot " + (live ? "live" : rows.length ? "idle" : "");
    const res = resumes.length;
    document.getElementById("status").textContent =
      `${rows.length.toLocaleString()} steps`
      + (res ? ` · restarted ${res === 1 ? "once" : res + " times"}` : "")
      + ` · ${live ? "live" : "idle"} · ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    document.getElementById("status").textContent = "lost the server — retrying";
  }
}

document.getElementById("theme").onclick = () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
  render();
};
document.getElementById("logloss").onclick = e => { logLoss = !logLoss; e.target.textContent = logLoss ? "linear loss" : "log loss"; render(); };
document.getElementById("ghosts").onclick = e => { ghosts = !ghosts; e.target.textContent = ghosts ? "hide earlier runs" : "show earlier runs"; render(); };
addEventListener("resize", () => rows.length && render());
poll(); if (!PRELOAD) setInterval(poll, __POLL__ * 1000);
</script>
"""


def read_rows(path: Path, start: int):
    """Rows from index `start` on, plus a flag for a file that was truncated or replaced."""
    if not path.exists():
        return [], False
    rows = []
    # errors="replace": a power cut left a tail of NUL bytes in this file once, and
    # decoding raises out of the `for line in fh` itself -- so /api/rows would 500 while
    # GET / still returned the page, and the dashboard sat blank and retrying at exactly
    # the moment the history mattered.
    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass          # a partially-flushed final line; it will be whole next poll
    return rows[start:], len(rows) < start


def build_page(title: str, poll: int, preload=None, mtime=None, shown="") -> str:
    return (PAGE.replace("__TITLE__", title)
                .replace("__PATH__", str(shown))
                .replace("__POLL__", str(poll))
                .replace("__PRELOAD__", json.dumps(preload) if preload else "null")
                .replace("__MTIME__", json.dumps(mtime)))


def serve(path: Path, port: int, poll: int, root: Path) -> None:
    """
    One server, any run: ?path= picks the file, so comparing two runs is two browser tabs
    rather than two processes on two ports.

    The path is a request parameter, so it is not trusted. It must land inside the directory
    the server was started in and end in .jsonl -- otherwise a page in any other tab could
    walk this endpoint up the filesystem and read whatever it liked, line by line.
    """

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _target(self, url):
            """The metrics file this request is about, or None if it asked for something else."""
            asked = (parse_qs(url.query).get("path") or [None])[0]
            if not asked:
                return path
            p = (root / asked).resolve()
            try:
                p.relative_to(root)
            except ValueError:
                return None
            return p if p.suffix == ".jsonl" and p.is_file() else None

        def do_GET(self):
            url = urlparse(self.path)
            target = self._target(url)
            if target is None:
                self._send(b'{"error": "no such metrics file under this directory"}',
                           "application/json", 404)
                return
            try:
                shown = target.resolve().relative_to(root)
            except ValueError:
                shown = target

            if url.path.rstrip("/").endswith("api/rows"):
                start = int(parse_qs(url.query).get("from", ["0"])[0])
                rows, reset = read_rows(target, start)
                mtime = target.stat().st_mtime if target.exists() else None
                self._send(json.dumps({"rows": rows, "reset": reset, "mtime": mtime}).encode(),
                           "application/json")
            elif url.path in ("/", "/index.html"):
                # Built per request rather than once at startup: the title and the path in the
                # header have to name the run this URL is actually showing.
                page = build_page(f"nanoSpeaker — {target.parent.name}", poll, shown=shown)
                self._send(page.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def log_message(self, *_):
            pass                  # the poll would otherwise print a line every few seconds

    try:
        shown = path.resolve().relative_to(root)
    except ValueError:
        shown = path
    httpd = HTTPServer(("127.0.0.1", port), Handler)   # bind before announcing the URL
    print(f"dashboard: http://127.0.0.1:{port}  tracking {shown}")
    print(f"           another run: http://127.0.0.1:{port}/?path=runs/<name>/metrics.jsonl")
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="?", default="runs/nanospeaker/metrics.jsonl",
                    help="path to the metrics.jsonl to track, relative to where you run this")
    ap.add_argument("--path", dest="metrics_flag",
                    help=argparse.SUPPRESS)      # the old spelling; the positional is the way now
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--poll", type=int, default=5, help="seconds between client polls")
    ap.add_argument("--snapshot", help="write a standalone HTML file with the data baked in, and exit")
    args = ap.parse_args()

    path = Path(args.metrics_flag or args.metrics)
    if not path.exists():
        ap.error(f"no such file: {path}  (resolved against {Path.cwd()})")

    # Shown relative wherever it can be: "runs/x/metrics.jsonl" is the answer someone wants
    # in a header, not sixty characters of home directory.
    try:
        shown = path.resolve().relative_to(Path.cwd())
    except ValueError:
        shown = path

    if args.snapshot:
        rows, _ = read_rows(path, 0)
        Path(args.snapshot).write_text(
            build_page(f"nanoSpeaker — {path.parent.name}", args.poll,
                       rows, path.stat().st_mtime, shown))
        print(f"{len(rows)} steps from {shown} -> {args.snapshot}")
        return
    serve(path, args.port, args.poll, Path.cwd().resolve())


if __name__ == "__main__":
    main()
