---
name: molab-fingerprints-notebook
description: >-
  Build and iterate on the molab Notebook Competition entry in this repo — an
  interactive marimo notebook explaining the history, mechanics, applications,
  and limitations of molecular fingerprints for cheminformatics. Use this skill
  whenever working on `fingerprints_notebook.py`, its supporting modules under
  `src/fingerprints/`, custom anywidgets, or any competition-related planning.
  It carries the project plan, the judging rubric, and the hard-won gotchas of
  this sandbox + marimo + RDKit setup.
---

# molab Fingerprints Notebook

We are building a competition entry for OpenADMET × marimo's first
cheminformatics **molab Notebook Competition**. The deliverable is a single,
compelling, interactive, visually satisfying **marimo notebook**
(`fingerprints_notebook.py`) that gives someone real intuition for **molecular
fingerprints**: their history, mechanics, applications, and limitations.

The competition prompt: turn real drug-discovery ADMET results into something
tangible, reproducible, and interactive. Anyone who works through the notebook
should walk away with genuine intuition for the concept and a sense of how to
try it themselves.

## Read this first

Also load the `marimo-pair` skill — it covers how to drive a running marimo
kernel (create/edit/run cells, install packages, wire widgets). This skill is
the *project* layer on top of it.

## What earns points (the rubric IS the spec)

Judged by a hand-grading panel including **Pat Walters** and **2 marimo team
members**. They have explicitly said generic AI-prompted notebooks get stale by
the 20th one — customization and a notebook that *feels ours* is rewarded. AI
use is fine but **must be disclosed in the notebook itself** (see the existing
README disclaimer; do the same in-notebook, in the spirit of Jesse Hartman's
example).

Weighted rubric — treat every build decision as "which box does this check?":

| Weight | Criterion | What it means for us |
|--------|-----------|----------------------|
| 20% | **Creativity & Impact** | Fresh angle, not a summary. Novel visualizations, cross-referencing related work, real-world ADMET relevance. Make the case for *why fingerprints matter* and their broader implications. |
| 20% | **Interactivity & Workflow Design** | Use marimo's reactive model fully. Widgets must be **meaningful, not decorative** — they drive genuine exploration. Cells build on each other logically. |
| 20% | **Design, Presentation & Shareability** | Polished layout, clear headings, good `mo.md()` context. Should make a viewer want to explore OpenADMET / try marimo. |
| 20% | **Customization** | Build our **own anywidget or package** rather than only stock `mo.ui`. It must solve a real need off-the-shelf components can't — a molecule viewer, structure editor, scaffold/reaction/bit explorer — and feel purpose-built, well-integrated into the reactive flow. |
| 10% | **Code Quality & Clarity** | Clean, readable cells; self-contained and reproducible with minimal setup; dependencies/data documented. **Notebooks with errors are DISQUALIFIED** — always exercise every cell path before declaring done. |
| 10% | **Chemical Validity** | Valid SMILES/structure parsing, correct stereochemistry, sensible edge cases (invalid structures, salts, tautomers). Scientifically sound comparisons — **no train/test leakage from near-duplicate structures**. A working chemist must trust the claims. |

Customization is 20% and the panel's stated differentiator: **the custom
anywidget is not optional polish — it is a primary scoring lever.** Chemical
validity being a disqualifier (errors) + 10% means we never ship a cell that can
throw on an edge case.

## The baseline we build from

This repo already contains a substantial fingerprint investigation (see
`README.md` — it is effectively the notebook's script). Reuse it; don't
reinvent it:

- **8 fingerprints under one interface** (`src/fingerprints/fingerprint_methods/`):
  Morgan, RDKit-topological, AtomPair, TopTorsion, MACCS, Avalon (classical) +
  CheMeleon, MIST-28M (neural). `FingerprintResult` exposes `.array`, `.kind`
  (`binary`/`count`/`continuous`), `.name`, `.n_features`.
- **Similarity math** (`fingerprint_methods/similarity.py`): `pairwise_similarity`
  (Tanimoto for binary, cosine for continuous), `pairwise_distance`.
- **Data loaders** (`src/fingerprints/data/`): hand-picked `pairs.py`, `chembl.py`,
  MoleculeACE cliffs `molace.py`, TDC ADME `tdc.py`.
- **Analyses** (`src/fingerprints/clustering/`): RV agreement, scaffolds,
  ADME alignment, activity cliffs (matched-control PR-AUC).
- **Figures** under `figures/` — the findings the notebook dramatizes.
- **`src/fingerprints/maccs_explorer.py`** (added for this notebook): resolve a
  MACCS bit against a molecule (`bit_hit`), render highlighted SVG
  (`highlight_svg`), curated `BIT_NAMES`. Handles special (count-based) bits.

Key findings worth surfacing interactively (from README):
- Fingerprints encode genuinely different similarity geometries (RV matrix).
- All FPs struggle with **activity cliffs** once graph distance is controlled
  (median matched-control PR-AUC 0.55–0.61; Morgan/TopTorsion best).
- Neural cosine is NOT on the same scale as Tanimoto — thresholds don't port.
- Property gradients (logS) are visible in neural-FP UMAP geometry, blob-like
  in classical FPs.

## Notebook plan (narrative arc)

`fingerprints_notebook.py`, `app = marimo.App(width="medium")`. Four-part arc:

1. **What is a fingerprint? — MACCS explorer** *(in progress, Tier 1 done)*
   Type SMILES → scrub a slider over the molecule's ON bits → highlight the
   exact atoms that fired each bit, with bit name + SMARTS + match count.
   MACCS chosen deliberately: least complex, each bit is a human-readable SMARTS.
   **This is our first WOW.** Keep scope tight. Upgrade to a custom **anywidget**
   for hover/click-to-highlight-in-browser (no round-trip) + a full 166-bit grid
   view — this is the customization scoring lever.

2. **Mechanics of other fingerprints** — how Morgan (atom environments),
   path-based, and neural FPs differ. Reuse the fingerprint methods + pair data.

3. **Applications — similarity & ADMET** — reactive similarity playground
   (pick 2 mols + FP + threshold), brushable UMAP colored by logS reusing the
   ADME embeddings.

4. **Limitations — activity cliffs** — dramatize the cliff-blindness finding
   interactively; let the user find pairs where FPs disagree with potency.

Include an **AI-use disclosure** cell and a **sources/reproducibility** cell.

## First WOW interaction (agreed scope)

MACCS bit explorer, kept deliberately simple to start:
- Tier 1 (done, pure `mo.ui`): slider over ON bits → server re-renders SVG.
- Anywidget upgrade (the goal): render substructures client-side, hover/click a
  bit to highlight instantly, show the bit grid. This is where "really cool"
  lives and directly targets the 20% Customization criterion.

## Interactivity tiers (reference)

- **Tier 1 — server-side reactive:** `mo.ui.*` (slider/text/dropdown/table) →
  Python recomputes → RDKit renders fresh SVG/PNG. No JS. Robust, exportable.
- **Tier 2 — anywidget:** custom HTML/CSS/JS with traitlet state synced to
  Python, bidirectional, no round-trip. The supported customization path.
- **Tier 3 — RDKit WASM (`@rdkit/rdkit`) inside an anywidget:** cheminformatics
  in the browser (parse SMILES, depict, fingerprint, substructure match) — the
  most fluid, makes exported HTML self-contained. Also JSME/Ketcher for drawing.

## Sandbox + environment gotchas (learned the hard way)

- The agent shell is **jailed to the repo root**. `~`, `/tmp`, `$HOME`, and the
  `.venv/bin/*` symlinks (which point outside the repo) all fail with a `jailed`
  error. Consequences:
  - Run Python via **`uv run python`** / **`uv run marimo`**, never `.venv/bin/*`.
  - Write logs and scratch files **inside the repo** (e.g. `marimo.log`,
    gitignored), never `/tmp`.
  - The `marimo-pair` skill scripts live outside the repo, so you **cannot**
    invoke `discover-servers.sh` / `execute-code.sh` from the jailed shell.
    Drive the notebook by editing `fingerprints_notebook.py` **while no kernel
    owns it**, or reach a running server by URL with `curl`.
- **Start marimo headless** here: the sandbox can't open a browser. Use
  `uv run marimo edit fingerprints_notebook.py --no-token --headless --port 2718`
  as a background task and tell the user to open `http://localhost:2718`.
- marimo 0.24.0, RDKit 2026.03.2, Python 3.11. Add deps with
  `uv add` (or `ctx.packages.add()` if driving a live kernel), not pip.

## Working rules for this project

- **Never ship a cell that can throw.** Errors = disqualification. Every cell
  that touches user input must handle invalid SMILES, empty input, salts,
  tautomers, and disconnected structures gracefully. Exercise every path with
  `uv run python` before claiming done.
- **Reuse `src/fingerprints/` modules**; put new reusable logic there (tested,
  importable) and keep notebook cells thin and readable.
- **Every widget must earn its place** — it drives exploration or it's cut.
- **Disclose AI assistance in-notebook.**
- **Prefer edits over shell redirects** for any file a human will read (use the
  `edit`/`write` tools so changes render as diffs, not opaque shell output).
- When chemistry claims appear in prose, they must match what the code computes
  and be defensible to a working chemist. No overreaching (the README already
  models this discipline).
