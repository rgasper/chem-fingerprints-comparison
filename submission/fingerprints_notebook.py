import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    from fingerprints import cliff_view as cv

    return alt, cv, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Exploring Molecular Fingerprints and How They Fail

    A **molecular fingerprint** turns a molecule into a fixed row of numbers so a computer can compare two
    molecules by comparing their rows. It's the workhorse representation behind
    a lot of cheminformatics operations: similarity search, clustering, and property
    prediction for example are frequently based partially or entirely off of fingerprints as input.
    This notebook takes fingerprints apart to see what they encode and where that encoding commonly fails.

    To begin, we'll highlight one of the well-known failings of molecular fingerprints: activity cliffs. We'll spend most of the rest of the notebook trying to build an intuitive understanding for how fingerprints are related to activity cliffs, and also how they're not!

    *AI was used in the creation of this notebook. For the full disclaimer, head to the very bottom*
    """)
    return


@app.cell
def _(mo):
    # --- Setup controls ------------------------------------------------------
    # Everything the plots need ships PRECOMPUTED under data/ (the cliff
    # censuses, kNN analysis, feature-importances, and the 3D Boltz poses), so
    # the notebook loads instantly. The only thing fetched on first run is the
    # 33 MB CheMeleon neural-fingerprint weight, which the live interactive
    # cells need. If you're curious how the precomputed data was made, tick the
    # box below to rebuild it all from scratch (downloads the source datasets
    # and re-runs every analysis — takes ~10 minutes, dominated by CheMeleon).
    recompute_toggle = mo.ui.checkbox(
        value=False, label="Recompute all analyses from scratch (~10 min)"
    )
    boltz_api_key = mo.ui.text(
        value="",
        label="Boltz API key (only to re-fold the 3D poses — optional)",
        kind="password",
        full_width=True,
        placeholder="leave blank to use the shipped poses",
    )
    return boltz_api_key, recompute_toggle


@app.cell
def _(boltz_api_key, mo, recompute_toggle):
    from fingerprints import chemeleon_fp as _chf
    from fingerprints import paths as _paths
    from fingerprints import recompute as _recompute

    # Report which precomputed artifacts are present (they ship with the repo).
    def _status(label, ok):
        mark = "✅" if ok else "⚠️"
        return f"{mark} {label}"

    _rows = [
        _status("ADMET cliff census (TDC + OpenADMET)", _paths.ADMET_CLIFFS.exists()),
        _status("kNN cliff analysis", _paths.KNN_CLIFFS.exists()),
        _status("Feature-importance models", _paths.IMPORTANCE.exists()),
        _status(
            "3D Boltz poses + PLIP interactions",
            _paths.BOLTZ_POSES.exists() and any(_paths.BOLTZ_POSES.glob("*.cif")),
        ),
    ]

    if recompute_toggle.value:
        # Full from-scratch rebuild, wrapped in a progress bar.
        _recompute.clear_outputs()
        _steps = _recompute.steps()
        with mo.status.progress_bar(
            total=len(_steps), title="Rebuilding analyses", remove_on_exit=False
        ) as _bar:
            for _step in _steps:
                _bar.update(subtitle=_step.title)
                _step.run(lambda msg, _t=_step.title: _bar.update(subtitle=f"{_t}: {msg}"))
                _bar.update(increment=1)
        setup_ready = True
        _setup_msg = mo.md(
            "**Rebuilt everything from scratch.** The plots below now read the "
            "freshly-computed files — identical machinery, no shipped caches."
        ).callout(kind="success")
    else:
        # Fast path: just make sure the neural-fingerprint weight is present.
        with mo.status.spinner(title="Fetching CheMeleon weights (first run only)…"):
            _chf.ensure_weights()
        setup_ready = True
        _setup_msg = mo.md(
            "Using the **precomputed** analysis data shipped with this notebook "
            "(the plots load instantly). CheMeleon neural-fingerprint weights are "
            "ready."
        ).callout(kind="neutral")

    # The 3D poses are the one artifact we don't rebuild inline: folding needs a
    # GPU and a Boltz API key. If a key is supplied we surface how to re-fold;
    # otherwise the shipped poses are used as-is.
    if boltz_api_key.value.strip():
        _pose_note = mo.md(
            "Boltz key detected. Re-folding the 3D poses is a separate, GPU-heavy "
            "job; run `python -m fingerprints.rebuild_poses` with `BOLTZ_API_KEY` "
            "set (extra deps: `pip install '.[poses]'`). The shipped poses are "
            "used until you do."
        ).callout(kind="info")
    else:
        _pose_note = mo.md("")

    mo.vstack(
        [
            _setup_msg,
            mo.md("**Precomputed artifacts:**  \n" + "  \n".join(_rows)),
            recompute_toggle,
            boltz_api_key,
            _pose_note,
        ]
    )
    return (setup_ready,)


@app.cell
def _(mo):
    mo.md(r"""
    Here are two real molecules that differ by only a few atoms - we've got a few options you can choose from, to help demonstrate that this is not a phenomenon specific to this exact choice of chemicals. Whichever case you pick, the two molecules are nearly identical in structure, and nearly - sometimes exactly - identical in fingerprint. And yet their measured potency against closely related enzymes differs by orders of magnitude — this is an *activity cliff*.

    Pick a target pair and a molecule pair below. These are
    two intentionally chosen, closely related, targets from the MoleculeACE dataset
    where the same small change
    to chemical structure is a cliff on one target and barely a ripple on the other.
    """)
    return


@app.cell
def _(mo):
    from fingerprints.data import context_cliffs as ctx

    # One canonical target-pair dropdown, bound to a GLOBAL so marimo tracks it
    # reactively. The molecule-pair dropdown is derived from it in the next cell
    # (its options depend on the chosen pair). Displaying the SAME element object
    # in several places keeps every copy in sync automatically - no mo.state,
    # no on_change handler recreating elements (which silently broke updates).
    pair_dd = mo.ui.dropdown(
        options=ctx.pair_options(),
        value=next(iter(ctx.pair_options())),
        label="Target pair",
    )
    return ctx, pair_dd


@app.cell
def _(ctx, mo, pair_dd):
    # Molecule-pair dropdown, rebuilt whenever the target pair changes so its
    # options match. Bound to a global (cliff_dd) for reactive tracking.
    _pk = pair_dd.value
    _cliff_opts = ctx.cliff_options(_pk)
    cliff_dd = mo.ui.dropdown(
        options=_cliff_opts,
        value=next(iter(_cliff_opts)),
        label="Molecule pair",
    )
    return (cliff_dd,)


@app.cell
def _(ctx, cliff_dd, mo, pair_dd):
    # Accessors used throughout the notebook. get_pair_key() / get_cliff_idx()
    # read the current dropdown selections; selectors() renders the synced row.
    def get_pair_key():
        return pair_dd.value

    def get_cliff_idx():
        _pk = pair_dd.value
        return ctx.clamp_cliff_idx(_pk, cliff_dd.value or 0)

    def selectors():
        """The synced (target-pair, molecule-pair) dropdown row. Renders the
        same global elements, so every placement stays in lock-step."""
        return mo.hstack([pair_dd, cliff_dd], justify="start", gap=2)

    return get_cliff_idx, get_pair_key, selectors


@app.cell
def _(ctx, get_pair_key, mo, selectors):
    _tp = ctx.by_key()[get_pair_key()]
    mo.vstack([mo.md(f"*{_tp.blurb}*"), selectors()])
    return


@app.cell
def _(ctx, cv, get_cliff_idx, get_pair_key, mo):
    _tp = ctx.by_key()[get_pair_key()]
    _pair = _tp.cliffs[get_cliff_idx()]
    _svg1, _svg2 = cv.pair_svgs(_pair, width=320, height=240)

    # Structures with the changed atoms highlighted in orange.
    _structures = mo.hstack(
        [
            mo.vstack([mo.Html(_svg1)], align="center"),
            mo.md("## →"),
            mo.vstack([mo.Html(_svg2)], align="center"),
        ],
        justify="center",
        gap=1,
    )
    _change = mo.md(
        f"The change: {_pair.change}.  \nThe orange atoms are all that differ "
        "between these two molecules."
    ).callout(kind="neutral")

    # Dual-endpoint activity readout: cliff on one, flat on the other.
    def _endpoint_card(target, pki1, pki2):
        delta = abs(pki1 - pki2)
        is_cliff = target == _pair.cliff_on
        fold = cv.fold_change(delta)
        kind = "danger" if is_cliff else "success"
        verdict = f"**{fold} potency change** — a cliff!" if is_cliff else (
            f"**{fold} — essentially unchanged** (flat)"
        )
        return mo.md(
            f"#### {target}\n\n"
            f"pKi: **{pki1}** → **{pki2}**  \n{verdict}"
        ).callout(kind=kind)

    _endpoints = mo.hstack(
        [
            _endpoint_card(_pair.target_a, _pair.pki_1_a, _pair.pki_2_a),
            _endpoint_card(_pair.target_b, _pair.pki_1_b, _pair.pki_2_b),
        ],
        widths=[1, 1],
        gap=2,
    )
    mo.vstack([_structures, _change, _endpoints, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Same two molecules, same fingerprints, opposite truths: a cliff on one
    target, flat on the other. Your first instinct might be that this is just a
    limitation of *similarity* — that a properly **trained model** would learn to
    tell these two apart. So let's test exactly that. We'll train a model on each
    target and ask where it looks.
    """)
    return




@app.cell
def _(cv, ctx, get_cliff_idx, get_pair_key, mo, setup_ready):
    from rdkit import Chem as _Chem

    from fingerprints import importance_view as iv

    assert setup_ready  # gate on CheMeleon weights (importance heatmaps use them)

    # Where does a MODEL look? For BOTH targets, train a RandomForest to predict
    # activity from each fingerprint, project its feature importances back onto
    # each cliff molecule, and diff the two targets' attention. The diff is the
    # real test: a fingerprint that captured the biology would re-weight toward
    # the atom that flips the activity. A static one barely moves — and where it
    # does move is not where the change is.
    _tp = ctx.by_key()[get_pair_key()]
    _cl = _tp.cliffs[get_cliff_idx()]
    _cliff_ep, _flat_ep = _cl.cliff_on, _cl.flat_on
    _eps = iv.endpoints() if iv.has_data() else []
    _have = _tp.target_a in _eps and _tp.target_b in _eps
    if not _have:
        _view = mo.md(
            "*Feature-importance models weren't precomputed for this pair yet. "
            "(Run `python -m fingerprints.analyses.importance` to add it.)*"
        ).callout(kind="info")
    else:
        _m1 = _Chem.MolFromSmiles(_cl.smiles_1)
        _m2 = _Chem.MolFromSmiles(_cl.smiles_2)
        _changed1, _changed2 = cv.changed_atoms(_cl)

        def _pred_label(mol, mol_idx, ep, fp):
            _a = _cl.actual_pki(mol_idx, ep)
            _p = iv.predict(ep, fp, mol)
            if _p is None:
                return f"measured **{_a:.2f}**"
            return f"pred **{_p:.2f}** / meas **{_a:.2f}**"

        def _mini(svg, caption):
            return mo.vstack([mo.Html(svg), mo.md(caption)], align="center")

        def _mol_row(mol, mol_idx, changed, fp):
            _h_cliff = iv.importance_heatmap_svg(mol, _cliff_ep, fp, width=230, height=180)
            _h_flat = iv.importance_heatmap_svg(mol, _flat_ep, fp, width=230, height=180)
            _h_diff = iv.importance_diff_heatmap_svg(
                mol, _cliff_ep, _flat_ep, fp, width=230, height=180
            )
            _st = iv.importance_diff_stats(mol, _cliff_ep, _flat_ep, fp, changed)
            _diff_cap = (
                f"**attention shift** {_cliff_ep} − {_flat_ep}  \n"
                + (
                    (
                        "the biggest shift **is** on a changed atom"
                        if _st["peak_on_changed"]
                        else "the biggest shift is on the wrong part of the "
                             "molecule — **not** the changed atoms"
                    )
                    if changed
                    else "the map still lights up — even though *nothing changed* "
                         "on this molecule"
                )
            )
            return mo.vstack([
                mo.md(f"**molecule {mol_idx}**"),
                mo.hstack(
                    [
                        _mini(_h_cliff, f"{_cliff_ep} (cliff)  \n{_pred_label(mol, mol_idx, _cliff_ep, fp)}"),
                        _mini(_h_flat, f"{_flat_ep} (flat)  \n{_pred_label(mol, mol_idx, _flat_ep, fp)}"),
                        _mini(_h_diff, _diff_cap),
                    ],
                    widths=[1, 1, 1], gap=1,
                ),
            ])

        def _fp_block(fp, fp_name):
            return mo.vstack([
                mo.md(f"#### {fp_name}"),
                _mol_row(_m1, 1, _changed1, fp),
                _mol_row(_m2, 2, _changed2, fp),
            ])

        _mt_ec, _mt_ef = iv.metrics(_cliff_ep, "ecfp"), iv.metrics(_flat_ep, "ecfp")
        _mt_cc, _mt_cf = iv.metrics(_cliff_ep, "chemeleon"), iv.metrics(_flat_ep, "chemeleon")

        # Quantify the miss on the CLIFF target: measured vs predicted gap.
        def _gap(ep, fp):
            _p1 = iv.predict(ep, fp, _m1)
            _p2 = iv.predict(ep, fp, _m2)
            return abs(_p1 - _p2) if (_p1 is not None and _p2 is not None) else None

        _true_gap = abs(_cl.actual_pki(1, _cliff_ep) - _cl.actual_pki(2, _cliff_ep))
        _pred_gap_e = _gap(_cliff_ep, "ecfp")
        # Rank-based check that MATCHES the autoscaled heatmap: does the model's
        # single biggest attention shift land on a changed atom? Computed
        # SEPARATELY per fingerprint, over the molecules that actually carry the
        # structural change - the two fingerprints can (and do) disagree here.
        def _shift_stats(fp):
            hits = [
                iv.importance_diff_stats(_mol, _cliff_ep, _flat_ep, fp, _ch)
                for _mol, _ch in ((_m1, _changed1), (_m2, _changed2))
                if _ch
            ]
            return sum(1 for s in hits if s["peak_on_changed"]), len(hits)

        _e_on, _n_hits = _shift_stats("ecfp")
        _c_on, _ = _shift_stats("chemeleon")

        _gap_line = (
            f" On **{_cliff_ep}** the measured pKi gap between the two molecules is "
            f"**{_true_gap:.1f}** log units, but the ECFP model predicts a gap of just "
            f"**{_pred_gap_e:.1f}** — it flattens the cliff."
            if _pred_gap_e is not None
            else ""
        )
        # Describe each fingerprint honestly: sometimes the learned fingerprint's
        # loudest shift *does* land on a changed atom, even when ECFP's doesn't.
        def _phrase(n_on, n):
            return f"on **{n_on} of {n}** the biggest shift lands on a changed atom"

        if _n_hits:
            _both_miss = _e_on == 0 and _c_on == 0
            if _both_miss:
                _diff_line = (
                    " The diff panels tell the same story visually: the model's "
                    "attention *does* move between targets, but the loudest shifts "
                    "land away from the handful of atoms that actually changed — "
                    f"for ECFP {_phrase(_e_on, _n_hits)}, and likewise for CheMeleon "
                    f"({_phrase(_c_on, _n_hits)}). The model re-weights plenty, just "
                    "not where the chemistry actually moved."
                )
            else:
                _diff_line = (
                    " The diff panels are more nuanced here. The models' attention "
                    "*does* move between targets: for the fixed **ECFP** fingerprint "
                    f"{_phrase(_e_on, _n_hits)}, while for the **learned CheMeleon** "
                    f"fingerprint {_phrase(_c_on, _n_hits)}. So on this pair the "
                    "learned representation sometimes *does* put its loudest shift on "
                    "an atom that changed — a hint that a representation learned from "
                    "data can occasionally localise the change better than a fixed "
                    "one. But note this attention landing on the right atom still "
                    "isn't enough: as the predicted-gap numbers show, neither model "
                    "actually *resolves* the potency cliff."
                )
        else:
            _diff_line = ""
        _summary = mo.md(
            "The similarity metric isn't the only culprit — a trained model inherits "
            "the same blind spot. Below, a RandomForest predicts pKi for **both** "
            f"targets ({_cliff_ep} and {_flat_ep}) from each fingerprint. For every "
            "molecule we show where the model looks for each target and, in the third "
            "panel, the **difference** between them."
            + _gap_line
            + _diff_line
        )
        _details = mo.accordion({
            "Model & training details": mo.md(
                "**Task.** Regress measured pKi from a frozen fingerprint (the "
                "fingerprint is *not* trained; only the head is), separately for each "
                "target.\n\n"
                "**Model.** `RandomForestRegressor` "
                f"({iv.config()['rf_trees']} trees, scikit-learn defaults otherwise), "
                "trained independently for each endpoint and each fingerprint.\n\n"
                "**Fingerprints.** ECFP (Morgan, radius 2, "
                f"{iv.config()['n_bits']} bits) and the frozen CheMeleon embedding.\n\n"
                "**Split.** 80/20 **Bemis–Murcko scaffold split** (whole scaffold "
                "groups go to one side only), so the reported R²/RMSE are measured on "
                "held-out chemotypes with no near-duplicate leakage. Note the curated "
                "cliff molecules shown above may fall in either fold — they're used "
                "only to *visualise* where the trained model looks, not as a "
                "leakage-free benchmark.\n\n"
                f"**{_cliff_ep} (cliff).** "
                f"ECFP: {_mt_ec['n_train']:,} train / {_mt_ec['n_test']:,} test, "
                f"R² {_mt_ec['r2']:.2f}, RMSE {_mt_ec['rmse']:.2f}. "
                f"CheMeleon: R² {_mt_cc['r2']:.2f}, RMSE {_mt_cc['rmse']:.2f}.\n\n"
                f"**{_flat_ep} (flat).** "
                f"ECFP: {_mt_ef['n_train']:,} train / {_mt_ef['n_test']:,} test, "
                f"R² {_mt_ef['r2']:.2f}, RMSE {_mt_ef['rmse']:.2f}. "
                f"CheMeleon: R² {_mt_cf['r2']:.2f}, RMSE {_mt_cf['rmse']:.2f}.\n\n"
                "**Attribution.** ECFP importances spread each ON bit's RF importance "
                "over the atoms of its environment; CheMeleon importances weight each "
                "atom by its exact contribution to the top model dimensions. The diff "
                "panel L1-normalises each target's attention, subtracts, and shades red "
                "where the cliff-target model leans harder, blue where the flat-target "
                "model does. Because that map is autoscaled to its own strongest atom, "
                "we read it by *rank*: we ask whether the single largest shift (the most "
                "saturated atom) falls on one of the changed atoms, rather than summing a "
                "raw magnitude the eye can't verify. All are honest *estimates* of where "
                "the model looks, not ground-truth substructure claims."
            )
        })
        _grid = mo.vstack([_fp_block("ecfp", "ECFP"), _fp_block("chemeleon", "CheMeleon")])
        _view = mo.vstack([_summary, _details, _grid])
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # What does a fingerprint actually encode?

    To understand why the fingerprints are failing to notice this huge difference in activity, we need to see in detail what a
    fingerprint records in the first place. Fingerprints come in two families:

    - **Classical** — MACCS, Morgan/ECFP, and the other classical
      RDKit fingerprints. A person or a fixed algorithm decided in advance which
      substructures cause which bits to activate. The simple way to compare molecules for similarity is the
      Tanimoto (AKA [Jaccard](https://en.wikipedia.org/wiki/Jaccard_index)) overlap of those bits, and this similarity metric is directly related to some human-understandable difference between molecules.
    - **Learned** — CheMeleon is a neural network *pre-trained* to read the molecular
      graph and predict a wide swath of physicochemical properties; we then extract out the final embedding vector before the MLP decision head to use as a fingerprint. While it is possible to use cosine similarity with chemeleon vectors, due to anisotropy in the embedding space it's often not meaningful - almost all realistic chemical compounds will end up quite close in embedding space. See [a recent related work studying this in transformer models](https://arxiv.org/html/2401.12143v2) for more discussion on that.

    These two fingerprint types differ in construction significantly, but share one important aspect - they're static. They can't reactively change to new contexts in chemical or target variable space.

    Below we look at exactly what the fingerprints encode for the **two molecules from the cliff you're examining** — the same near-identical pair from above. Pick either molecule (or type your own SMILES), then scrub through bits/dimensions to see what parts of the molecule each fingerprint records. Because the two cliff molecules differ by only a handful of atoms, watching how little the fingerprint changes between them is the whole point.
    """)
    return


@app.cell
def _(ctx, get_cliff_idx, get_pair_key, mo, selectors):
    # Reuse the SAME two molecules from the cliff the reader picked above,
    # instead of an unrelated gallery — it keeps this section tied to the
    # story. The menu is rebuilt reactively from the current (pair, cliff)
    # selection; a synced copy of the selectors is shown here so the reader
    # can switch cliffs without scrolling back up.
    _mol_opts = ctx.cliff_molecule_options(get_pair_key(), get_cliff_idx())
    mol_choice = mo.ui.dropdown(
        options=_mol_opts,
        value=next(iter(_mol_opts)),
        label="Molecule from this cliff",
    )
    custom_smiles = mo.ui.text(
        value="",
        label="…or type your own SMILES (overrides the dropdown)",
        full_width=True,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    )
    _picker = mo.vstack(
        [
            mo.md("**Switch the cliff pair (synced with the sections above):**"),
            selectors(),
        ]
    )
    _picker
    return custom_smiles, mol_choice


@app.cell
def _(custom_smiles, mo, mol_choice):
    from fingerprints import maccs_explorer as mx

    # Resolve the single upstream molecule: custom SMILES wins if provided,
    # otherwise use the molecule picked from the current cliff pair. The
    # dropdown's VALUE is already the molecule's SMILES (see
    # ctx.cliff_molecule_options), so everything downstream depends only on
    # `current_mol` / `current_label`.
    _typed = custom_smiles.value.strip()
    if _typed:
        _mol = mx.mol_from_smiles(_typed)
        _source_label = "custom SMILES"
        _picked_note = ""
    else:
        _smiles = mol_choice.value
        _mol = mx.mol_from_smiles(_smiles)
        # the dropdown key (e.g. "Molecule 1 — before (…)") is the nice label
        _source_label = next(
            (k for k, v in mol_choice.options.items() if v == _smiles),
            "cliff molecule",
        )
        _picked_note = "one of the two molecules from the cliff above"

    current_mol = _mol
    mol_valid = current_mol is not None
    current_label = _source_label

    if _typed and not mol_valid:
        _feedback = mo.md(
            f"⚠️ Could not parse SMILES `{_typed}` — showing nothing downstream "
            f"until it's valid."
        ).callout(kind="warn")
    elif mol_valid:
        from rdkit import Chem

        _canon = Chem.MolToSmiles(current_mol)
        _note = f" — *{_picked_note}*" if _picked_note else ""
        _feedback = mo.md(
            f"**Active molecule: {current_label}**{_note}  \n`{_canon}`"
        ).callout(kind="success")
    else:
        _feedback = mo.md("")

    mo.vstack([mo.hstack([mol_choice, custom_smiles], widths=[1, 2]), _feedback])
    return current_mol, mol_valid, mx


@app.cell
def _(current_mol, mo, mol_valid, mx):
    # Always-visible depiction of the active molecule, so the selector has an
    # immediate, satisfying response independent of the analyses below.
    if mol_valid:
        _svg = mx.highlight_svg(
            current_mol,
            mx.blank_hit(),
            width=360,
            height=260,
        )
        _view = mo.Html(_svg)
    else:
        _view = mo.md("*No valid molecule selected.*")
    _view
    return


@app.cell
def _(mo, mol_valid, mx):
    # Always scrub all 166 MACCS keys, in order, so OFF bits are explorable too.
    scrub_bits = mx.all_bits() if mol_valid else []
    if scrub_bits:
        bit_slider = mo.ui.slider(
            start=0,
            stop=len(scrub_bits) - 1,
            value=0,
            label="Scrub all 166 MACCS keys",
            full_width=True,
            show_value=False,
        )
    else:
        bit_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")
    return bit_slider, scrub_bits


@app.cell
def _(bit_slider, current_mol, mo, mol_valid, mx, scrub_bits):
    if mol_valid and scrub_bits:
        _bit = scrub_bits[bit_slider.value]
        _hit = mx.bit_hit(current_mol, _bit)
        _query = mx.query_svg(_bit, width=240, height=190)

        # Show where it matches (ON) or the plain molecule (OFF). The match/OFF
        # state is already spelled out in the match line below, so no separate
        # "present" badge is needed — it would be redundant.
        if _hit.is_on:
            _mol_svg = mx.highlight_svg(current_mol, _hit, width=460, height=340)
            _mol_panel = mo.vstack([mo.md("**Where it matches:**"), mo.Html(_mol_svg)])
        else:
            _mol_svg = mx.highlight_svg(current_mol, mx.blank_hit(), width=460, height=340)
            _mol_panel = mo.vstack(
                [
                    mo.md("**Not present** — the molecule is shown plain:"),
                    mo.Html(_mol_svg),
                ]
            )

        # "What the bit looks for": the SMARTS query depiction, or — for the
        # three procedurally-computed keys — a plain-language explanation.
        if _query is not None:
            _query_panel = mo.vstack(
                [mo.md("**What this bit looks for:**"), mo.Html(_query)]
            )
        else:
            _query_panel = mo.vstack(
                [
                    mo.md("**What this bit looks for:**"),
                    mo.md(mx.describe_special(_bit) or "*No drawable pattern.*").callout(
                        kind="info"
                    ),
                ]
            )

        # Headline the plain-English description; tuck the terse SMARTS and the
        # official MDL key definition into an expandable "technical details" pane
        # so nobody has to parse a SMARTS string to understand the bit.
        if _hit.is_special:
            _match_line = mo.md(
                "*Special key — RDKit computes this one directly instead of by "
                "matching a SMARTS pattern (see explanation at right).*"
            )
            _details = mo.accordion(
                {
                    "Technical details": mo.md(
                        f"**Official MACCS key:** `{_hit.official}`  \n"
                        "**SMARTS:** *(none — computed procedurally)*"
                    )
                }
            )
        else:
            if _hit.threshold > 0:
                _match_line = mo.md(
                    f"This key needs **more than {_hit.threshold}** matches to turn "
                    f"on. Found **{_hit.match_count}** → "
                    f"{'**ON**' if _hit.is_on else '**OFF**'}."
                )
            else:
                _match_line = mo.md(f"Matches in this molecule: **{_hit.match_count}**")
            _details = mo.accordion(
                {
                    "Technical details": mo.md(
                        f"**Official MACCS key:** `{_hit.official}`  \n"
                        f"**SMARTS:** `{_hit.smarts}`"
                    )
                }
            )

        _header = mo.vstack(
            [
                mo.md(f"### Bit {_hit.bit} · {_hit.name}"),
                _match_line,
                _details,
            ]
        )
        # The strip goes below the molecule image (same layout as Morgan): it's
        # easier to parse the full-fingerprint context after seeing the match.
        _strip = mx.fingerprint_strip_svg(current_mol, _bit, width=920, height=44)
        _strip_legend = mo.md(
            '<span style="color:#2f9e44">█ on</span> &nbsp; '
            '<span style="color:#adb5bd">░ off</span> &nbsp; '
            '<span style="color:#1c7ed6">█ current bit</span>'
        )
        _view = mo.vstack(
            [
                mo.md("---"),
                _header,
                mo.hstack([_query_panel, _mol_panel], justify="start", gap=2, widths=[1, 2]),
                mo.md("**Where this bit sits in the whole 166-bit fingerprint:**"),
                mo.Html(_strip),
                _strip_legend,
                bit_slider,
                mo.md("---"),
            ]
        )
    else:
        _view = mo.vstack(
            [mo.md("*Select a valid molecule to explore its MACCS bits.*"), mo.md("---")]
        )
    _view
    return


@app.cell
def _(current_mol, mo, mol_valid):
    from fingerprints import morgan_explorer as me

    if mol_valid:
        _on = me.on_bits(current_mol)
    else:
        _on = []
    morgan_on_bits = _on

    if _on:
        morgan_slider = mo.ui.slider(
            start=0,
            stop=len(_on) - 1,
            value=0,
            label=f"Scrub the {len(_on)} Morgan bits that are ON (radius 2, 2048 bits)",
            full_width=True,
            show_value=False,
        )
    else:
        morgan_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")
    return me, morgan_on_bits, morgan_slider


@app.cell
def _(current_mol, me, mo, mol_valid, morgan_on_bits, morgan_slider):
    if mol_valid and morgan_on_bits:
        _bit = morgan_on_bits[morgan_slider.value]
        _hit = me.bit_hit(current_mol, _bit)
        _svg = me.highlight_svg(current_mol, _hit, width=460, height=340)
        _radii = ", ".join(str(r) for r in _hit.radii)
        _center_word = "center" if _hit.n_instances == 1 else "centers"
        _card = mo.vstack(
            [
                mo.md(f"### Bit {_hit.bit}"),
                mo.md(
                    f"Set by **{_hit.n_instances}** atom {_center_word} "
                    f"(radius {_radii})."
                ),
                mo.md(
                    f"Environment spans **{len(_hit.atoms)} atoms** "
                    f"and **{len(_hit.bonds)} bonds**."
                ),
                mo.md(
                    "*Radius 0 bits are single atoms; larger radii capture more "
                    "of the surrounding structure.*"
                ),
            ]
        )
        _view = mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[1, 1])
    else:
        _view = mo.md("*Select a valid molecule to explore its Morgan bits.*")
    _view
    return


@app.cell
def _(current_mol, me, mo, mol_valid, morgan_on_bits, morgan_slider):
    # Same strip idea as MACCS, but Morgan is long (2048) and sparse: an OFF bit
    # means "no environment happened to hash here" — it has no specific meaning,
    # so the scrubber only visits ON bits and the strip is mostly blank.
    if mol_valid and morgan_on_bits:
        _bit = morgan_on_bits[morgan_slider.value]
        _strip = me.fingerprint_strip_svg(current_mol, _bit, width=920, height=36)
        _legend = mo.md(
            '<span style="color:#2f9e44">█ on</span> &nbsp; '
            '<span style="color:#1c7ed6">█ current bit</span> &nbsp; '
            "the rest is off"
        )
        _note = mo.md(
            f"Only **{len(morgan_on_bits)} of 2048** bits are on — Morgan vectors "
            "are sparse. Unlike MACCS, an off bit here carries no meaning of its "
            "own - it just means no atom environment hashed to that bit, so the "
            "scrubber skips straight between the on bits."
        )
        _view = mo.vstack(
            [mo.md("**The whole 2048-bit fingerprint:**"), mo.Html(_strip), _legend, _note,
             morgan_slider, mo.md("---")]
        )
    else:
        _view = mo.md("---")
    _view
    return


@app.cell
def _(current_mol, me, mo, mol_valid):
    _SHORT = 8
    _collisions = me.find_collisions(current_mol, n_bits=_SHORT) if mol_valid else []
    if _collisions:
        collision_slider = mo.ui.slider(
            start=0,
            stop=len(_collisions) - 1,
            value=0,
            label=f"Scrub the {len(_collisions)} colliding bits at {_SHORT} bits",
            full_width=True,
            show_value=False,
        )
    else:
        collision_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no collisions)")
    short_collisions = _collisions
    return collision_slider, short_collisions


@app.cell
def _(collision_slider, current_mol, me, mo, mol_valid, short_collisions):
    _PALETTE_HEX = ["#e64d3d", "#338cf2", "#33a659", "#d98c1a", "#9959cc"]
    if not mol_valid:
        collision_card = mo.md("*Select a valid molecule.*")
    elif not short_collisions:
        collision_card = mo.md(
            "This molecule has so few atom environments that **none of them "
            "collide** even at 8 bits — try a bigger drug-like molecule from "
            "the selector (e.g. Gefitinib or Imatinib)."
        ).callout(kind="info")
    else:
        _col = short_collisions[collision_slider.value]
        _svg = me.collision_svg(current_mol, _col, width=520, height=380)
        # One legend row per colliding environment, colored to match the drawing.
        _rows = []
        for _i, _sig in enumerate(_col.signatures):
            _c = _PALETTE_HEX[_i % len(_PALETTE_HEX)]
            _label = _sig.replace("atom:", "single atom ")
            _rows.append(f'<span style="color:{_c}">█</span> `{_label}`')
        _legend = mo.md("  \n".join(_rows))
        _card = mo.vstack(
            [
                mo.md(f"### Bit {_col.bit} at 8 bits"),
                mo.md(
                    f"**{_col.n_distinct} different substructures** all hash to this "
                    "one bit. To the fingerprint they are indistinguishable:"
                ),
                _legend,
                mo.md(
                    "*In a 2048-bit fingerprint these substructures would almost always land "
                    "on separate bits — that extra length allows greater specificity."
                ),
            ]
        )
        collision_card = mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[3, 2])
    return (collision_card,)


@app.cell
def _(alt, current_mol, me, mo, mol_valid, pd):
    # Collision rate as the vector lengthens: the payoff of a longer fingerprint.
    if mol_valid:
        _curve = me.collision_curve(current_mol)
        _distinct = _curve[0].distinct_envs
        _df = pd.DataFrame(
            {
                "length": [cp.n_bits for cp in _curve],
                "rate": [
                    round(100 * cp.collisions / cp.distinct_envs, 1)
                    if cp.distinct_envs
                    else 0.0
                    for cp in _curve
                ],
            }
        )
        _chart = (
            alt.Chart(_df)
            .mark_line(point=True, color="#e8590c")
            .encode(
                x=alt.X(
                    "length:O",
                    title="fingerprint length (bits)",
                    sort=[str(cp.n_bits) for cp in _curve],
                ),
                y=alt.Y(
                    "rate:Q",
                    title="collision rate (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("length:O", title="bits"),
                    alt.Tooltip("rate:Q", title="collision rate %"),
                ],
            )
            .properties(height=200, title="Collision rate vs. fingerprint length")
        )
        collision_curve_view = mo.vstack(
            [
                mo.as_html(_chart),
                mo.md(
                    f"This molecule has {_distinct} distinct atom environments. "
                    "The collision rate falls off fast — which is why **2048 bits** is a common "
                    "default: long enough that collisions are rare, short enough to "
                    "stay cheap."
                ),
            ]
        )
    else:
        collision_curve_view = mo.md("")
    return (collision_curve_view,)


@app.cell
def _(collision_card, collision_curve_view, collision_slider, mo):
    mo.vstack(
        [
            mo.accordion(
                {
                    "🔍 Aside: why is a Morgan fingerprint 2048 bits long? (hash collisions)": mo.vstack(
                        [
                            mo.md(
                                "Morgan has *no* fixed vocabulary, so it can't reserve a slot "
                                "per feature the way MACCS does — it **hashes** each atom "
                                "environment into one of *N* bits. Make *N* too small and "
                                "different substructures collide onto the same bit. Squeeze it "
                                "down to just **8 bits** and watch distinct environments pile "
                                "up on one slot:"
                            ),
                            collision_card,
                            collision_curve_view,
                            collision_slider,
                        ]
                    )
                }
            ),
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(current_mol, mo, mol_valid):
    from fingerprints import classical_explorer as ce

    # Each fingerprint gets its OWN top-level slider variable. marimo only tracks
    # reactivity for mo.ui elements bound directly to a global name - sliders
    # hidden inside a dict/list do NOT trigger downstream re-runs.
    def _slider(key):
        on = ce.on_bits(current_mol, key) if mol_valid else []
        if on:
            return mo.ui.slider(
                start=0, stop=len(on) - 1, value=0,
                label=f"Scrub the {len(on)} bits that are ON",
                full_width=True, show_value=False,
            )
        return mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")

    topo_slider = _slider("rdkit_topo")
    ap_slider = _slider("atom_pair")
    tt_slider = _slider("top_torsion")
    return ap_slider, ce, topo_slider, tt_slider


@app.cell
def _(ap_slider, ce, current_mol, mo, mol_valid, topo_slider, tt_slider):
    def _fp_tab(key, slider):
        info = ce.FP_INFO[key]
        on = ce.on_bits(current_mol, key) if mol_valid else []
        if not (mol_valid and on):
            body = mo.md("*Select a valid molecule.*")
        else:
            bit = on[min(slider.value, len(on) - 1)]
            hit = ce.bit_hit(current_mol, key, bit)
            svg = ce.highlight_svg(current_mol, hit, width=440, height=320)
            strip = ce.fingerprint_strip_svg(current_mol, key, bit, width=900, height=34)
            card = mo.vstack(
                [
                    mo.md(f"### Bit {hit.bit}"),
                    mo.md(
                        f"Set by **{hit.n_instances}** substructure"
                        f"{'s' if hit.n_instances != 1 else ''} — highlighting "
                        f"**{len(hit.atoms)} atoms**."
                    ),
                ]
            )
            body = mo.vstack(
                [
                    mo.hstack([mo.Html(svg), card], justify="start", gap=2, widths=[3, 2]),
                    mo.md("**Where this bit sits in the full 2048-bit vector:**"),
                    mo.Html(strip),
                    slider,
                ]
            )
        return mo.vstack([mo.md(f"*{info.blurb}*"), body])

    tabbed_fps = mo.ui.tabs(
        {
            ce.FP_INFO["rdkit_topo"].label: _fp_tab("rdkit_topo", topo_slider),
            ce.FP_INFO["atom_pair"].label: _fp_tab("atom_pair", ap_slider),
            ce.FP_INFO["top_torsion"].label: _fp_tab("top_torsion", tt_slider),
        }
    )
    mo.vstack(
        [
            mo.md("### The rest of the RDKit toolbox (topological, atom-pair, torsion)"),
            mo.md(
                "Beyond MACCS's checklist and Morgan's circular environments, "
                "RDKit ships several more \"classical\" fingerprints. They each "
                "encode a different notion of structure — paths, atom pairs at "
                "a distance, torsions — but share Morgan's hashing machinery. "
                "These differences matter: later we'll demonstrate that "
                "certain fingerprints can properly capture activity cliffs for one ADMET endpoint "
                "but can fail on another, and which fingerprint works best for which endpoint is not consistent."
            ),
            tabbed_fps,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Now the *learned* family: CheMeleon

    **CheMeleon** is a message-passing neural network
    trained on a large dataset, and its 2048 dimensions were learned,
    not designed. There's no vocabulary to scrub and no Tanimoto — instead each
    dimension is a continuous feature with a complex relationship to the input molecule. Below we show one way of visualizing what it's doing: pick a dimension and see which atoms drive that dimension for the currently selected molecule. In contrast to the "classical" fingerprints, we have to use shading instead of clear attribution, and you'll see that some of the most sensitive dimensions encode multiple seemingly unrelated parts of the molecule at once.

    In this case, we're using the last embedding dimension before the MLP decision head that was used when CheMeleon was
    pre-trained to predict thousands of physico-chemical features on millions of random drug-like molecules. If you practiced the usual approach to fine-tune the model on your particular dataset, this analysis would produce different results on identical molecules, since the model had to learn a new mapping of molecular structure to data.
    """)
    return


@app.cell
def _(mo):
    chemeleon_floor = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.05,
        value=0.5,
        label="Structure-sensitivity floor (fraction of this molecule's most structure-sensitive dimension)",
        show_value=True,
        full_width=True,
    )
    return (chemeleon_floor,)


@app.cell
def _(chemeleon_floor, current_mol, mo, mol_valid, setup_ready):
    from fingerprints import chemeleon_fp as chf

    # CheMeleon is a *pretrained* neural fingerprint: nobody hand-designed its
    # 2048 dimensions - a message-passing network learned them from large
    # molecular data. Pick a dimension and see which atoms drive it for the
    # current molecule (the learned analog of the Morgan bit-scrubber). The
    # floor knob controls how sensitive a dimension must be to count.
    # `setup_ready` gates this cell so the neural-fingerprint weights are
    # guaranteed present (downloaded by the setup cell) before the first
    # CheMeleon forward pass runs.
    assert setup_ready
    if mol_valid and current_mol.GetNumAtoms() >= 2:
        _dims = chf.most_active_dims(current_mol, floor_frac=chemeleon_floor.value)
    else:
        _dims = [0]
    chemeleon_dim = mo.ui.slider(
        start=0,
        stop=max(len(_dims) - 1, 0),
        value=0,
        label=f"Scrub {len(_dims)} structure-sensitive dimensions (by index)",
        full_width=True,
        show_value=False,
    )
    chemeleon_dims = _dims
    return chemeleon_dim, chemeleon_dims, chf


@app.cell
def _(chemeleon_dim, chemeleon_dims, chemeleon_floor, chf, current_mol, mo, mol_valid):
    # Render the current molecule as a heatmap of one learned dimension's
    # per-atom contributions. Exact decomposition: the graph fingerprint is a
    # mean over atoms, so atom i's share of dimension k is H[i,k]/n_atoms.
    if not mol_valid:
        _view = mo.md("*Select a valid molecule.*")
    else:
        _dim = chemeleon_dims[min(chemeleon_dim.value, len(chemeleon_dims) - 1)]
        _svg = chf.heatmap_svg(current_mol, _dim, width=460, height=340)
        _strip = chf.strip_svg(
            current_mol, _dim, active_dims=chemeleon_dims, width=920, height=40
        )
        _strip_legend = mo.md(
            '<span style="color:#7048e8">█ dimension magnitude |fₖ|</span> &nbsp; '
            '<span style="color:#2f9e44">█ structure-sensitive (scrubbable)</span> &nbsp; '
            '<span style="color:#1c7ed6">█ selected dimension</span>'
        )
        _card = mo.md(
            f"### CheMeleon dimension {_dim}\n\n"
            f"A **pretrained, learned** fingerprint — nobody chose these features.\n\n"
            f"<span style='color:#2b8a3e'>● green</span> atoms push this dimension "
            f"up, <span style='color:#c2255c'>● pink</span> push it down. This is an "
            f"estimated read of what parts of this molecule most affect this dimension — "
            f"not a fixed substructure like a Morgan bit."
        )
        _view = mo.vstack(
            [
                mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[3, 2]),
                mo.md("**Where this dimension sits in the full 2048-long vector:**"),
                mo.Html(_strip),
                _strip_legend,
                chemeleon_floor,
                chemeleon_dim,
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📐 What does “structure-sensitivity” mean? (the math)": mo.md(
                r"""
    CheMeleon reads the molecular graph and, after message passing, produces a
    **per-atom hidden vector** $h_i \in \mathbb{R}^{2048}$ for every atom $i$. The
    molecule's fingerprint is just the **mean over atoms**:

    $$ f_k \;=\; \frac{1}{N}\sum_{i=1}^{N} h_{i,k}, \qquad k = 1,\dots,2048 $$

    Because the pooling is a plain mean, each atom's contribution to dimension $k$
    is **exact** (no approximation):

    $$ c_{i,k} \;=\; \frac{h_{i,k}}{N}, \qquad \sum_{i=1}^{N} c_{i,k} = f_k $$

    That is what the molecule heatmap draws for a chosen $k$.

    **Structure-sensitivity** of dimension $k$ is how much that contribution
    *varies across the atoms* of this molecule — we use the spread

    $$ s_k \;=\; \max_i h_{i,k} \;-\; \min_i h_{i,k} $$

    - **Large $s_k$:** different atoms push the dimension very differently, so the
      dimension is reading a **local** structural feature — its heatmap is
      informative (some atoms light up, others don't).
    - **Small $s_k$:** every atom contributes about the same, so the dimension
      encodes something **diffuse/global** and its heatmap would be flat.

    The scrubber offers only the **structure-sensitive** dimensions: those whose
    spread clears a relative floor you set with the knob,

    $$ s_k \;\ge\; \phi \cdot \max_j s_j, \qquad \phi \in [0.1,\,0.9] $$

    i.e. at least a fraction $\phi$ as sensitive as this molecule's most-sensitive
    dimension (default $\phi = 0.5$). Raise $\phi$ for a stricter, smaller set;
    lower it to include more. Either way the *count* varies by molecule (like the
    on-bit count of ECFP/MACCS), and the scrubber steps through them **in index
    order**. The purple strip shows each dimension's magnitude $|f_k|$; green ticks
    mark the structure-sensitive dimensions; blue marks the one you're viewing.

    This is an exact decomposition of the **mean-pool**, but a
    single learned dimension rarely maps to one human-named substructure the way a
    Morgan bit does — read it as “where this dimension looks,” not “what it is.”
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Activity cliffs are not a protein-binding-specific issue

    So far the presentation has centered on examining pairs of compounds which exhibit an activity cliff against one protein but don't against a second closely related protein. The point of this was to clearly demonstrate that this is not a fault of the chemistry. The core concept of the activity cliff is generalizable: it's an unavoidable property of any map of chemical structure to measured data - so it should break the
    same way for **ADMET** properties, where there's no obliging second target,
    just one number per molecule. If the cliff is really a property of the
    *encoding* and not the biology, we should find it here too.

    So let's run an analysis on two
    ADMET endpoints from Therapeutics Data Commons: **aqueous solubility
    (AqSolDB)** and **lipophilicity (AstraZeneca logD)**. Same question: among
    the molecules a fingerprint calls *similar*, how often is the property
    actually similar?
    """)
    return


@app.cell
def _(mo):
    from fingerprints import admet_view as adm

    if adm.has_data():
        _eps = adm.endpoints()
        admet_choice = mo.ui.dropdown(
            options=_eps, value=_eps[0], label="ADMET endpoint"
        )
    else:
        admet_choice = mo.ui.dropdown(options={"(none)": ""}, value="(none)")
    admet_choice
    return adm, admet_choice


@app.cell
def _(adm, admet_choice, alt, mo, pd):
    from rdkit import Chem as _Chem
    from rdkit.Chem.Draw import rdMolDraw2D as _d2d

    # The ADMET census, split per fingerprint: for EACH fingerprint we show its
    # own gap histogram (which pairs it calls "similar", and how cliffy they
    # are) and, directly beneath, the single sharpest cliff THAT fingerprint
    # would wave through as similar. The columns make the point concrete: every
    # fingerprint draws "similar" differently, yet each has its own bad cliff.
    def _pair_svg(smi, w=132, h=100):
        m = _Chem.MolFromSmiles(smi)
        d = _d2d.MolDraw2DSVG(w, h)
        d.drawOptions().addStereoAnnotation = False
        if m is not None:
            _d2d.PrepareAndDrawMolecule(d, m)
        d.FinishDrawing()
        return d.GetDrawingText()

    if not adm.has_data() or admet_choice.value not in adm.endpoints():
        admet_census_view = mo.md(
            "*ADMET census not precomputed — run "
            "`python -m fingerprints.analyses.admet`.*"
        ).callout(kind="info")
    else:
        _ep = admet_choice.value
        _mt = adm.meta(_ep)
        _s = adm.smoothness(_ep)
        _unit = _mt["unit"]

        def _label_gap(h):
            return f"{h['lo']:.1f}\u2013{h['hi']:.1f}" if h["hi"] < 20 else f"{h['lo']:.1f}+"

        def _classify(h):
            if h["hi"] <= _s["flat_gap"]:
                return "flat"
            if h["lo"] >= _s["cliff_gap"]:
                return "cliff"
            return "middle"

        def _hist_chart(fp_entry):
            _rows = [
                {"gap": _label_gap(h), "lo": h["lo"], "count": h["count"],
                 "kind": _classify(h)}
                for h in fp_entry["gap_hist"]
            ]
            return (
                alt.Chart(pd.DataFrame(_rows))
                .mark_bar()
                .encode(
                    x=alt.X("gap:N", sort=alt.SortField("lo"), title=None,
                            axis=alt.Axis(labelAngle=-45, labelFontSize=7)),
                    y=alt.Y("count:Q", title="similar pairs"),
                    color=alt.Color(
                        "kind:N",
                        scale=alt.Scale(
                            domain=["flat", "middle", "cliff"],
                            range=["#2b8a3e", "#adb5bd", "#e03131"],
                        ),
                        legend=None,
                    ),
                    tooltip=["gap:N", "count:Q"],
                )
                .properties(width=150, height=130)
            )

        def _cliff_block(fp_entry):
            c = fp_entry.get("top_cliff")
            if not c:
                return mo.md(
                    "<div style='text-align:center;color:#adb5bd;font-size:11px'>"
                    "no cliff among its similar pairs</div>"
                )
            _mult = f"{10 ** c['gap']:,.0f}\u00d7" if c["gap"] < 12 else "huge"
            return mo.vstack(
                [
                    mo.hstack(
                        [mo.Html(_pair_svg(c["smiles_1"])),
                         mo.Html(_pair_svg(c["smiles_2"]))],
                        justify="center", gap=0.25,
                    ),
                    mo.md(
                        f"<div style='text-align:center;font-size:10px'>"
                        f"T <b>{c['tanimoto']:.2f}</b> · {_unit} "
                        f"{c['act_1']:.1f} vs {c['act_2']:.1f}<br>"
                        f"<span style='color:#e03131'>gap {c['gap']:.1f} "
                        f"(~{_mult} \u2260)</span></div>"
                    ),
                ]
            )

        _per_fp = _s.get("per_fp", {})
        if _per_fp:
            _cols = []
            for _fv in _per_fp.values():
                _cliff_pct_fp = round(_fv["frac_cliff"] * 100, 1)
                _cols.append(
                    mo.vstack(
                        [
                            mo.md(
                                f"<div style='text-align:center;font-size:12px'>"
                                f"<b>{_fv['label']}</b><br>"
                                f"<span style='color:#868e96'>"
                                f"{_fv['n_similar_pairs']:,} similar · "
                                f"{_cliff_pct_fp}% cliffs</span></div>"
                            ),
                            mo.as_html(_hist_chart(_fv)),
                            _cliff_block(_fv),
                        ]
                    )
                )
            _grid = mo.hstack(_cols, widths=[1] * len(_cols), gap=1)
        else:
            _grid = mo.md("")

        _flat_pct = round(_s["frac_flat"] * 100)
        _cliff_pct = round(_s["frac_cliff"] * 100, 1)
        _msg = mo.md(
            f"**{_ep}** ({_unit}) — **{_mt['n_total']:,}** unique structures "
            f"(salts stripped, deduplicated), scaffold-split "
            f"{_mt['n_train']:,} train / {_mt['n_test']:,} test.\n\n"
            f"Among the **{_s['n_similar_pairs']:,}** pairs **Morgan** calls "
            f"*similar* (Tanimoto ≥ {_s['sim_threshold']:.1f}), **{_flat_pct}%** "
            f"are flat (within {_s['flat_gap']:.1f} log unit) but **{_cliff_pct}%** "
            f"are outright **cliffs** (> {_s['cliff_gap']:.1f} log units — more than "
            f"~30× apart). A single-atom or chain-length "
            f"change can move solubility by orders of magnitude while the "
            f"fingerprint barely changes - or sometimes not at all. **Each column repeats the census under a "
            f"different fingerprint's similarity, with that fingerprint's own "
            f"sharpest cliff drawn beneath** — the exact 'similar' set shifts, "
            f"but every fingerprint has a stubborn red cliff tail and a real pair "
            f"to show for it."
        ).callout(kind="danger" if _s["frac_cliff"] > 0.2 else "warn")
        admet_census_view = mo.vstack(
            [mo.md(f"*{_mt['blurb']}*"), _msg, _grid]
        )
    admet_census_view
    return


@app.cell
def _(adm, alt, mo, pd):
    from fingerprints import knn_view as _knn

    # The payoff comparison across ALL THREE endpoints, split by fingerprint:
    # cliffiness varies wildly by target (you can't know in advance), AND which
    # fingerprint happens to minimise cliffs flips per target - so you can't
    # know in advance which encoding will serve a new endpoint either.
    if not adm.has_data():
        admet_compare_view = mo.md("")
    else:
        _palette = _knn.fp_colors()  # shared with the kNN charts
        _rows = []
        for _e in adm.endpoints():
            _unit = adm.meta(_e)["unit"]
            _elabel = f"{_e}\n({_unit})"
            for _fv in adm.per_fp(_e).values():
                _rows.append(
                    {
                        "endpoint": _elabel,
                        "fingerprint": _fv["label"],
                        "cliff_pct": round(_fv["frac_cliff"] * 100, 1),
                    }
                )
        _df = pd.DataFrame(_rows)
        _ep_order = [f"{_e}\n({adm.meta(_e)['unit']})" for _e in adm.endpoints()]
        _fp_order = [_fv["label"] for _fv in adm.per_fp(adm.endpoints()[0]).values()]
        _chart = (
            alt.Chart(_df)
            .mark_bar()
            .encode(
                x=alt.X("endpoint:N", title=None, sort=_ep_order,
                        axis=alt.Axis(labelAngle=0, labelLimit=200)),
                xOffset=alt.XOffset("fingerprint:N", sort=_fp_order),
                y=alt.Y("cliff_pct:Q",
                        title="% of similar pairs that are cliffs",
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "fingerprint:N",
                    scale=alt.Scale(
                        domain=_fp_order,
                        range=[_palette.get(lbl, "#868e96") for lbl in _fp_order],
                    ),
                    legend=alt.Legend(title=None, orient="bottom", columns=5),
                ),
                tooltip=["endpoint:N", "fingerprint:N", "cliff_pct:Q"],
            )
            .properties(height=280)
        )
        admet_compare_view = mo.vstack(
            [
                mo.md(
                    "### You can't know the 'cliffiness' — or the best fingerprint "
                    "— in advance\n\n"
                    "The same census across those ADMET endpoints — three public "
                    "TDC benchmarks **plus OpenADMET's own ExpansionRx LogD and "
                    "solubility** — now broken out by fingerprint:"
                ),
                mo.as_html(_chart),
                mo.md(
                    "Three things are only knowable after you "
                    "have the data:\n\n"
                    "1. **'Cliffiness 'swings by endpoint.** Aqueous "
                    "solubility (AqSolDB) has very many activity cliffs, lipophilicity "
                    "far less, and Caco-2 permeability barely any. There is some intuitive sense here - Solubility "
                    "hinges on crystal packing and H-bonding a single atom can "
                    "shatter; logD is a smoother bulk average - but telling ourselves we can understand this after-the-fact does not mean we can reliably predict it for new dataset in the future.\n\n"
                    "2. **No fingerprint is safest everywhere.** On solubility "
                    "**atom-pair** often flags fewer cliffs than Morgan or MACCS "
                    "— its distance-based similarity happens to align with what "
                    "drives solubility — but that lead can disappear on other "
                    "endpoints. You could only *learn* which encoding suits an "
                    "endpoint by measuring it first.\n\n"
                    "3. **Activity cliffs are partially a symptom of less standardized data** OpenADMET's "
                    "ExpansionRx LogD and solubility show a *lower* cliff rate "
                    "than the aggregated public AqSolDB — consistent with a "
                    "single-platform, controlled-condition measurement (less "
                    "inter-lab variation in measurements). The presence of cliffs is a "
                    "property of the data as well as the chemistry. The problem is that until you have a "
                    "more tightly standardized dataset to compare, you may not know that you're currently working with the noisier one. A model trained on data from another lab may not play well with data from your lab.\n\n"
                ).callout(kind="info"),
                mo.md("---"),
            ]
        )
    admet_compare_view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why any fingerprint-based model will fail to learn the cliffs

    So the same failure shows up whether we're predicting binding or solubility
    — it travels with the fingerprint, not the biology. That raises the real
    question: is this a modelling mistake we could engineer away, or something
    deeper?

    To explore that, first we strip the model down to a minimal example. Use **k-nearest-neighbours**
    on the fingerprint: a molecule's prediction is just the **average value of
    its k most-similar neighbours**, where "similar" is Tanimoto Similarity of the fingerprints. Nothing is
    learned on top — the fingerprint's notion of similarity *is* the entire
    model. So whatever kNN structurally cannot do is the fingerprint's own
    limitation, with no fancy decision boundaries to obfuscate the situation.
    """)
    return


@app.cell
def _(alt, ctx, cv, get_cliff_idx, get_pair_key, mo, pd):
    # Concrete lead-in to the kNN argument: every CLASSICAL structure
    # fingerprint scores the cliff pair as broadly similar - one Tanimoto
    # number, blind to which endpoint it's applied to. Classical fingerprints
    # only: their native metric is Tanimoto, so the comparison is
    # apples-to-apples. (Learned fingerprints don't define a similarity of
    # their own - that's why kNN below is a fair, minimal stand-in.)
    _tp = ctx.by_key()[get_pair_key()]
    _pair = _tp.cliffs[get_cliff_idx()]
    _scores = cv.fingerprint_scores(_pair, classical_only=True)
    _df = pd.DataFrame(
        [
            {"fingerprint": s.label, "similarity": round(s.similarity, 3)}
            for s in _scores
        ]
    )
    _chart = (
        alt.Chart(_df)
        .mark_bar(cornerRadius=3, color="#4c6ef5")
        .encode(
            x=alt.X(
                "similarity:Q",
                title="fingerprint similarity",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("fingerprint:N", sort="-x"),
            tooltip=[alt.Tooltip("fingerprint:N"), alt.Tooltip("similarity:Q")],
        )
        .properties(height=250)
    )
    _lo = min(s.similarity for s in _scores)
    _hi = max(s.similarity for s in _scores)
    _punchline = mo.md(
        f"Each named fingerprint calls this pair similar (Tanimoto "
        f"{_lo:.2f}–{_hi:.2f}). Since kNN's only notion of 'close' IS this "
        f"similarity, its neighbours — and its prediction — will be nearly "
        f"identical for the two molecules, right for {_pair.flat_on} and wrong "
        f"for {_pair.cliff_on}."
    ).callout(kind="warn")
    mo.vstack(
        [
            mo.md("**What each fingerprint's similarity metric sees:**"),
            mo.as_html(_chart),
            _punchline,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    # Resample button for the flat-pair gallery below (a fresh random draw of
    # 'similar structure, similar activity' pairs - the majority the assumption
    # gets right). value increments on each click -> reactive reseed.
    resample_flat = mo.ui.button(
        label="🎲 Sample different flat pairs", value=0, on_click=lambda v: v + 1
    )
    return (resample_flat,)


@app.cell
def _(alt, ctx, get_cliff_idx, get_pair_key, mo, pd, resample_flat):
    from fingerprints import knn_view as knn

    # The general impossibility argument, made concrete. Any structure-only
    # model is effectively a *smooth* function of the fingerprint: similar
    # fingerprint -> similar predicted activity. This census shows WHY that
    # assumption is forced - among structurally similar pairs, the overwhelming
    # majority really are flat, so a model must default to smooth to be
    # accurate, and the rare cliff is collateral damage.
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    _cliff_ep = _tp.cliffs[_idx].cliff_on if _tp.cliffs else None
    _eps = knn.endpoints() if knn.has_data() else []

    if _cliff_ep not in _eps:
        _view = mo.md(
            "*This census wasn't precomputed for this pair yet — "
            "pick another pair above.*"
        ).callout(kind="info")
    else:
        _s = knn.smoothness(_cliff_ep)
        _hist = pd.DataFrame(
            [
                {
                    "gap": f"{h['lo']:.1f}–{h['hi']:.1f}" if h["hi"] < 10 else f"{h['lo']:.1f}+",
                    "lo": h["lo"],
                    "count": h["count"],
                    "kind": (
                        "flat" if h["hi"] <= _s["flat_gap"]
                        else "cliff" if h["lo"] >= _s["cliff_gap"]
                        else "middle"
                    ),
                }
                for h in _s["gap_hist"]
            ]
        )
        _chart = (
            alt.Chart(_hist)
            .mark_bar()
            .encode(
                x=alt.X("gap:N", sort=alt.SortField("lo"),
                        title="activity gap between the pair (|ΔpKi|, log units)"),
                y=alt.Y("count:Q", title="number of similar pairs"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=["flat", "middle", "cliff"],
                        range=["#2b8a3e", "#adb5bd", "#e03131"],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["gap:N", "count:Q"],
            )
            .properties(height=220)
        )
        _flat_pct = round(_s["frac_flat"] * 100)
        _cliff_pct = _s["frac_cliff"] * 100
        _ratio = round(_s["frac_flat"] / max(_s["frac_cliff"], 1e-9))
        _msg = mo.md(
            f" Take every pair of {_cliff_ep} "
            f"molecules that a fingerprint calls *similar* (Tanimoto ≥ "
            f"{_s['sim_threshold']:.1f}): **{_s['n_similar_pairs']:,}** pairs. "
            f"**{_flat_pct}%** of them are **flat** (activity within "
            f"{_s['flat_gap']:.0f} log unit) and only **{_cliff_pct:.1f}%** are "
            f"true cliffs — roughly **{_ratio}:1**.\n\n"
            f"So 'similar structure → similar activity' is *right the vast majority "
            f"of the time*. Any model that predicts from structure alone is "
            f"rewarded for learning it — and a model that instead predicted big "
            f"activity jumps for near-identical structures would be wrong on the flat "
            f"{_flat_pct}% to catch the cliffy {_cliff_pct:.1f}%. "
            "This results in an unresolvable tension between specific and global accuracy for whatever fingerprint-based model we pick. **A cliff is where reality "
            f"breaks the very assumption that makes the fingerprint useful.** No "
            f"amount of model cleverness recovers information the structure encoding "
            f"never contained — which is why activity cliffs are a well-documented "
            f"challenge in cheminformatics, not a modelling failure."
        ).callout(kind="danger")

        # A gallery of the flat majority: similar structures whose activity
        # really is similar (resampled on the button click).
        from rdkit import Chem as _Chem
        from rdkit.Chem.Draw import rdMolDraw2D as _d2d

        def _pair_svg(s1, s2, w=150, h=110):
            def one(s):
                m = _Chem.MolFromSmiles(s)
                d = _d2d.MolDraw2DSVG(w, h)
                d.drawOptions().addStereoAnnotation = False
                if m is not None:
                    _d2d.PrepareAndDrawMolecule(d, m)
                d.FinishDrawing()
                return d.GetDrawingText()
            return one(s1), one(s2)

        def _flat_card(fp):
            a, b = _pair_svg(fp["smiles_1"], fp["smiles_2"])
            gap = abs(fp["act_1"] - fp["act_2"])
            return mo.vstack(
                [
                    mo.hstack([mo.Html(a), mo.Html(b)], justify="center", gap=0.5),
                    mo.md(
                        f"<div style='text-align:center;font-size:12px'>"
                        f"Tanimoto <b>{fp['tanimoto']:.2f}</b> · pKi "
                        f"{fp['act_1']:.1f} vs {fp['act_2']:.1f} — "
                        f"<span style='color:#2b8a3e'>gap {gap:.2f} (flat ✓)</span></div>"
                    ),
                ]
            )

        _flats = knn.sample_flat_pairs(_cliff_ep, 3, seed=resample_flat.value)
        _gallery = mo.vstack(
            [
                mo.md(
                    "The flat majority — **similar structure, similar activity** is almost always right. "
                ),
                mo.hstack(
                    [_flat_card(fp) for fp in _flats] or [mo.md("*(no pairs)*")],
                    widths=[1] * max(len(_flats), 1), gap=1,
                ),
                resample_flat,
            ]
        )
        _view = mo.vstack([_msg, mo.as_html(_chart), _gallery])
    mo.vstack([_view, mo.md("---")])
    return (knn,)


@app.cell
def _(knn, mo):
    # The only knob kNN has is neighbourhood size k - the honest analog of
    # "move the decision boundary". Top-level so the section reacts to it.
    _grid = knn.k_grid() if knn.has_data() else [1, 5, 20]
    k_slider = mo.ui.slider(
        steps=_grid,
        value=_grid[min(3, len(_grid) - 1)],
        label="Neighbourhood size k",
        show_value=True,
        full_width=True,
    )
    return (k_slider,)


@app.cell
def _(mo, selectors):
    # Synced selector mirror: switch target pair / molecule pair right
    # here without scrolling back to the top (same global elements).
    mo.vstack([mo.md('**Pick the target pair / molecule pair for this kNN demo:**'), selectors()])
    return


@app.cell
def _(alt, ctx, get_cliff_idx, get_pair_key, k_slider, knn, mo, pd):
    # kNN cliff analysis, precomputed per endpoint (all curated pairs).
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    _k = k_slider.value

    _eps = knn.endpoints() if knn.has_data() else []
    _cliff_ep = _tp.cliffs[_idx].cliff_on if _tp.cliffs else None
    _pair = knn.cliff_pair(_cliff_ep, _idx) if _cliff_ep in _eps else None

    if _pair is None:
        _view = mo.md(
            "*The kNN cliff analysis wasn't precomputed for this pair yet — "
            "pick another pair above. "
            "(Run `python -m fingerprints.analyses.knn` to add it.)*"
        ).callout(kind="info")
    else:
        _ep = _pair["cliff_on"]
        _m1, _m2 = _pair["mol1"], _pair["mol2"]
        _meta = knn.endpoint_meta(_ep)
        _best = knn.best_k(_ep)

        # (a) The k tradeoff, overlaid for EVERY fingerprint's similarity - the
        # accuracy/k curve (and where it peaks) is itself a choice of
        # fingerprint. Each line is one fingerprint's held-out kNN accuracy.
        # One fixed palette is shared with the per-fingerprint cliff scatter (b).
        _fp_palette = knn.fp_colors()
        _by_fp = knn.k_curves_by_fp(_ep)
        if _by_fp:
            _rows = []
            for _fk, _fv in _by_fp.items():
                for _pt in _fv["k_curve"]:
                    _rows.append(
                        {"k": _pt["k"], "r2": _pt["r2"], "fingerprint": _fv["label"]}
                    )
            _curve = pd.DataFrame(_rows)
            _fp_labels = [_fv["label"] for _fv in _by_fp.values()]
            _color_enc = alt.Color(
                "fingerprint:N",
                scale=alt.Scale(
                    domain=_fp_labels,
                    range=[_fp_palette.get(lbl, "#868e96") for lbl in _fp_labels],
                ),
                legend=alt.Legend(title=None, orient="bottom", columns=2),
            )
            _line = (
                alt.Chart(_curve)
                .mark_line(point=True)
                .encode(
                    x=alt.X("k:Q", title="neighbourhood size k",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("r2:Q", title="held-out R² (accuracy)"),
                    color=_color_enc,
                    tooltip=["fingerprint:N", "k:Q",
                             alt.Tooltip("r2:Q", format=".3f")],
                )
            )
        else:
            _curve = pd.DataFrame(knn.k_curve(_ep))
            _line = (
                alt.Chart(_curve)
                .mark_line(point=True, color="#4c6ef5")
                .encode(
                    x=alt.X("k:Q", title="neighbourhood size k",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("r2:Q", title="held-out R² (accuracy)"),
                    tooltip=["k:Q", alt.Tooltip("r2:Q", format=".3f")],
                )
            )
        _rule = (
            alt.Chart(pd.DataFrame({"k": [_k]}))
            .mark_rule(color="#e8820c", strokeWidth=2)
            .encode(x="k:Q")
        )
        _acc_top = (_line + _rule).properties(height=170, width=300)

        # (a2) The SAME k axis, but RMSE on the *cliff molecules only* (every
        # cliff pair in this endpoint, not just the highlighted one). Stacked
        # directly under the global-accuracy panel so the one orange current-k
        # rule reads across both: as you slide k, watch global accuracy rise to
        # a peak while the cliff error stays stubbornly flat and high - no
        # neighbourhood size rescues the cliffs.
        _cliff_rmse = knn.cliff_rmse_by_fp(_ep)
        _n_cliff = knn.n_cliff_pairs(_ep)
        if _cliff_rmse:
            _crows = []
            for _fv in _cliff_rmse.values():
                for _pt in _fv["rmse_curve"]:
                    _crows.append(
                        {"k": _pt["k"], "rmse": _pt["rmse"],
                         "fingerprint": _fv["label"]}
                    )
            _cdf = pd.DataFrame(_crows)
            _cline = (
                alt.Chart(_cdf)
                .mark_line(point=True, strokeDash=[4, 2])
                .encode(
                    x=alt.X("k:Q", title="neighbourhood size k",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("rmse:Q", title="cliff-pair RMSE (log units)"),
                    color=_color_enc if _by_fp else alt.value("#4c6ef5"),
                    tooltip=["fingerprint:N", "k:Q",
                             alt.Tooltip("rmse:Q", format=".3f")],
                )
            )
            _acc_bottom = (_cline + _rule).properties(height=170, width=300)
            _acc_chart = alt.vconcat(_acc_top, _acc_bottom).resolve_scale(
                color="shared"
            )
        else:
            _acc_chart = _acc_top

        # (b) The cliff itself, "scatter-where-a-box-would-be", now split PER
        # FINGERPRINT: a kNN prediction is the MEAN of the k nearest neighbours,
        # and each fingerprint picks *different* neighbours. So within each
        # molecule we lay out one sub-column per fingerprint (matched to the
        # colours in (a)): faint dots = that fingerprint's neighbour cloud, a
        # bold tick = its prediction. The single black diamond is the molecule's
        # TRUE activity (fingerprint-independent ground truth). Whatever the
        # fingerprint, every prediction sits far from the diamond on at least
        # one molecule - the cliff none of them resolve.
        _p1 = knn.pred_at_k(_m1, _k)
        _p2 = knn.pred_at_k(_m2, _k)
        _pred_gap = abs(_p1 - _p2)
        _true_gap = abs(_m1["true"] - _m2["true"])

        _cliff_fps = knn.cliff_fps(_m1) or {"morgan": "Morgan (ECFP4)"}
        _fp_keys = list(_cliff_fps)
        _fp_lbls = [_cliff_fps[k] for k in _fp_keys]
        _cloud_rows, _pred_rows = [], []
        for _m, _name in [(_m1, "molecule 1"), (_m2, "molecule 2")]:
            for _fk in _fp_keys:
                _lbl = _cliff_fps[_fk]
                for _a in knn.neighbor_acts_at_k(_m, _k, fp=_fk):
                    _cloud_rows.append({"mol": _name, "pKi": _a, "fingerprint": _lbl})
                _pred_rows.append(
                    {"mol": _name, "pKi": knn.pred_at_k(_m, _k, fp=_fk),
                     "fingerprint": _lbl}
                )
        _cloud = pd.DataFrame(_cloud_rows)
        _preds = pd.DataFrame(_pred_rows)
        _truth = pd.DataFrame(
            [
                {"mol": "molecule 1", "pKi": _m1["true"]},
                {"mol": "molecule 2", "pKi": _m2["true"]},
            ]
        )
        _y = alt.Y("pKi:Q", title=f"{_ep} pKi", scale=alt.Scale(zero=False))
        _fp_color = alt.Color(
            "fingerprint:N",
            scale=alt.Scale(
                domain=_fp_lbls,
                range=[_fp_palette.get(lbl, "#868e96") for lbl in _fp_lbls],
            ),
            legend=None,
        )
        _fp_xoffset = alt.XOffset("fingerprint:N", scale=alt.Scale(domain=_fp_lbls))
        # each fingerprint's neighbour cloud (jittered within its sub-column)
        _dots = (
            alt.Chart(_cloud)
            .mark_circle(size=22, opacity=0.28)
            .encode(
                x=alt.X("mol:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=_fp_xoffset,
                y=_y,
                color=_fp_color,
                tooltip=["fingerprint:N", alt.Tooltip("pKi:Q", format=".2f")],
            )
        )
        # each fingerprint's kNN prediction (bold tick where a box median sits)
        _pred_tick = (
            alt.Chart(_preds)
            .mark_tick(thickness=3, size=16)
            .encode(
                x=alt.X("mol:N", title=None),
                xOffset=_fp_xoffset,
                y=_y,
                color=_fp_color,
                tooltip=["fingerprint:N",
                         alt.Tooltip("pKi:Q", format=".2f", title="kNN pred")],
            )
        )
        # the molecule's true activity: one black diamond, fingerprint-agnostic
        _true_pt = (
            alt.Chart(_truth)
            .mark_point(shape="diamond", size=170, filled=True, color="#212529")
            .encode(
                x=alt.X("mol:N", title=None),
                y=_y,
                tooltip=[alt.Tooltip("pKi:Q", format=".2f", title="true pKi")],
            )
        )
        _cliff_chart = (_dots + _pred_tick + _true_pt).properties(
            width=280, height=230
        )

        # Honest, per-pair statement: the gap kNN opens at each k. At tiny k it
        # can occasionally match (or exceed) the true gap by copying a single
        # near-identical neighbour, but that collapses as soon as the
        # neighbourhood grows; at the accuracy-optimal k the gap is small.
        _gaps_by_k = {
            a["k"]: abs(a["pred"] - b["pred"])
            for a, b in zip(_m1["pred_by_k"], _m2["pred_by_k"])
        }
        _max_gap = max(_gaps_by_k.values())
        _argmax_k = max(_gaps_by_k, key=_gaps_by_k.get)
        _gap_at_best = _gaps_by_k.get(_best["k"], _max_gap)
        # Does any k actually resolve the cliff (reach most of the true gap)?
        _resolves = _max_gap >= 0.8 * _true_gap
        if _resolves:
            _range_clause = (
                f"the widest gap it ever opens is **{_max_gap:.2f}** — but only at "
                f"**k={_argmax_k}**, where the prediction is just *copying a single "
                f"near-identical neighbour*; the moment the neighbourhood grows the "
                f"gap collapses (down to **{_gap_at_best:.2f}** at the accuracy-optimal "
                f"**k={_best['k']}**)"
            )
        else:
            _range_clause = (
                f"the widest gap it ever opens for this pair is only "
                f"**{_max_gap:.2f}** — no neighbourhood size comes close to the true "
                f"**{_true_gap:.2f}**"
            )
        _verdict = mo.md(
            f"At **k={_k}**, kNN predicts these two molecules **{_p1:.2f}** and "
            f"**{_p2:.2f}** — a gap of just **{_pred_gap:.2f}**, though the real gap "
            f"is **{_true_gap:.2f}**. "
            f"**Slide k across its whole range:** {_range_clause}. Meanwhile global "
            f"accuracy peaks near **k={_best['k']}** (R² {_best['r2']:.2f}); the k "
            f"that fits the dataset best still can't see this cliff."
        ).callout(kind="warn")

        _view = mo.vstack(
            [
                mo.md(
                    f"**{_ep}** — {_meta['n_total']} molecules. These two panels "
                    f"summarise the *endpoint this cliff sits on* (**{_ep}**), not the "
                    f"single pair — so picking a different molecule pair that happens to "
                    f"be a cliff on the other target will swap which endpoint you're "
                    f"looking at here. The plots share the "
                    "same k axis and the same orange current-k rule. **Top:** "
                    "global held-out accuracy (R²) per fingerprint — it rises to a "
                    "peak at some k. **Bottom:** RMSE on the **cliff pairs only** "
                    f"(all {_n_cliff} in this endpoint) — it just gets worse with increasing k. "
                    "Slide k and watch the contradiction: the very "
                    "neighbourhood size that maximises average accuracy just makes predictions over cliffs worse, because it washes out the contriubtion of small structural changes."
                ),
                mo.hstack(
                    [
                        mo.vstack([mo.md(f"**{_ep}: accuracy (top) vs cliff error (bottom) vs k**"), mo.as_html(_acc_chart)]),
                        mo.vstack([
                            mo.md(
                                "**This cliff pair at k** — one coloured "
                                "sub-column per fingerprint (matched to the lines "
                                "at left): <span style='color:#868e96'>faint dots = "
                                "its neighbour cloud, tick = its prediction</span>. "
                                "<span style='color:#212529'>◆ black = true "
                                "activity</span>."
                            ),
                            mo.as_html(_cliff_chart),
                        ]),
                    ],
                    widths=[1, 1], gap=2,
                ),
                k_slider,
                _verdict,
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Binary fingerprints literally can't count

    Before we look at some of the ways that scientist try to overcome activity cliffs, one concrete failure mode is worth isolating
    because it makes "the information isn't in the encoding" problem quite visible. Many
    ADMET properties are **accumulating** — solubility, for instance: tack on
    another –CH₂– and the desolvation cost keeps climbing. To predict such a
    property you need to know *how many* hydrophobic units a molecule has. But a
    standard **binary** fingerprint only records *whether* a substructure is
    present, not **how many times**. This makes it quite hard to calculate accumulating properties from this input!

    To demonstrate, we fit several encodings on AqSolDB and compare them on two targets:

    - a **pure accumulator** we control exactly — *heavy-atom count*, which is
      by definition a sum over the molecule
    - **experimental solubility**

    The fingerprints span both families from earlier: **binary Morgan** (the usual
    fixed fingerprint), **count Morgan** (same bits, but each slot holds a count
    so a linear head can literally add them up), and
    the **CheMeleon** fingerprint. CheMeleon is interesting here: it's pre-trained for molecular
    property prediction, so you might expect it to ace an accumulation task - and it gets close but not quite there. All heads are fairly tuned (RidgeCV) and trained on a scaffold split. To check if a sufficiently deep neural net
    could somehow reconstruct accumulated properties from a fingerprint, we also try the **binary Morgan** feature with a narrow-but-deep MLP decision head.
    """)
    return


@app.cell
def _(alt, mo, pd, setup_ready):
    from fingerprints import accumulation as acc

    assert setup_ready

    _targets = acc.target_labels()

    def _collect_scores():
        _n = acc.n_molecules()
        _rows = []
        for _t in _targets:
            for _s in acc.scores(_t):
                _rows.append(
                    {
                        "target": _t,
                        "model": _s.label,
                        "family": _s.family,
                        "r2": round(_s.r2, 3),
                    }
                )
        return _n, _rows

    # Precomputed & shipped under data/ (instant); only fits live if the cache
    # is missing, e.g. during a from-scratch recompute.
    if acc.has_data():
        _n, _rows = _collect_scores()
    else:
        with mo.status.spinner(
            title="Training the models on AqSolDB (live, ~10 s)…"
        ):
            _n, _rows = _collect_scores()
    _df = pd.DataFrame(_rows)
    # keep the model order stable (as returned by the analysis)
    _model_order = list(dict.fromkeys(_df["model"]))
    # colour by ENCODING FAMILY so the point reads at a glance: red = binary
    # (can't count), green = count (can), purple = learned.
    _fam_color = {"binary": "#e03131", "count": "#2b8a3e", "learned": "#7048e8"}
    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("target:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=_model_order),
            y=alt.Y("r2:Q", title="held-out R²", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "model:N",
                sort=_model_order,
                scale=alt.Scale(
                    domain=_model_order,
                    range=[
                        _fam_color[
                            _df.loc[_df["model"] == m, "family"].iloc[0]
                        ]
                        for m in _model_order
                    ],
                ),
                legend=alt.Legend(title=None, orient="bottom", columns=1),
            ),
            tooltip=["target:N", "model:N", "r2:Q"],
        )
        .properties(height=300, width=340)
    )

    def _r2(target, key):
        for _s in acc.scores(target):
            if _s.key == key:
                return _s.r2
        return float("nan")

    _acc = "heavy-atom count"
    _bl, _cl, _mlp = (
        _r2(_acc, "binary_linear"),
        _r2(_acc, "count_linear"),
        _r2(_acc, "binary_mlp"),
    )
    _che = _r2(_acc, "chemeleon_linear")
    _che_line = (
        f" The **learned** CheMeleon fingerprint does markedly better — **R² "
        f"{_che:.2f}** — well above the binary fingerprint. Even though it "
        f"*mean*-pools over atoms (which divides out molecule size), its "
        f"pretrained per-atom features carry enough size-correlated "
        f"signal to reconstruct much of the count. A representation *learned* "
        f"from data recovers a lot of what a fixed binarised encoding threw "
        f"away. Given that CheMeleon was trained for calculated property prediction, it is not surprising that it's good at this task - but it's not perfect."
    )
    _verdict = mo.md(
        f"On the **pure accumulator**, the story is blatant. A **count** "
        f"fingerprint + a plain linear model scores **R² {_cl:.2f}** — it just "
        f"sums the bits, which *is* the target. The **binary** fingerprint + the "
        f"same linear model is capped at **R² {_bl:.2f}**: once you binarise, you "
        f"can't tell one –CH₂– from six. Handing the binary "
        f"fingerprint to a **narrow-but-deep neural net** moves it to "
        f"**R² {_mlp:.2f}** — depth reshuffles which bits co-occur but can't "
        f"recover multiplicity the encoding never stored."
        f"{_che_line}\n\n"
        f"The lesson: **whether you can accumulate is decided by the encoding "
        f"(and its pooling) before any model runs** — a count vector or a learned "
        f"representation can, a fixed binary presence-vector fundamentally can't. "
        f"Real solubility is only *partly* accumulation (crystal packing, "
        f"H-bonding and charge matter too), so on the right the gaps shrink. The "
        f"controlled target exposes the mechanism cleanly, but the real data helps remind us reality (un?)fortunately is not so simple."
    ).callout(kind="info")

    mo.vstack(
        [
            mo.md(
                f"**Encodings vs. two targets** — trained on "
                f"**{_n:,}** AqSolDB molecules (scaffold split). "
                f"<span style='color:#e03131'>■ binary</span> · "
                f"<span style='color:#2b8a3e'>■ count</span> · "
                f"<span style='color:#7048e8'>■ learned</span>:"
            ),
            mo.as_html(_chart),
            _verdict,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Follow-up: does *counting* help every fingerprint — and every property?

    The headline used Morgan. But binary-vs-count is a knob on **every** classical
    RDKit fingerprint, so let's turn it on all of them at once — and, crucially,
    on a **third kind of target**. So far every property has been ADMET-flavoured
    (accumulating). Binding potency is different: it's molecular **recognition**
    — does the molecule present the right shape to the pocket? — where the *count*
    of a feature should matter far less than its *presence*.

    The grid below is the same fair linear model (RidgeCV, scaffold split) run for
    **four fingerprints × {binary, count} × three targets**: our pure accumulator
    (heavy-atom count), aqueous solubility, and **Dopamine D3 binding pKi**
    (MoleculeACE / ChEMBL). Watch where switching to counts helps — and where it
    quietly backfires.
    """)
    return


@app.cell
def _(alt, mo, pd, setup_ready):
    from fingerprints import accumulation as acc2

    assert setup_ready

    def _collect_survey():
        return (
            acc2.survey(),
            acc2.survey_targets(),
            acc2.survey_fingerprints(),
            acc2.survey_n(),
        )

    # Precomputed & shipped under data/ (instant); only fits live if the cache
    # is missing, e.g. during a from-scratch recompute.
    if acc2.has_data():
        _cells, _targets, _fps, _counts = _collect_survey()
    else:
        with mo.status.spinner(
            title="Fitting 4 fingerprints × binary/count × 3 targets (live, ~40 s)…"
        ):
            _cells, _targets, _fps, _counts = _collect_survey()
    _df = pd.DataFrame(
        [
            {
                "fingerprint": c.fingerprint,
                "encoding": c.encoding,
                "target": c.target,
                "r2": round(c.r2, 3),
            }
            for c in _cells
        ]
    )

    # Grouped bars: one facet per target, binary vs count side by side per fp.
    _bars = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("encoding:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("r2:Q", title="held-out R²", scale=alt.Scale(domain=[-0.2, 1.0])),
            color=alt.Color(
                "encoding:N",
                scale=alt.Scale(
                    domain=["binary", "count"], range=["#e03131", "#2b8a3e"]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["fingerprint:N", "encoding:N", "target:N", "r2:Q"],
        )
        .properties(width=95, height=150)
        .facet(
            column=alt.Column("fingerprint:N", sort=_fps, title=None),
            row=alt.Row("target:N", sort=_targets, title=None),
        )
        .resolve_scale(y="shared")
    )

    # Quantify the flip: mean count-minus-binary delta per target.
    def _mean_delta(target):
        ds = [d for _, d in acc2.count_minus_binary(target)]
        return sum(ds) / len(ds) if ds else float("nan")

    _d_hac = _mean_delta("heavy-atom count")
    _d_sol = _mean_delta("aqueous solubility")
    _d_bind = _mean_delta(_targets[2])
    _verdict = mo.md(
        f"I'd suggest to read down each column. On the **pure accumulator** target "
        f"(top row) switching binary→count *raises* R² for every fingerprint "
        f"(mean **{_d_hac:+.2f}**) — counting is exactly what an accumulated total "
        f"needs, and Morgan/topological (which at most barely encode multiplicity when "
        f"binarised) gain the most.\n\n"
        f"On the **real** properties the sign flips. Counts give **no** benefit on "
        f"solubility (mean **{_d_sol:+.2f}**) and actively **hurt** binding "
        f"(mean **{_d_bind:+.2f}**, worse for every fingerprint). Two possible reasons"
        f": real endpoints are only partly accumulation, and binding is "
        f"**recognition** — whether the right pharmacophore is *present* drives "
        f"potency, while *how many copies* of a fragment a molecule has is mostly "
        f"noise that a count vector lets the model overfit. So the encoding that "
        f"is provably best on the controlled accumulator is the *wrong* choice on "
        f"real properties — there is no "
        f"universally best fingerprint - only the right one for the given task, and you can't usually know which it is in advance."
    ).callout(kind="info")

    mo.vstack(
        [
            mo.md(
                f"**binary vs count, every classical fingerprint, three targets** "
                f"— accumulator + solubility on **{_counts['accumulator']:,}** "
                f"AqSolDB molecules, binding on **{_counts['binding']:,}** "
                f"Dopamine-D3 molecules (all scaffold-split). "
                f"<span style='color:#e03131'>■ binary</span> · "
                f"<span style='color:#2b8a3e'>■ count</span>:"
            ),
            mo.as_html(_bars),
            _verdict,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## So can *anything* see the cliff?

    We've shown the hard part — that *no* structure-similarity model, fixed or
    learned, can see cliffs reliably, because a cliff is exactly where "similar
    structure → similar number" mapping breaks. The required information to detect these cliffs without losing accuracy on most non-cliff data simply isn't in the
    encoding.

    Two closing questions. First, the
    static CheMeleon fingerprint nearly matched *count* fingerprints on some admet properties
    properties — so does it finally **beat classical fingerprints on the
    cliffs**? Second, a brief foray into a real solution: **compute something the 2D graph
    threw away** — the actual 3D interaction in the pocket.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Does the *learned* fingerprint crack the cliffs?

    Earlier, CheMeleon nearly matched count fingerprints on smooth properties.
    So here's a more intentional test on the hard case: for each binding endpoint, train a
    model on **ECFP (fixed)** vs **CheMeleon** fingerprints — and, because a smarter
    *head* is the obvious next move in the modeling, we try three: a **linear** read-out, a
    **kNN** (similarity) read-out, and a **nonlinear MLP**. Then we measure RMSE
    on the molecules MoleculeACE flags as **activity-cliff members** (very
    similar to a neighbour, yet ≥10× different in potency) versus everyone else.

    If the learned fingerprint or a fancier head genuinely resolved cliffs,
    you'd see its cliff bar drop. Watch what actually happens.
    """)
    return


@app.cell
def _(alt, mo, pd, setup_ready):
    from fingerprints import learned_cliffs as lc

    assert setup_ready

    # Precomputed and shipped under data/ (loads instantly); only trains live
    # (~50 s) if the cache is missing, e.g. during a from-scratch recompute.
    if lc.has_data():
        _res = lc.results()
    else:
        with mo.status.spinner(
            title="Training {ECFP, CheMeleon} × {linear, kNN, MLP} on 4 binding "
            "endpoints (live, ~50 s)…"
        ):
            _res = lc.results()
    _rows = []
    for _r in _res:
        _rows.append(
            {
                "endpoint": _r.endpoint,
                "fingerprint": _r.fingerprint,
                "head": _r.head,
                "group": f"{_r.fingerprint} · {_r.head}",
                "cliff RMSE": round(_r.cliff_rmse, 3),
                "non-cliff RMSE": round(_r.noncliff_rmse, 3),
            }
        )
    _df = pd.DataFrame(_rows)
    # Mean cliff RMSE across endpoints for each (fingerprint, head).
    _summary = (
        _df.groupby(["fingerprint", "head"], as_index=False)["cliff RMSE"]
        .mean()
        .round(2)
    )
    _summary["group"] = _summary["fingerprint"] + " · " + _summary["head"]
    _head_order = lc.heads()
    _fp_order = lc.fingerprints()
    _group_order = [f"{fp} · {h}" for fp in _fp_order for h in _head_order]
    _chart = (
        alt.Chart(_summary)
        .mark_bar()
        .encode(
            x=alt.X("group:N", sort=_group_order, title=None,
                    axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("cliff RMSE:Q",
                    title="mean cliff-pair RMSE (log units, lower=better)",
                    scale=alt.Scale(domain=[0, 1.2])),
            color=alt.Color(
                "fingerprint:N",
                scale=alt.Scale(
                    domain=_fp_order, range=["#4c6ef5", "#7048e8"]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["fingerprint:N", "head:N",
                     alt.Tooltip("cliff RMSE:Q", format=".2f")],
        )
        .properties(height=260, width=360)
    )
    _lo = _summary["cliff RMSE"].min()
    _hi = _summary["cliff RMSE"].max()
    _verdict = mo.md(
        f"**No combination cracks it.** Across all six fingerprint×head combos, "
        f"mean cliff RMSE stays in a tight **{_lo:.2f}–{_hi:.2f}** log-unit band. "
        f"The learned fingerprint doesn't beat the classical one on cliffs, and "
        f"neither a similarity read-out (kNN) nor a nonlinear net (MLP) helps — "
        f"because a head only ever sees the representation, and two cliff "
        f"molecules land at nearly the *same point* in any 2D-structure "
        f"embedding. You can't un-collapse them downstream.\n\n"
        f"**The important caveat — and the real point.** This tests CheMeleon as "
        f"a *frozen* fingerprint. Fine-tune the whole message-passing network "
        f"**end-to-end** on a target and it *can* score well on cliff benchmarks "
        f"like MoleculeACE — but that's the tell, not a refutation: you're no "
        f"longer using a fixed representation, you're *learning a new one per "
        f"task*, letting it carve apart molecules a generic encoding collapses. "
        f"Deep learning didn't repeal the limitation of static fingerprints; it "
        f"made the representation itself trainable. As a **static** fingerprint, "
        f"learned or hand-designed, the cliff stays invisible."
    ).callout(kind="info")
    mo.vstack(
        [
            mo.md(
                "**Cliff-pair RMSE by fingerprint and head** (mean over the four "
                "binding endpoints; MoleculeACE train/test split):"
            ),
            mo.as_html(_chart),
            _verdict,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Leave fingerprints behind and look in the pocket

    If the signal isn't in *any* 2D-structure fingerprint, we need to try computation that's more information-rich, but harder to pull off. In this case, what we're going to look at is the **3D interactions** between molecule and protein in the binding pocket, not the
    2D graph. Below, we fold each ligand into the pocket with **Boltz**, detect
    its contacts with **PLIP**, and read an interaction fingerprint off the
    pose.

    Let's be honest about what this is and isn't:

    - It's a **clue in a direction**, not a general fix. For the μ-opioid pair
      it points at a plausible cause (a single extra H-bond); for other datasets it is even less clear if the Boltz prediction can explain the experimental data.
    - These are **predicted** poses — binding-mode *hypotheses*, not
      experimental structures — for a **handful** of curated pairs. A
      qualitative contrast, not a benchmark or validated hypothesis.
    - Making this a real method would need **more data** and **local context in
      both spaces at once**: nearby chemical structure *and* nearby protein
      structure. Activity cliffs stay an open, actively-researched problem.

    With those caveats in advance, here's the demonstration.
    """)
    return


@app.cell
def _(mo, selectors):
    # Synced selector mirror: switch target pair / molecule pair right
    # here without scrolling back to the top (same global elements).
    mo.vstack([mo.md('**Pick the target pair / molecule pair for these 3D poses:**'), selectors()])
    return


@app.cell
def _(ctx, cv, get_cliff_idx, get_pair_key, mo):
    from fingerprints import pose_view as pv
    from fingerprints.complex_viewer import ComplexViewer

    _tp = ctx.by_key()[get_pair_key()]
    _cliff = _tp.cliffs[get_cliff_idx()]
    _pair_key, _idx = _tp.key, get_cliff_idx()

    if not pv.has_poses(_pair_key, _idx):
        _view = mo.md(
            f"*No precomputed 3D poses for this cliff yet ({_tp.target_a} vs "
            f"{_tp.target_b}, pair {_idx + 1}). Poses were folded offline with "
            "Boltz-2 for a subset of cliffs — pick one of those, or run "
            "`python -m fingerprints.rebuild_poses` to add this one.*"
        ).callout(kind="info")
    else:
        _poses = pv.load_all(_pair_key, _idx)

        def _anchor_resi(pose):
            for rn, seq, _d in pv.pocket_residues(pose.cif_text):
                if rn == "ASP":
                    return seq
            return ""

        def _potency_word(pki):
            if pki >= 9.0:
                return "very potent", "#2b8a3e"
            if pki >= 7.5:
                return "potent", "#40a060"
            if pki >= 6.0:
                return "moderate", "#e8820c"
            return "weak", "#e03131"

        def _panel(mol_id, target, counterpart_mol, pki):
            p = _poses[f"{mol_id}_{target}"]
            changed = pv.changed_atom_names(p, _poses[f"{counterpart_mol}_{target}"])
            v = ComplexViewer(
                structure=p.cif_text,
                format="cif",
                highlight_resi=_anchor_resi(p),
                highlight_atoms=changed,
                interactions=pv.load_interactions(p),
                height=320,
            )
            _word, _color = _potency_word(pki)
            _mol_name = "molecule 1" if mol_id == "mol1" else "molecule 2"
            badge = mo.md(
                f"<div style='text-align:center'>"
                f"<b>{_mol_name}</b> — pKi <b>{pki}</b> "
                f"<span style='color:{_color}'><b>({_word})</b></span></div>"
            )
            return mo.vstack([badge, mo.ui.anywidget(v)])

        _ta, _tb = _tp.target_a, _tp.target_b

        # Summary table first: molecule x target grid so the cliff is obvious
        # before looking at any 3D. The cliff target's cells are boxed.
        def _cell(pki, is_cliff_target):
            _word, _color = _potency_word(pki)
            _border = "2px solid #e03131" if is_cliff_target else "1px solid #dee2e6"
            return (
                f"<td style='border:{_border};padding:6px 14px;text-align:center'>"
                f"pKi <b>{pki}</b><br>"
                f"<span style='color:{_color};font-size:12px'>{_word}</span></td>"
            )

        _cliff_on = _cliff.cliff_on
        _table = (
            "<table style='border-collapse:collapse;margin:0 auto'>"
            f"<tr><th></th>"
            f"<th style='padding:4px 14px'>{_ta}</th>"
            f"<th style='padding:4px 14px'>{_tb}</th></tr>"
            f"<tr><td style='padding:4px 10px;text-align:right'><b>molecule 1</b></td>"
            + _cell(_cliff.pki_1_a, _cliff_on == _ta)
            + _cell(_cliff.pki_1_b, _cliff_on == _tb)
            + "</tr>"
            f"<tr><td style='padding:4px 10px;text-align:right'><b>molecule 2</b><br>"
            f"<span style='font-size:11px;color:#868e96'>({_cliff.change})</span></td>"
            + _cell(_cliff.pki_2_a, _cliff_on == _ta)
            + _cell(_cliff.pki_2_b, _cliff_on == _tb)
            + "</tr></table>"
        )
        _summary = mo.vstack(
            [
                mo.Html(_table),
                mo.md(
                    f"The red-boxed column is **{_cliff_on}**, where the one-atom "
                    f"change is a **{cv.fold_change(max(_cliff.delta_a, _cliff.delta_b))} "
                    f"cliff**. On the other target it barely moves. Same two "
                    "molecules, both columns — the poses below are grouped by "
                    "target so you can compare the two molecules in the *same* "
                    "pocket side by side."
                ),
            ]
        )

        _view = mo.vstack(
            [
                _summary,
                mo.md(f"#### Both molecules in {_ta}"),
                mo.hstack(
                    [
                        _panel("mol1", _ta, "mol2", _cliff.pki_1_a),
                        _panel("mol2", _ta, "mol1", _cliff.pki_2_a),
                    ],
                    widths=[1, 1],
                    gap=1,
                ),
                mo.md(f"#### Both molecules in {_tb}"),
                mo.hstack(
                    [
                        _panel("mol1", _tb, "mol2", _cliff.pki_1_b),
                        _panel("mol2", _tb, "mol1", _cliff.pki_2_b),
                    ],
                    widths=[1, 1],
                    gap=1,
                ),
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(ctx, get_cliff_idx, get_pair_key, mo):
    from fingerprints import pose_view as pv2

    # The interaction fingerprint: encode each pose by the contacts it makes
    # (residue x interaction-type bits) and lay the four poses side by side.
    # Unlike the 2D fingerprints earlier in the notebook, these bits are read
    # off the binding event itself — the data telling us what to encode.
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    if not pv2.has_poses(_tp.key, _idx):
        _view = mo.md("")
    else:
        _poses = pv2.load_all(_tp.key, _idx)
        _cliff = _tp.cliffs[_idx]

        def _potency_word(pki):
            if pki >= 9.0:
                return "very potent"
            if pki >= 7.5:
                return "potent"
            if pki >= 6.0:
                return "moderate"
            return "weak"

        _keys = [
            f"mol1_{_tp.target_a}", f"mol2_{_tp.target_a}",
            f"mol1_{_tp.target_b}", f"mol2_{_tp.target_b}",
        ]
        _col_pki = [_cliff.pki_1_a, _cliff.pki_2_a, _cliff.pki_1_b, _cliff.pki_2_b]
        _col_labels = [
            f"mol 1 · {_tp.target_a}", f"mol 2 · {_tp.target_a}",
            f"mol 1 · {_tp.target_b}", f"mol 2 · {_tp.target_b}",
        ]
        _col_labels = [
            f"{_lab}  —  {_potency_word(_p)} (pKi {_p})"
            for _lab, _p in zip(_col_labels, _col_pki)
        ]
        _bits, _fps = pv2.aligned_interaction_fingerprints(_poses, _keys)
        _type_label = dict(pv2.INTERACTION_TYPES)
        _type_color = {
            "saltbridge": "#e0a800", "hbond": "#4dabf7", "pistack": "#20c997",
            "pication": "#e64980", "hydrophobic": "#adb5bd",
        }

        # Hand-built HTML grid: rows = interaction bits, cols = the 4 poses.
        _html = [
            "<table style='border-collapse:collapse;font-size:12px'>",
            "<tr><th style='text-align:left;padding:2px 8px'>interaction bit</th>"
            + "".join(
                f"<th style='padding:2px 6px;writing-mode:vertical-rl;"
                f"transform:rotate(180deg)'>{c}</th>"
                for c in _col_labels
            )
            + "</tr>",
        ]
        for _res, _typ in _bits:
            _dot = _type_color.get(_typ, "#868e96")
            _label = (
                f"<span style='color:{_dot}'>●</span> {_res} "
                f"<span style='color:#868e96'>({_type_label.get(_typ, _typ)})</span>"
            )
            _cells = ""
            for _k in _keys:
                _on = _fps[_k].get((_res, _typ), 0)
                _bg = _dot if _on else "#f1f3f5"
                _cells += (
                    f"<td style='padding:0;border:1px solid #fff;width:70px;"
                    f"height:20px;background:{_bg}'></td>"
                )
            _html.append(
                f"<tr><td style='padding:2px 8px'>{_label}</td>{_cells}</tr>"
            )
        _html.append("</table>")

        _view = mo.vstack(
            [
                mo.md(
                    "**An interaction fingerprint, read off the pose.** Each row is "
                    "a contact the ligand makes; a filled cell means that pose has "
                    "it. Same idea as the 2D fingerprints from earlier — but here "
                    "the *data* (the binding pose) decides the bits, instead of us "
                    "imposing them."
                ),
                mo.Html("".join(_html)),
                mo.md(
                    "*Contacts via PLIP on the predicted poses. The salt bridge to "
                    "the conserved aspartate is the constant anchor; the cliff shows "
                    "up only as a subtle reshuffle of weaker H-bond / hydrophobic "
                    "bits — even this data-derived fingerprint doesn't obviously "
                    "explain the potency gap.*"
                ),
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### About this notebook

    **AI use:** This notebook was built in
    a pairing session with an AI coding assistant: it helped scaffold the marimo
    cells, the custom `ComplexViewer` anywidget, and the analysis scripts, work through experiments, and write early drafts of the prose. Every chemical claim, data source, and result was
    reviewed; the majority of prose was overwritten by Raymond Gasper.

    All chemical structure handling runs through **RDKit**. Binding data are from **MoleculeACE**
    (curated ChEMBL bioactivities with published activity-cliff labels); the
    **ADMET** data (aqueous solubility from **AqSolDB**, lipophilicity from
    **AstraZeneca**) come from **Therapeutics Data Commons**, where we strip
    salts, keep the largest organic fragment, and de-duplicate by canonical
    parent SMILES before analysis. Every train/test split — for the learned
    model and the ADMET census alike — is a **Bemis–Murcko scaffold split** - I'm aware that this has limitations and may not be the most rigorous way to do a chemical dataset splitting, but didn't want to get overly complex just for the demonstrations. 3D
    complexes are **Boltz-2** predictions — framed throughout as *hypotheses*,
    not experimental structures — and protein–ligand interactions are detected
    with **PLIP**.

    **Reproducibility.** Heavy compute (folding, interaction detection, model
    training) runs offline and is cached in `data/`; the notebook only reads
    those caches, so it stays instant and deterministic. Fingerprint code and
    analyses live in this repo — see `README.md`.

    **Credits.** RDKit · chemprop (D-MPNN) · CheMeleon · Boltz-2 · PLIP ·
    MoleculeACE · Therapeutics Data Commons (AqSolDB, AstraZeneca) ·
    3Dmol.js · Altair · marimo. Thanks to OpenADMET and the
    marimo team for the competition.
    """)
    return


if __name__ == "__main__":
    app.run()
