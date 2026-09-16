import marimo

__generated_with = "0.24.0"
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
    # Molecular Fingerprints, Made Tangible

    *A molab Notebook Competition entry.*
    """)
    return


@app.cell
def _(mo):
    from fingerprints import gallery as gal

    mol_choice = mo.ui.dropdown(
        options=[g.label for g in gal.GALLERY],
        value=gal.default_label(),
        label="Choose a molecule",
    )
    custom_smiles = mo.ui.text(
        value="",
        label="…or type your own SMILES (overrides the dropdown)",
        full_width=True,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    )
    return custom_smiles, gal, mol_choice


@app.cell
def _(custom_smiles, gal, mo, mol_choice):
    from fingerprints import maccs_explorer as mx

    # Resolve the single upstream molecule: custom SMILES wins if provided,
    # otherwise fall back to the gallery pick. Everything downstream depends
    # only on `current_mol` / `current_label`.
    _typed = custom_smiles.value.strip()
    if _typed:
        _mol = mx.mol_from_smiles(_typed)
        _source_label = "custom SMILES"
        _picked_note = ""
    else:
        _entry = gal.by_label()[mol_choice.value]
        _mol = mx.mol_from_smiles(_entry.smiles)
        _source_label = _entry.label
        _picked_note = _entry.note

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
    bit_slider
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
                _header,
                mo.hstack([_query_panel, _mol_panel], justify="start", gap=2, widths=[1, 2]),
                mo.md("**Where this bit sits in the whole 166-bit fingerprint:**"),
                mo.Html(_strip),
                _strip_legend,
            ]
        )
    else:
        _view = mo.md("*Select a valid molecule to explore its MACCS bits.*")
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
    morgan_slider
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
            "are *sparse*. Unlike MACCS, an off bit here carries no meaning of its "
            "own (it just means no atom environment hashed to that slot), so the "
            "scrubber skips straight between the on bits."
        )
        _view = mo.vstack(
            [mo.md("**The whole 2048-bit fingerprint:**"), mo.Html(_strip), _legend, _note]
        )
    else:
        _view = mo.md("")
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
                    "*In a 2048-bit fingerprint these would (almost always) land "
                    "on separate bits — that extra length is what buys the "
                    "resolution.*"
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
                    f"This molecule has **{_distinct} distinct atom environments**. "
                    "The rate falls off fast — which is why **2048 bits** is a common "
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
                    collision_slider,
                    collision_card,
                    collision_curve_view,
                ]
            )
        }
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
                    slider,
                    mo.hstack([mo.Html(svg), card], justify="start", gap=2, widths=[3, 2]),
                    mo.md("**Where this bit sits in the full 2048-bit vector:**"),
                    mo.Html(strip),
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
    mo.accordion(
        {
            "🧰 Aside: the rest of the RDKit toolbox (topological, atom-pair, torsion)": mo.vstack(
                [
                    mo.md(
                        "Beyond MACCS's checklist and Morgan's circular environments, "
                        "RDKit ships several more classical fingerprints. They each "
                        "encode a different notion of structure — paths, atom pairs at "
                        "a distance, torsions — but share Morgan's hashing machinery. "
                        "Worth knowing they exist; not central to the story."
                    ),
                    tabbed_fps,
                ]
            )
        }
    )
    return


@app.cell
def _(mo):
    from fingerprints.data import context_cliffs as ctx

    target_pair_choice = mo.ui.dropdown(
        options={
            f"{tp.target_a} vs {tp.target_b}": tp.key for tp in ctx.TARGET_PAIRS
        },
        value=f"{ctx.TARGET_PAIRS[0].target_a} vs {ctx.TARGET_PAIRS[0].target_b}",
        label="Target pair",
    )
    return ctx, target_pair_choice


@app.cell
def _(ctx, mo, target_pair_choice):
    _tp = ctx.by_key()[target_pair_choice.value]
    # Label each curated pair by its plain-English change + which target it's a
    # cliff on, so the dropdown itself previews the story.
    _opts = {}
    for _i, _c in enumerate(_tp.cliffs):
        _opts[f"{_c.change}  —  cliff on {_c.cliff_on}"] = _i
    cliff_choice = mo.ui.dropdown(
        options=_opts, value=next(iter(_opts)), label="Molecule pair"
    )
    mo.vstack([mo.md(f"*{_tp.blurb}*"), mo.hstack([target_pair_choice, cliff_choice], justify="start", gap=2)])
    return (cliff_choice,)


@app.cell
def _(cliff_choice, ctx, cv, mo, target_pair_choice):
    _tp = ctx.by_key()[target_pair_choice.value]
    _pair = _tp.cliffs[cliff_choice.value]
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
        f"**The change:** {_pair.change}.  \nThe orange atoms are all that differ "
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
    mo.vstack([_structures, _change, _endpoints])
    return


@app.cell
def _(alt, cliff_choice, ctx, cv, mo, pd, target_pair_choice):
    # The reveal: every fingerprint scores this pair as fairly similar - one
    # number, blind to which endpoint it's being applied to.
    _tp = ctx.by_key()[target_pair_choice.value]
    _pair = _tp.cliffs[cliff_choice.value]
    _scores = cv.fingerprint_scores(_pair)
    _df = pd.DataFrame(
        {
            "fingerprint": [s.label for s in _scores],
            "similarity": [round(s.similarity, 3) for s in _scores],
        }
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
        .properties(height=220)
    )
    _lo = min(s.similarity for s in _scores)
    _hi = max(s.similarity for s in _scores)
    _punchline = mo.md(
        f"Every fingerprint calls this pair **similar** (similarity "
        f"{_lo:.2f}–{_hi:.2f}) — they only see the small structural change. "
        f"That verdict is **right for {_pair.flat_on}** (where the pair really is "
        f"flat) and **badly wrong for {_pair.cliff_on}** (where it's a cliff). "
        "One structural similarity, two opposite biological realities — the "
        "fingerprint cannot tell which target you mean."
    ).callout(kind="warn")
    mo.vstack(
        [
            mo.md("**How similar each fingerprint thinks this pair is:**"),
            mo.as_html(_chart),
            _punchline,
        ]
    )
    return


@app.cell
def _(cliff_choice, ctx, cv, mo, target_pair_choice):
    from fingerprints import pose_view as pv
    from fingerprints.complex_viewer import ComplexViewer

    _tp = ctx.by_key()[target_pair_choice.value]
    _cliff = _tp.cliffs[cliff_choice.value]
    _pair_key, _idx = _tp.key, cliff_choice.value

    if not pv.has_poses(_pair_key, _idx):
        _view = mo.md(
            f"*No precomputed 3D poses for this cliff yet ({_tp.target_a} vs "
            f"{_tp.target_b}, pair {_idx + 1}). Poses were folded offline with "
            "Boltz-2 for a subset of cliffs — pick one of those, or run "
            "`scripts/boltz_fold_cliffs.py` to add this one.*"
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
    _view
    return


@app.cell
def _(cliff_choice, ctx, mo, target_pair_choice):
    from fingerprints import pose_view as pv2

    # The interaction fingerprint: encode each pose by the contacts it makes
    # (residue x interaction-type bits) and lay the four poses side by side.
    # Unlike the 2D fingerprints earlier in the notebook, these bits are read
    # off the binding event itself — the data telling us what to encode.
    _tp = ctx.by_key()[target_pair_choice.value]
    _idx = cliff_choice.value
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
    _view
    return


@app.cell
def _(mo):
    alpha_knob = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.25,
        value=0.5,
        label="α — loss weight toward the first endpoint (← second · first →)",
        show_value=True,
        full_width=True,
    )
    alpha_knob
    return (alpha_knob,)


@app.cell
def _(alpha_knob, alt, cliff_choice, ctx, mo, pd, target_pair_choice):
    from fingerprints import learned_fp_view as lfv

    # Follow the target-pair picker from Section 4; fall back to any trained grid
    # if the selected pair hasn't been trained yet.
    _sel = ctx.by_key()[target_pair_choice.value].key
    if lfv.has_grid(_sel):
        _pair = _sel
    elif lfv.available_pairs():
        _pair = lfv.available_pairs()[0]
    else:
        _pair = None
    if _pair is None:
        _view = mo.md(
            "*No trained α-grid found. Run "
            "`uv run python scripts/train_alpha_grid.py` to generate it.*"
        ).callout(kind="warn")
    else:
        _g = lfv.load_grid(_pair)
        _ta, _tb = _g["target_a"], _g["target_b"]
        _res = {r["alpha"]: r for r in _g["results"]}
        _cur = _res.get(alpha_knob.value, _g["results"][len(_g["results"]) // 2])
        _note = (
            ""
            if _pair == _sel
            else f"  \n*(showing {_ta} vs {_tb} — the picked pair isn't trained yet)*"
        )

        # RMSE per endpoint for a grid row. Prefer the cached mean-over-seeds;
        # if an older cache lacks it, fall back to computing from the seed-0
        # scatter so the notebook never KeyErrors while the grid is regenerating.
        def _rmse_of(row, key):
            _mean = row.get(f"rmse_{key}_mean")
            if _mean is not None:
                return _mean
            sc = (row.get("scatter") or {}).get(key)
            if not sc:
                return float("nan")
            import math

            _pairs = [
                (a, p)
                for a, p in zip(sc["actual"], sc["pred"])
                if a is not None and p is not None and a == a and p == p
            ]
            if len(_pairs) < 5:
                return float("nan")
            return math.sqrt(
                sum((a - p) ** 2 for a, p in _pairs) / len(_pairs)
            )

        # Two big RMSE readouts for the current alpha (lower is better, pKi
        # units). alpha weights the FIRST endpoint (task A); 1-alpha the second.
        def _card(target, rmse):
            if rmse != rmse:  # NaN
                return mo.md(f"#### {target}\n\ntest RMSE = **n/a**").callout(
                    kind="neutral"
                )
            _kind = "success" if rmse < 0.8 else ("danger" if rmse > 1.2 else "neutral")
            _verdict = "learns it" if rmse < 0.8 else (
                "fails" if rmse > 1.2 else "partial"
            )
            return mo.md(
                f"#### {target}\n\ntest RMSE = **{rmse:.2f}** pKi — {_verdict}"
            ).callout(kind=_kind)

        _readout = mo.hstack(
            [
                _card(f"{_ta}  (α = {alpha_knob.value})", _rmse_of(_cur, "a")),
                _card(f"{_tb}  (1−α = {round(1 - alpha_knob.value, 2)})", _rmse_of(_cur, "b")),
            ],
            widths=[1, 1],
            gap=2,
        )

        # Trade-off curve: RMSE on each endpoint across the whole alpha grid,
        # with the current alpha marked. Lower is better.
        _rows = []
        for r in _g["results"]:
            _rows.append({"alpha": r["alpha"], "RMSE": _rmse_of(r, "a"), "endpoint": _ta})
            _rows.append({"alpha": r["alpha"], "RMSE": _rmse_of(r, "b"), "endpoint": _tb})
        _df = pd.DataFrame(_rows).dropna()
        _ymax = float(_df["RMSE"].max()) * 1.1 if not _df.empty else 2.0
        _line = (
            alt.Chart(_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("alpha:Q", title="α (loss weight toward the first endpoint)"),
                y=alt.Y(
                    "RMSE:Q",
                    title="test RMSE (pKi) — lower is better",
                    scale=alt.Scale(domain=[0, _ymax]),
                ),
                color=alt.Color("endpoint:N", title=None),
            )
            .properties(height=240, width=440)
        )
        _rule = (
            alt.Chart(pd.DataFrame({"alpha": [alpha_knob.value]}))
            .mark_rule(color="#868e96", strokeDash=[4, 4])
            .encode(x="alpha:Q")
        )
        _tradeoff = mo.as_html(_line + _rule)

        # Endpoint-vs-endpoint view: plot endpoint A pKi (x) against endpoint B
        # pKi (y). Each molecule shows a MEASURED mark (filled circle) and a
        # PREDICTED mark (hollow diamond) joined by a line; the picked cliff pair
        # is highlighted while everything else fades back. This makes the cliff
        # tangible: two molecules that sit almost on top of each other in one
        # axis but far apart in the other — and whether the model can follow.
        from rdkit import Chem as _Chem

        def _canon(smi):
            _m = _Chem.MolFromSmiles(smi)
            return _Chem.MolToSmiles(_m) if _m else None

        def _endpoint_vs_endpoint():
            sc = _cur.get("scatter") or {}
            if "smiles" not in sc:
                return mo.md(
                    "*Per-molecule endpoint–endpoint data isn't in this cache yet — "
                    "re-run `scripts/train_alpha_grid.py`.*"
                ).callout(kind="info")

            # SMILES of the two molecules in the currently-picked cliff, so we can
            # spotlight them among all the faded background molecules. Only do this
            # when the grid we loaded matches the picked pair (otherwise the cliff
            # SMILES won't be among this grid's molecules).
            _tp = ctx.by_key()[target_pair_choice.value]
            _cl = (
                _tp.cliffs[cliff_choice.value]
                if _pair == _sel and cliff_choice.value < len(_tp.cliffs)
                else None
            )
            _hi_map = {}
            if _cl is not None:
                _c1, _c2 = _canon(_cl.smiles_1), _canon(_cl.smiles_2)
                if _c1:
                    _hi_map[_c1] = "molecule 1"
                if _c2:
                    _hi_map[_c2] = "molecule 2"

            _rows = []
            for _i, _smi in enumerate(sc["smiles"]):
                _cs = _canon(_smi)
                _grp = _hi_map.get(_cs, "other molecules")
                _aa, _ab = sc["actual_a"][_i], sc["actual_b"][_i]
                _pa, _pb = sc["pred_a"][_i], sc["pred_b"][_i]
                # Measured mark only if BOTH endpoints are labeled.
                if _aa is not None and _ab is not None and _aa == _aa and _ab == _ab:
                    _rows.append({"x": _aa, "y": _ab, "src": "measured", "grp": _grp})
                if _pa is not None and _pb is not None and _pa == _pa and _pb == _pb:
                    _rows.append({"x": _pa, "y": _pb, "src": "predicted", "grp": _grp})
            _pts = pd.DataFrame(_rows)
            if _pts.empty:
                return mo.md("*No molecules with labels to plot.*")

            _bg = _pts[_pts["grp"] == "other molecules"]
            _fg = _pts[_pts["grp"] != "other molecules"]

            _base = alt.Chart(_bg)
            _x = alt.X(f"x:Q", title=f"{_ta} pKi")
            _y = alt.Y(f"y:Q", title=f"{_tb} pKi")
            _shape = alt.Shape(
                "src:N",
                title=None,
                scale=alt.Scale(
                    domain=["measured", "predicted"], range=["circle", "diamond"]
                ),
            )
            # Faded background cloud.
            _cloud = _base.mark_point(opacity=0.12, size=35, color="#868e96").encode(
                x=_x, y=_y, shape=_shape
            )
            _layers = [_cloud]
            if not _fg.empty:
                _color = alt.Color(
                    "grp:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["molecule 1", "molecule 2"],
                        range=["#1c7ed6", "#e8590c"],
                    ),
                )
                # A line linking each highlighted molecule's measured->predicted
                # marks, to show how far the model's guess drifts.
                _link = (
                    alt.Chart(_fg)
                    .mark_line(opacity=0.5)
                    .encode(x=_x, y=_y, color=_color, detail="grp:N")
                )
                _marks = (
                    alt.Chart(_fg)
                    .mark_point(size=170, filled=False, strokeWidth=2.5)
                    .encode(
                        x=_x,
                        y=_y,
                        shape=_shape,
                        color=_color,
                        tooltip=["grp:N", "src:N", "x:Q", "y:Q"],
                    )
                )
                _layers += [_link, _marks]
            return mo.as_html(
                alt.layer(*_layers).properties(height=380, width=440)
            )

        _scatters = _endpoint_vs_endpoint()
        _caption = mo.md(
            f"The whole plot is the two-endpoint space: **{_ta} pKi** across, "
            f"**{_tb} pKi** up. Each molecule appears as **● measured** and "
            f"**◇ predicted**, joined by a line. The "
            f"<span style='color:#1c7ed6'>**molecule 1**</span> / "
            f"<span style='color:#e8590c'>**molecule 2**</span> marks are the cliff "
            f"pair you picked above — near-identical structures that sit far apart on "
            f"one axis. The gap between a molecule's ● and ◇ is how far the learned "
            f"fingerprint's guess drifted from the truth. (Measured marks need both "
            f"endpoints assayed; the picked cliff molecules usually sit in the "
            f"model's *training* split — scaffold-splitting keeps their shared "
            f"scaffold together — so their predictions are optimistic, but their "
            f"*position* still shows the cliff.)"
        )
        _view = mo.vstack(
            [
                mo.md(_note) if _note else mo.md(""),
                _readout,
                _scatters,
                _caption,
                mo.md("**How each endpoint's accuracy trades off as you move α:**"),
                _tradeoff,
            ]
        )
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### About this notebook

    **AI use (disclosed per competition guidelines).** This notebook was built in
    a pairing session with an AI coding assistant: it helped scaffold the marimo
    cells, the custom `ComplexViewer` anywidget, and the analysis scripts, and
    drafted the prose. Every chemical claim, data source, and result was
    reviewed by a human; the AI wrote no chemistry it wasn't checked on.

    **Chemistry & validity.** All structure handling runs through **RDKit**.
    SMILES are validated on input and invalid structures are handled gracefully
    — no cell throws on bad input. Activity data are from **MoleculeACE**
    (curated ChEMBL bioactivities with published activity-cliff labels); the
    learned model uses **scaffold splits** to avoid train/test leakage. 3D
    complexes are **Boltz-2** predictions — framed throughout as *hypotheses*,
    not experimental structures — and protein–ligand interactions are detected
    with **PLIP**. Predicted poses are never presented as ground truth.

    **Reproducibility.** Heavy compute (folding, interaction detection, model
    training) runs offline and is cached in `data/`; the notebook only reads
    those caches, so it stays instant and deterministic. Fingerprint code and
    analyses live in this repo — see `README.md`.

    **Credits.** RDKit · chemprop (D-MPNN) · Boltz-2 · PLIP · MoleculeACE ·
    3Dmol.js · Altair · marimo. Thanks to OpenADMET and the marimo team for the
    competition.
    """)
    return


if __name__ == "__main__":
    app.run()
