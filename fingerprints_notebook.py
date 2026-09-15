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

    *A molab Notebook Competition entry — the history, mechanics, applications,
    and limitations of molecular fingerprints in cheminformatics.*

    A **molecular fingerprint** turns a molecule into a vector of numbers so a
    computer can ask "how similar are these two molecules?" without understanding
    chemistry. Fingerprints are the workhorse behind similarity search, clustering,
    and cheap ADMET models. But they all encode *different* notions of similarity —
    and each has blind spots.

    **Pick a molecule once, below. Everything in this notebook reacts to it.**
    That's marimo's reactive dataflow: change the molecule and every visualization
    downstream recomputes, so you can build intuition by exploration.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The molecule selector — the one input everything flows from
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
def _(mo):
    mo.md(r"""
    ## 1 · What *is* a fingerprint? Start with MACCS

    MACCS keys are the friendliest fingerprint to learn from: a fixed list of
    **166 predefined structural questions** ("is there a carbonyl?", "an aromatic
    ring?", "a chlorine?"). Most questions are written as a
    [SMARTS](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html)
    pattern; a bit is **1** if the molecule contains that substructure. A handful
    of keys are *count-based* — they only turn on past a threshold (e.g. "more
    than 3 oxygens") — and three keys are *special*, computed directly rather
    than by pattern-matching. That's the whole fingerprint: a 166-long checklist.

    Because every bit *is* a named substructure, we can describe each one in
    **plain English**, show the **pattern it's looking for** (drawn from its
    definition, independent of any molecule), point to **where it matches** on
    the active molecule, and mark where the bit sits within the **whole
    fingerprint**. Scrub through all 166 keys — the ones that are *on* tell you
    what the molecule has, and the *off* ones are just as informative: they tell
    you what it's **missing**. (The terse original SMARTS is tucked under
    "Technical details" — nobody reads those at a glance anyway.)
    """)
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
def _(mo):
    mo.md(r"""
    ## 2 · The most-used fingerprint: Morgan (ECFP)

    If you use one fingerprint in cheminformatics, it's this one. **Morgan**
    (a.k.a. ECFP) takes a different tack from MACCS's fixed checklist: it has
    *no* predefined patterns. For every atom it looks at the **circular
    neighborhood** growing outward — radius 0 (the atom alone), radius 1 (plus
    immediate neighbors), radius 2 (their neighbors too) — and hashes each of
    those environments into a bit.

    So a Morgan bit has no human name. But we can still show exactly what it
    means: **scrub the slider** and each bit lights up the atom environment that
    produced it. The <span style="color:#f25a40">**red center atom**</span> is
    where the environment grew from; the <span style="color:#338cf2">**blue
    shading**</span> is how far it reached.
    """)
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
def _(mo):
    mo.md(r"""
    ### Why is Morgan 2048 bits long? Hash collisions.

    Morgan has *no* fixed vocabulary, so it can't reserve a slot per feature the
    way MACCS does. Instead it **hashes** each atom environment into one of a
    fixed number of bits. When two *different* environments hash to the **same**
    bit, that's a **collision** — the fingerprint literally cannot tell them
    apart anymore.

    Let's *see* it. Below we deliberately fold the active molecule into an
    absurdly short **8-bit** Morgan fingerprint and find the bits that ended up
    shared. Each colored region is a **different** substructure — but the short
    fingerprint records them all as the *same* single bit. Scrub through the
    collisions:
    """)
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
        _view = mo.md("*Select a valid molecule.*")
    elif not short_collisions:
        _view = mo.md(
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
        _view = mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[3, 2])
    _view
    return


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
        _view = mo.vstack(
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
        _view = mo.md("")
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    MACCS and Morgan sit at two extremes: a fixed expert checklist versus
    hashed local environments. On the **same molecule** they highlight totally
    different things — whole named motifs vs. many small overlapping
    neighborhoods. That's the first hint of a theme we'll keep hitting:
    **"similar" means something different to every fingerprint.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · The rest of the RDKit toolbox

    Beyond MACCS (a substructure-key fingerprint) and Morgan (a circular
    atom-environment fingerprint), RDKit ships several more classical
    fingerprints. They fall into a few families:

    - **Path-based** — hash linear walks through the molecular graph
      (RDKit topological).
    - **Atom-pair** — encode pairs of atoms and the distance between them.
    - **Torsion-based** — encode short 4-atom backbone fragments
      (topological torsion).

    Browse them below — same molecule, same bit-highlight idea — to see how
    each one "sees" structure differently. (They react to the molecule selector
    at the top.)
    """)
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
    tabbed_fps
    return


@app.cell
def _(mo):
    mo.md(r"""
    **One more, without a highlight: Avalon.** Avalon is a path- and
    feature-based fingerprint computed by a separate C++ toolkit that RDKit
    wraps as a black box — it returns only the final bit vector, with **no
    per-bit atom mapping**. So unlike the others, we can't point at which atoms
    set each bit. That opacity is itself the lesson: a fingerprint's
    interpretability depends on whether its implementation hands back
    provenance. Avalon performs well on similarity tasks but won't tell you
    *why*.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4 · Where fingerprints break: activity cliffs

    Everything so far rests on one assumption: **similar structure → similar
    behavior**. That's why fingerprints work for search and cheap property
    models. **Activity cliffs** are the pairs where it fails — a tiny structural
    change causing a huge change in potency.

    Here's the twist that makes them genuinely hard: a cliff is **not a property
    of the molecule pair alone — it depends on the endpoint you ask about.** The
    same one-atom swap can be a 100× cliff for one target and completely flat for
    a closely related one. A fingerprint sees only structure, so it assigns *one*
    similarity to the pair — and that single number is right for the endpoint
    where the pair is flat and badly wrong for the endpoint where it's a cliff.

    Pick a target pair and a molecule pair below and see it happen. (These pairs
    are curated from the [MoleculeACE](https://github.com/molML/MoleculeACE)
    benchmark — each is a real medicinal-chemistry change, hand-checked so
    there are no tautomer or assay-artifact traps.)
    """)
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
def _(mo):
    mo.md(r"""
    **Why this is the hard case.** A cliff isn't noise — these are real,
    reproducible measurements. It's that the property surface is genuinely
    *rugged* in a way a structure-only representation can't anticipate, and
    *differently* rugged for every target. (As a sanity check: two targets with
    near-identical binding sites — JAK1 and JAK2 — share hundreds of molecules
    but yield **zero** context-dependent cliffs in this benchmark. Cliffs only
    appear where the biology actually diverges.)

    So what can learn the difference? Before we get there, let's look at the
    cliff in **3D** — does the physical picture explain it?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4b · Does 3D structure explain the cliff?

    Fingerprints only see the 2D graph, so of course they miss the cliff. But
    surely the **3D structure** — the ligand actually sitting in each pocket —
    would reveal *why* the same change matters on one receptor and not the other?

    For the cliff pairs we co-folded with
    [Boltz-2](https://github.com/jwohlwend/boltz) (a state-of-the-art structure
    predictor), the four predicted complexes appear below — both molecules in
    both receptors. Rotate them; they start from a common orientation. The
    <span style="color:#e8820c">**orange atom**</span> is the one that changes
    across the cliff, and <span style="color:#c026a3">**magenta**</span> is the
    conserved aspartate (D3.32) that every aminergic-GPCR ligand's amine anchors
    to — confirming the poses land in the real orthosteric pocket. Dashed lines
    are protein–ligand interactions detected by
    [PLIP](https://github.com/pharmai/plip):
    <span style="color:#e0a800">**salt bridge**</span>,
    <span style="color:#4dabf7">**H-bond**</span>,
    <span style="color:#20c997">**π-stack**</span>,
    <span style="color:#e64980">**π-cation**</span>,
    <span style="color:#adb5bd">**hydrophobic**</span>.
    *(The picker above drives this; poses are precomputed, so only folded cliffs
    show 3D.)*
    """)
    return


@app.cell
def _(cliff_choice, ctx, mo, target_pair_choice):
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

        def _panel(mol_id, target, counterpart_mol, title):
            p = _poses[f"{mol_id}_{target}"]
            changed = pv.changed_atom_names(p, _poses[f"{counterpart_mol}_{target}"])
            v = ComplexViewer(
                structure=p.cif_text,
                format="cif",
                highlight_resi=_anchor_resi(p),
                highlight_atoms=changed,
                interactions=pv.load_interactions(p),
                height=340,
            )
            cap = mo.md(
                f"**{title}**  \nBoltz binding confidence "
                f"**{p.binding_confidence:.2f}**, ligand ipTM **{p.ligand_iptm:.2f}**"
            )
            return mo.vstack([cap, mo.ui.anywidget(v)])

        _ta, _tb = _tp.target_a, _tp.target_b
        _view = mo.vstack(
            [
                mo.md(f"#### Molecule 1 — *{_cliff.change}* (before)"),
                mo.hstack(
                    [
                        _panel("mol1", _ta, "mol2", f"in {_ta} — pKi {_cliff.pki_1_a}"),
                        _panel("mol1", _tb, "mol2", f"in {_tb} — pKi {_cliff.pki_1_b}"),
                    ],
                    widths=[1, 1],
                    gap=1,
                ),
                mo.md("#### Molecule 2 — after the change"),
                mo.hstack(
                    [
                        _panel("mol2", _ta, "mol1", f"in {_ta} — pKi {_cliff.pki_2_a}"),
                        _panel("mol2", _tb, "mol1", f"in {_tb} — pKi {_cliff.pki_2_b}"),
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
        _keys = [
            f"mol1_{_tp.target_a}", f"mol2_{_tp.target_a}",
            f"mol1_{_tp.target_b}", f"mol2_{_tp.target_b}",
        ]
        _col_labels = [
            f"mol 1 · {_tp.target_a}", f"mol 2 · {_tp.target_a}",
            f"mol 1 · {_tp.target_b}", f"mol 2 · {_tp.target_b}",
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
    mo.md(r"""
    **The honest result:** across these cliffs, Boltz places the two molecules
    in near-identical poses in each pocket, anchored the same way by the
    conserved aspartate — and often with similar confidence on the target where
    the pair is a cliff and the one where it's flat. The changed atom lands in
    essentially the same spot regardless of receptor. Even a state-of-the-art
    structure predictor rarely shows an *obvious* reason for the cliff.

    (On the μ/κ "ring CH₂→NH" pair, for instance, all four complexes score
    0.93–0.99 and the new NH sits ~3.8 Å from the anchoring aspartate in **both**
    the μ pocket, where it costs 810× potency, and the κ pocket, where it costs
    nothing.)

    That's not a failure of the demo — it *is* the point, now at the 3D level.
    The cliff is real and reproducible, but its cause lives in the things a
    single static pose doesn't capture: precise electrostatics, protonation,
    ordered waters, and receptor dynamics. Fingerprints miss cliffs because they
    only see 2D structure; here we see that even a full 3D model struggles. This
    is why activity cliffs remain one of the genuinely hard problems in
    computational drug discovery.

    *(Predicted poses are hypotheses, not experimental structures. We use them
    to reason about plausibility, not to assert a mechanism.)*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The turn: let the data define the fingerprint

    Step back and notice the pattern. Sections 1–3 built fingerprints by
    **imposing a lens**: MACCS's expert checklist, Morgan's circular
    environments, path and torsion hashes. *We* decided what a molecule's
    features are — and then Section 4 showed the cost: that fixed lens is blind
    to activity cliffs, because "similar structure" was *our* rule, not the
    biology's.

    The interaction fingerprint above is the first hint of the opposite move.
    Its bits weren't designed by us — they're **read off the binding event**:
    the pose tells us which contacts matter. The fingerprint came *from the
    data*.

    The next sections push that idea all the way. Instead of hand-designing
    features, we let a model **learn** the representation — first a pretrained
    foundation model (**CheMeleon**), then a small network we train ourselves on
    these very endpoints. The question throughout: can a fingerprint *learned
    from activity data* do what a fixed structural one can't — and what does it
    cost us when we try?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · A fingerprint the data learns — and the catch

    Now we let the data define the representation *end to end*. A
    [chemprop](https://github.com/chemprop/chemprop) **D-MPNN** (message-passing
    graph neural network) reads the raw molecular graph and **learns its own
    fingerprint**, driven only by the activity labels — no MACCS keys, no Morgan
    radius, no features we chose. (Feeding a fixed fingerprint into a network
    would just smuggle our imposed lens back in; the whole point is to let the
    graph speak.)

    We train it on **two related endpoints at once** — one shared learned
    fingerprint feeding two prediction heads. One knob, **α**, sets how much the
    training loss cares about endpoint A vs. endpoint B. Slide it and watch what
    the data gives you.
    """)
    return


@app.cell
def _(mo):
    alpha_knob = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.25,
        value=0.5,
        label="α — loss weight toward endpoint A (μ / D3)",
        show_value=True,
        full_width=True,
    )
    alpha_knob
    return (alpha_knob,)


@app.cell
def _(alpha_knob, alt, mo, pd):
    from fingerprints import learned_fp_view as lfv

    # Use whichever endpoint pair has a trained grid (mu/kappa is trained first).
    _pair = "mu_vs_kappa" if lfv.has_grid("mu_vs_kappa") else (
        lfv.available_pairs()[0] if lfv.available_pairs() else None
    )
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

        # Two big R2 readouts for the current alpha.
        def _card(target, r2, is_weighted):
            _kind = "success" if r2 > 0.3 else ("danger" if r2 < 0.1 else "neutral")
            _verdict = "learns it" if r2 > 0.3 else (
                "fails" if r2 < 0.1 else "partial"
            )
            return mo.md(
                f"#### {target}\n\ntest R² = **{r2:.2f}** — {_verdict}"
            ).callout(kind=_kind)

        _readout = mo.hstack(
            [
                _card(_ta, _cur["r2_a_mean"], True),
                _card(_tb, _cur["r2_b_mean"], True),
            ],
            widths=[1, 1],
            gap=2,
        )

        # Trade-off curve: R2 on each endpoint across the whole alpha grid,
        # with the current alpha marked.
        _rows = []
        for r in _g["results"]:
            _rows.append({"alpha": r["alpha"], "R2": r["r2_a_mean"], "endpoint": _ta})
            _rows.append({"alpha": r["alpha"], "R2": r["r2_b_mean"], "endpoint": _tb})
        _df = pd.DataFrame(_rows)
        _line = (
            alt.Chart(_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("alpha:Q", title="α (loss weight toward endpoint A)"),
                y=alt.Y("R2:Q", title="test R²", scale=alt.Scale(domain=[-0.1, 0.6])),
                color=alt.Color("endpoint:N", title=None),
            )
            .properties(height=240, width=440)
        )
        _rule = (
            alt.Chart(pd.DataFrame({"alpha": [alpha_knob.value]}))
            .mark_rule(color="#868e96", strokeDash=[4, 4])
            .encode(x="alpha:Q")
        )
        _view = mo.vstack([_readout, mo.as_html(_line + _rule)])
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The catch — and the payoff of the whole notebook.** Push α all the way to
    one endpoint and the model learns a fingerprint that's excellent there and
    *useless* on the other (R² near zero). There's a broad middle where one
    shared representation serves both endpoints decently — but you can't have it
    all: the fingerprint the data gives you **depends on which question you ask
    it**.

    That's the arc closing. Fixed fingerprints (MACCS, Morgan) impose one lens
    and are stuck with its blind spots — the activity cliff. Learned
    representations remove the imposed lens, but they don't escape the deeper
    truth: *there is no single, universal "similar"*. Structure only means
    something **relative to a question** — a target, an endpoint, an assay. Both
    halves of this notebook — the interaction fingerprint read off a pose, and
    the D-MPNN trained on labels — are the same move: stop dictating how the
    molecule should be a vector, and let the phenomenon tell you.

    *Rigor note: the numbers above are a deliberately simplified, in-notebook
    demo (one scaffold split, 3 seeds, a small D-MPNN). A fuller offline
    benchmark — 5×5-fold CV comparing the learned fingerprint against models on
    fixed fingerprints, with proper significance testing — tells the same story
    more carefully; see the repo.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### About this notebook

    Built as a pairing session with an AI coding assistant (disclosed per
    competition guidelines). All chemistry runs through **RDKit**; fingerprint
    code and analyses are reused from this repo's own investigation
    (see `README.md`). SMILES are validated on input and invalid structures are
    handled gracefully.
    """)
    return


if __name__ == "__main__":
    app.run()
