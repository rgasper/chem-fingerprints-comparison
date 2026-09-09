import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    return alt, mo


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
    ## 2 · The opposite idea — Morgan's atom environments

    MACCS asks a fixed list of expert questions. **Morgan** (the ECFP family,
    the most-used fingerprint in modern cheminformatics) does the opposite: it
    has *no* predefined patterns. For every atom it looks at the **circular
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
def _(alt, current_mol, me, mo, mol_valid):
    # Collision rate as the vector lengthens: the payoff of a longer fingerprint.
    if mol_valid:
        _curve = me.collision_curve(current_mol)
        _distinct = _curve[0].distinct_envs
        _rows = [
            {
                "length": cp.n_bits,
                "rate": round(100 * cp.collisions / cp.distinct_envs, 1)
                if cp.distinct_envs
                else 0.0,
            }
            for cp in _curve
        ]
        _chart = (
            alt.Chart(alt.Data(values=_rows))
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
                mo.ui.altair_chart(_chart),
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
    Flip between the MACCS explorer above and this one on the **same molecule**:
    MACCS highlights whole named motifs (a carbonyl, a ring), while Morgan
    highlights many small overlapping neighborhoods. Two fundamentally different
    ways to describe the same structure — which is exactly why they disagree
    about what "similar" means.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Same molecule, six different fingerprints

    MACCS and Morgan are just two of many. Each fingerprint below asks a
    *different question* about the **same** structure — and produces a very
    different vector. Notice how they disagree even on something as basic as
    *how many bits light up*: a compact expert checklist (MACCS) versus dense
    hashed paths (RDKit-topological) versus sparse atom environments (Morgan).

    This is the whole point: **"similar" means something different to each
    fingerprint.** Change the molecule above and watch every bar move.
    """)
    return


@app.cell
def _(alt, current_mol, mo, mol_valid):
    from fingerprints import fp_overview as ov

    if mol_valid:
        _summaries = ov.summarize_all(current_mol)
        # Altair accepts a list of dicts directly — no pandas/pyarrow needed.
        _rows = [
            {
                "fingerprint": s.name,
                "bits_on": s.n_on,
                "n_features": s.n_features,
                "density_pct": round(100 * s.density, 1),
                "encodes": s.description,
            }
            for s in _summaries
        ]
        _chart = (
            alt.Chart(alt.Data(values=_rows))
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("bits_on:Q", title="number of bits set to 1"),
                y=alt.Y("fingerprint:N", sort="-x", title=None),
                color=alt.Color(
                    "density_pct:Q",
                    title="% of bits on",
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    alt.Tooltip("fingerprint:N"),
                    alt.Tooltip("bits_on:Q", title="bits on"),
                    alt.Tooltip("n_features:Q", title="total bits"),
                    alt.Tooltip("density_pct:Q", title="% on"),
                    alt.Tooltip("encodes:N", title="encodes"),
                ],
            )
            .properties(height=240, title="Bits set for the active molecule")
        )
        _view = mo.vstack(
            [
                mo.ui.altair_chart(_chart),
                mo.md(
                    "Hover a bar to read what that fingerprint encodes. The color "
                    "shows *density* — what fraction of the whole vector is on — "
                    "which is a rough proxy for how finely the fingerprint slices "
                    "structure."
                ),
            ]
        )
    else:
        _view = mo.md("*Select a valid molecule to compare fingerprints.*")
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    *Next sections (Morgan atom-environment explorer, similarity playground,
    ADMET applications, and activity-cliff limitations) are under construction.
    The custom anywidget bit-explorer upgrade lands here.*

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
