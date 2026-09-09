import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
            mx.BitHit(bit=-1, is_on=False, smarts=None, name="", match_count=0, atoms=(), bonds=()),
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
    ring?", "a chlorine?"). Each question is a
    [SMARTS](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html)
    pattern; a bit is **1** if the molecule contains that substructure. That's the
    whole fingerprint: a 166-long yes/no checklist.

    Because every bit *is* a named substructure, we can point at exactly which
    atoms answered "yes." **Scrub the slider** to walk through the bits that are
    ON for the active molecule and watch the matching substructure highlight.
    """)
    return


@app.cell
def _(current_mol, mo, mol_valid, mx):
    if mol_valid:
        _on = mx.on_bits(current_mol)
    else:
        _on = []
    on_bit_list = _on

    if _on:
        bit_slider = mo.ui.slider(
            start=0,
            stop=len(_on) - 1,
            value=0,
            label=f"Scrub the {len(_on)} MACCS bits that are ON",
            full_width=True,
            show_value=False,
        )
    else:
        bit_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")
    bit_slider
    return bit_slider, on_bit_list


@app.cell
def _(bit_slider, current_mol, mo, mol_valid, mx, on_bit_list):
    if mol_valid and on_bit_list:
        _bit = on_bit_list[bit_slider.value]
        _hit = mx.bit_hit(current_mol, _bit)
        _svg = mx.highlight_svg(current_mol, _hit, width=460, height=340)
        _smarts_line = (
            f"**SMARTS:** `{_hit.smarts}`"
            if _hit.smarts
            else "*count-based key (no SMARTS pattern)*"
        )
        _card = mo.vstack(
            [
                mo.md(f"### Bit {_hit.bit} · {_hit.name}"),
                mo.md(_smarts_line),
                mo.md(
                    f"Matches in this molecule: **{_hit.match_count}** "
                    f"(highlighted atoms: {len(_hit.atoms)})"
                ),
            ]
        )
        _view = mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[1, 1])
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
def _(current_mol, mo, mol_valid):
    import altair as alt

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
