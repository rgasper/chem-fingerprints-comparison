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
            "are *sparse*. Unlike MACCS, an off bit here carries no meaning of its "
            "own (it just means no atom environment hashed to that slot), so the "
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
            ),
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    chemeleon_floor = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.05,
        value=0.5,
        label="Structure-sensitivity floor (fraction of this molecule's max)",
        show_value=True,
        full_width=True,
    )
    return (chemeleon_floor,)


@app.cell
def _(chemeleon_floor, current_mol, mo, mol_valid):
    from fingerprints import chemeleon_fp as chf

    # CheMeleon is a *pretrained* neural fingerprint: nobody hand-designed its
    # 2048 dimensions - a message-passing network learned them from large
    # molecular data. Pick a dimension and see which atoms drive it for the
    # current molecule (the learned analog of the Morgan bit-scrubber). The
    # floor knob controls how sensitive a dimension must be to count.
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
            f"*estimated* read of what the dimension keys on for this molecule — "
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

*Caveat:* this is an honest, exact decomposition of the **mean-pool**, but a
single learned dimension rarely maps to one human-named substructure the way a
Morgan bit does — read it as “where this dimension looks,” not “what it is.”
"""
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
    mo.vstack([_structures, _change, _endpoints, mo.md("---")])
    return


@app.cell
def _(alt, cliff_choice, ctx, cv, mo, pd, target_pair_choice):
    # The reveal: every fingerprint scores this pair as fairly similar - one
    # number, blind to which endpoint it's being applied to.
    _tp = ctx.by_key()[target_pair_choice.value]
    _pair = _tp.cliffs[cliff_choice.value]
    _scores = cv.fingerprint_scores(_pair)
    # An interaction fingerprint reads the 3D pose, not the 2D graph - add it as
    # a contrasting bar when this cliff has cached poses (in the cliff target's
    # pocket, the endpoint where potency actually moves).
    _plif = cv.plif_similarity(_tp.key, cliff_choice.value, _pair.cliff_on)
    _rows = [
        {"fingerprint": s.label, "similarity": round(s.similarity, 3), "kind": "structure"}
        for s in _scores
    ]
    if _plif is not None:
        _rows.append(
            {
                "fingerprint": f"Interaction FP · {_pair.cliff_on} pocket",
                "similarity": round(_plif, 3),
                "kind": "interaction",
            }
        )
    _df = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X(
                "similarity:Q",
                title="fingerprint similarity",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("fingerprint:N", sort="-x"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["structure", "interaction"],
                    range=["#4c6ef5", "#e8820c"],
                ),
                legend=alt.Legend(title="reads", orient="bottom"),
            ),
            tooltip=[alt.Tooltip("fingerprint:N"), alt.Tooltip("similarity:Q")],
        )
        .properties(height=250)
    )
    _lo = min(s.similarity for s in _scores)
    _hi = max(s.similarity for s in _scores)
    _punchline = mo.md(
        f"Every **structure** fingerprint calls this pair **similar** (similarity "
        f"{_lo:.2f}–{_hi:.2f}) — they only see the small structural change. "
        f"That verdict is **right for {_pair.flat_on}** (where the pair really is "
        f"flat) and **badly wrong for {_pair.cliff_on}** (where it's a cliff). "
        "One structural similarity, two opposite biological realities — the "
        "fingerprint cannot tell which target you mean."
    ).callout(kind="warn")
    _bridge = None
    if _plif is not None:
        _bridge = mo.md(
            f"The **orange bar** is different in kind. It's read off each ligand's "
            f"predicted **3D pose in the {_pair.cliff_on} pocket** — which specific "
            f"contacts (H-bonds, π-stacks, salt bridges) each molecule actually "
            f"makes. On that footing the pair scores **{_plif:.2f}**, far below the "
            f"structure fingerprints: the interaction fingerprint *resolves* a "
            f"difference the 2D graph hides. (These are Boltz-predicted poses for a "
            f"handful of pairs, so read the number as a qualitative contrast, not a "
            f"benchmark.) That's the move the next section makes — leaving 2D behind "
            f"and looking at the pose itself."
        ).callout(kind="info")
    mo.vstack(
        [
            mo.md("**How similar each fingerprint thinks this pair is:**"),
            mo.as_html(_chart),
            _punchline,
            *([_bridge] if _bridge is not None else []),
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(cliff_choice, ctx, mo, target_pair_choice):
    from rdkit import Chem as _Chem

    from fingerprints import importance_view as iv

    # Where does a MODEL look? Train a RandomForest to predict activity from
    # each fingerprint, then project its feature importances back onto the
    # cliff pair - as a molecule heatmap and as an importance-tinted strip.
    # (Trained on the Dopamine D3/D4 datasets; shown when that pair is picked.)
    _tp = ctx.by_key()[target_pair_choice.value]
    _cl = _tp.cliffs[cliff_choice.value]
    _eps = iv.endpoints() if iv.has_data() else []
    _have = _tp.target_a in _eps and _tp.target_b in _eps
    if not _have:
        _view = mo.md(
            "*Feature-importance models were trained on the **Dopamine D3/D4** "
            "pair - pick that pair above to see where the models look. "
            "(Run `scripts/train_importance.py` to add more.)*"
        ).callout(kind="info")
    else:
        _m1 = _Chem.MolFromSmiles(_cl.smiles_1)
        _m2 = _Chem.MolFromSmiles(_cl.smiles_2)

        def _panel(mol, mol_name, fp, fp_name):
            _heat = iv.importance_heatmap_svg(mol, _cl.cliff_on, fp, width=300, height=220)
            _strip = iv.strip_svg(mol, _cl.cliff_on, fp, width=300, height=26)
            return mo.vstack(
                [
                    mo.md(f"**{mol_name} - {fp_name}**"),
                    mo.Html(_heat),
                    mo.md("*importance-tinted fingerprint*"),
                    mo.Html(_strip),
                ]
            )

        _mt_e = iv.metrics(_cl.cliff_on, "ecfp")
        _mt_c = iv.metrics(_cl.cliff_on, "chemeleon")
        _intro = mo.md(
            f"A RandomForest predicting **{_cl.cliff_on}** pKi "
            f"(ECFP R² {_mt_e['r2']:.2f} · CheMeleon R² {_mt_c['r2']:.2f}). "
            f"Green marks where the model leans to make its call. Even with the "
            f"regions highlighted, the two near-identical molecules light up "
            f"almost the same - the model has no special signal for the cliff."
        )
        _grid = mo.vstack(
            [
                mo.hstack(
                    [_panel(_m1, "molecule 1", "ecfp", "ECFP"),
                     _panel(_m2, "molecule 2", "ecfp", "ECFP")],
                    widths=[1, 1], gap=2,
                ),
                mo.hstack(
                    [_panel(_m1, "molecule 1", "chemeleon", "CheMeleon"),
                     _panel(_m2, "molecule 2", "chemeleon", "CheMeleon")],
                    widths=[1, 1], gap=2,
                ),
            ]
        )
        _view = mo.vstack([_intro, _grid])
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why a similarity model can't see the cliff

    Feature-importance shows *where* a model looks; it doesn't explain *why* it
    stays blind. For that, use the simplest possible model — **k-nearest-
    neighbours** on the fingerprint. A molecule's predicted activity is just the
    **average activity of its k most-similar neighbours**, where "similar" is
    the fingerprint's Tanimoto. Nothing is learned on top: the fingerprint's
    notion of similarity *is* the entire model. So whatever kNN can't do here is
    the fingerprint's limitation, laid bare.
    """)
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
def _(alt, cliff_choice, ctx, mo, pd, resample_flat, target_pair_choice):
    from fingerprints import knn_view as knn

    # The general impossibility argument, made concrete. Any structure-only
    # model is effectively a *smooth* function of the fingerprint: similar
    # fingerprint -> similar predicted activity. This census shows WHY that
    # assumption is forced - among structurally similar pairs, the overwhelming
    # majority really are flat, so a model must default to smooth to be
    # accurate, and the rare cliff is collateral damage.
    _tp = ctx.by_key()[target_pair_choice.value]
    _idx = cliff_choice.value
    _cliff_ep = _tp.cliffs[_idx].cliff_on if _tp.cliffs else None
    _eps = knn.endpoints() if knn.has_data() else []

    if _cliff_ep not in _eps:
        _view = mo.md(
            "*This census was precomputed for the **Dopamine D3/D4** datasets — "
            "pick that pair above.*"
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
            f"**The impossibility, in one chart.** Take every pair of {_cliff_ep} "
            f"molecules that a fingerprint calls *similar* (Tanimoto ≥ "
            f"{_s['sim_threshold']:.1f}): **{_s['n_similar_pairs']:,}** pairs. "
            f"**{_flat_pct}%** of them are **flat** (activity within "
            f"{_s['flat_gap']:.0f} log unit) and only **{_cliff_pct:.1f}%** are "
            f"true cliffs — roughly **{_ratio}:1**.\n\n"
            f"So 'similar structure → similar activity' is *right the vast majority "
            f"of the time*. Any model that predicts from structure alone is "
            f"rewarded for assuming it — and a model that instead predicted big "
            f"activity jumps for near-identical structures would be wrong on those "
            f"{_flat_pct}% to catch the {_cliff_pct:.1f}%. **A cliff is where nature "
            f"breaks the very assumption that makes the fingerprint useful.** No "
            f"amount of model cleverness recovers information the structure encoding "
            f"never contained — which is why activity cliffs are a well-documented "
            f"hard limit in QSAR, not a modelling bug."
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
                    "**The flat majority — similar structure, similar activity.** "
                    "These random 'similar' pairs behave exactly as the assumption "
                    "predicts, which is why the assumption pays off. Resample to see "
                    "more; you'll have to hunt to find a cliff."
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
def _(alt, cliff_choice, ctx, k_slider, knn, mo, pd, target_pair_choice):
    # kNN cliff analysis was precomputed for the Dopamine D3/D4 datasets.
    _tp = ctx.by_key()[target_pair_choice.value]
    _idx = cliff_choice.value
    _k = k_slider.value

    _eps = knn.endpoints() if knn.has_data() else []
    _cliff_ep = _tp.cliffs[_idx].cliff_on if _tp.cliffs else None
    _pair = knn.cliff_pair(_cliff_ep, _idx) if _cliff_ep in _eps else None

    if _pair is None:
        _view = mo.md(
            "*The kNN cliff analysis was precomputed for the **Dopamine D3/D4** "
            "pair — pick that pair above to explore it. "
            "(Run `scripts/analyze_knn_cliffs.py` to add more.)*"
        ).callout(kind="info")
    else:
        _ep = _pair["cliff_on"]
        _m1, _m2 = _pair["mol1"], _pair["mol2"]
        _meta = knn.endpoint_meta(_ep)
        _best = knn.best_k(_ep)

        # (a) The k tradeoff: held-out R^2 vs k, with the chosen k marked.
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
        _acc_chart = (_line + _rule).properties(height=200)

        # (b) The cliff itself: predicted vs true for both molecules at this k.
        _p1 = knn.pred_at_k(_m1, _k)
        _p2 = knn.pred_at_k(_m2, _k)
        _pred_gap = abs(_p1 - _p2)
        _true_gap = abs(_m1["true"] - _m2["true"])
        _bars = pd.DataFrame(
            [
                {"mol": "molecule 1", "kind": "true activity", "pKi": _m1["true"]},
                {"mol": "molecule 1", "kind": f"kNN predicted (k={_k})", "pKi": _p1},
                {"mol": "molecule 2", "kind": "true activity", "pKi": _m2["true"]},
                {"mol": "molecule 2", "kind": f"kNN predicted (k={_k})", "pKi": _p2},
            ]
        )
        _cliff_chart = (
            alt.Chart(_bars)
            .mark_bar()
            .encode(
                x=alt.X("kind:N", title=None, axis=alt.Axis(labelAngle=-15)),
                y=alt.Y("pKi:Q", title=f"{_ep} pKi"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(range=["#2b8a3e", "#adb5bd"]),
                    legend=None,
                ),
                column=alt.Column("mol:N", title=None),
            )
            .properties(width=150, height=200)
        )

        # Neighbour panels: draw each cliff molecule and its nearest
        # neighbours as structures, so you SEE it sits among similar-activity
        # molecules (colour the neighbour caption by how close its activity is).
        from rdkit import Chem as _Chem
        from rdkit.Chem.Draw import rdMolDraw2D as _d2d

        def _mol_svg(smi, w, h, highlight=False):
            m = _Chem.MolFromSmiles(smi)
            d = _d2d.MolDraw2DSVG(w, h)
            d.drawOptions().addStereoAnnotation = False
            if m is not None:
                _d2d.PrepareAndDrawMolecule(d, m)
            d.FinishDrawing()
            return d.GetDrawingText()

        def _neighbor_panel(m, name):
            _cap = 8  # most neighbour structures we'll draw
            _show = min(_k, _cap)
            _nbrs = m["neighbors"][:_show]
            head = mo.vstack(
                [
                    mo.md(f"**{name}** — true pKi **{m['true']:.2f}**"),
                    mo.Html(_mol_svg(m["smiles"], 200, 150)),
                    mo.md(
                        f"<div style='text-align:center'>kNN predicts "
                        f"<b>{knn.pred_at_k(m, _k):.2f}</b> at k={_k}</div>"
                    ),
                    mo.md(
                        f"<div style='text-align:center;color:#868e96'>↓ the "
                        f"{'nearest neighbour' if _show == 1 else f'{_show} nearest neighbours'} "
                        f"kNN averages at k={_k} ↓</div>"
                    ),
                ]
            )
            cards = []
            for nb in _nbrs:
                near = abs(nb["activity"] - m["true"]) < 1.0
                col = "#2b8a3e" if near else "#e8820c"
                cards.append(
                    mo.vstack(
                        [
                            mo.Html(_mol_svg(nb["smiles"], 120, 90)),
                            mo.md(
                                f"<div style='text-align:center;font-size:11px'>"
                                f"T={nb['tanimoto']:.2f}<br>"
                                f"<span style='color:{col}'>pKi {nb['activity']:.2f}</span></div>"
                            ),
                        ]
                    )
                )
            if _k > _cap:
                cards.append(
                    mo.md(
                        f"<div style='text-align:center;color:#868e96;font-size:12px;"
                        f"padding-top:35px'>…and<br><b>{_k - _cap}</b> more<br>averaged in</div>"
                    )
                )
            return mo.vstack(
                [head, mo.hstack(cards, justify="center", gap=0.5)]
            )

        # Honest, per-pair statement: the *largest* gap kNN opens across ANY k.
        _max_gap = max(
            abs(a["pred"] - b["pred"])
            for a, b in zip(_m1["pred_by_k"], _m2["pred_by_k"])
        )
        _verdict = mo.md(
            f"At **k={_k}**, kNN predicts these two molecules **{_p1:.2f}** and "
            f"**{_p2:.2f}** — a gap of just **{_pred_gap:.2f}**, though the real gap "
            f"is **{_true_gap:.2f}**. The structures below are exactly the {_k} "
            f"molecule{'s' if _k != 1 else ''} kNN averages at this k. "
            f"**Slide k across its whole range:** the widest gap it ever opens for "
            f"this pair is only **{_max_gap:.2f}** — no neighbourhood size, not even "
            f"k=1, comes close to the true **{_true_gap:.2f}**. Meanwhile global "
            f"accuracy peaks near **k={_best['k']}** (R² {_best['r2']:.2f}); the k "
            f"that fits the dataset best still can't see this cliff."
        ).callout(kind="warn")

        _view = mo.vstack(
            [
                mo.md(
                    f"**{_ep}** — {_meta['n_total']} molecules. The only knob is k; "
                    "the orange line marks your choice."
                ),
                mo.hstack(
                    [
                        mo.vstack([mo.md("**Global accuracy vs k**"), mo.as_html(_acc_chart)]),
                        mo.vstack([mo.md("**This cliff pair at k**"), mo.as_html(_cliff_chart)]),
                    ],
                    widths=[1, 1], gap=2,
                ),
                k_slider,
                _verdict,
                _neighbor_panel(_m1, "molecule 1"),
                mo.md("---"),
                _neighbor_panel(_m2, "molecule 2"),
            ]
        )
    mo.vstack([_view, mo.md("---")])
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
    mo.vstack([_view, mo.md("---")])
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
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 🧪 Fingerprint playground: poke the molecule

    Take the molecule you picked at the top, apply one small, real medicinal-
    chemistry edit, and watch how the fingerprints react. Some edits barely
    nudge them; some flip a surprising number of bits. **That spread is the
    whole point** — a fingerprint is a *chosen* notion of similarity, so the
    same one-atom change lands very differently depending on which fingerprint
    is looking.
    """)
    return


@app.cell
def _(current_mol, mo, mol_valid):
    from fingerprints import mol_edits as med

    # Offer only edits that produce a valid, distinct product for THIS molecule,
    # so a user can never click a button that no-ops or errors.
    _edits = med.applicable_edits(current_mol) if mol_valid else []
    if _edits:
        edit_choice = mo.ui.dropdown(
            options={e.label: e.key for e in _edits},
            value=_edits[0].label,
            label="Pick an edit to apply",
        )
    else:
        edit_choice = mo.ui.dropdown(options={"(none)": ""}, value="(none)")
    applicable = _edits
    return applicable, edit_choice, med


@app.cell
def _(applicable, current_mol, edit_choice, med, mo, mol_valid):
    from rdkit.Chem.Draw import rdMolDraw2D as _draw2d

    def _svg(mol, width=280, height=210):
        d = _draw2d.MolDraw2DSVG(width, height)
        d.drawOptions().addStereoAnnotation = False
        _draw2d.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()

    if not mol_valid or not applicable:
        _view = mo.md(
            "*Pick a valid molecule at the top with at least one applicable edit "
            "(single atoms like `C` or `O` have none).*"
        ).callout(kind="info")
    else:
        _key = edit_choice.value
        _edit = med.EDITS_BY_KEY.get(_key)
        _product = med.apply_edit(current_mol, _key) if _key else None
        if _product is None:
            _view = mo.md("*That edit didn't apply here — pick another.*").callout(
                kind="info"
            )
        else:
            _stats = med.ecfp_diff_stats(current_mol, _product)
            _ecfp_svg = med.ecfp_diff_svg(current_mol, _product, width=900, height=46)
            _chem_svg = med.chemeleon_delta_svg(
                current_mol, _product, width=900, height=46
            )

            _structures = mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**before**"), mo.Html(_svg(current_mol))],
                        align="center",
                    ),
                    mo.md("## →"),
                    mo.vstack(
                        [mo.md("**after**"), mo.Html(_svg(_product))],
                        align="center",
                    ),
                ],
                justify="center",
                gap=1,
            )
            _what = mo.md(f"**{_edit.label}.** {_edit.description}").callout(
                kind="neutral"
            )

            # ECFP: the bit vector's response, drawn as a diff strip.
            _ecfp_legend = mo.md(
                f'ECFP Tanimoto **{_stats["tanimoto"]:.2f}** &nbsp;—&nbsp; '
                f'<span style="color:#868e96">█ {_stats["shared"]} shared</span> &nbsp; '
                f'<span style="color:#2f9e44">█ {_stats["added"]} switched on</span> &nbsp; '
                f'<span style="color:#e03131">█ {_stats["removed"]} switched off</span>'
            )
            _ecfp_block = mo.vstack(
                [
                    mo.md("**ECFP (Morgan) — which bits flipped?**"),
                    mo.Html(_ecfp_svg),
                    _ecfp_legend,
                ]
            )

            # CheMeleon: continuous embedding, so show the signed per-dimension
            # shift for the dimensions that moved most.
            if _chem_svg is not None:
                _chem_block = mo.vstack(
                    [
                        mo.md(
                            "**CheMeleon (learned) — how the embedding shifted**"
                        ),
                        mo.Html(_chem_svg),
                        mo.md(
                            '<span style="color:#1c7ed6">█ dimension moved up</span> &nbsp; '
                            '<span style="color:#e8820c">█ dimension moved down</span> &nbsp; '
                            "(all 2048 dimensions; shade = size of change — no "
                            "discrete bits, just a continuous shift)"
                        ),
                    ]
                )
            else:
                _chem_block = mo.md(
                    "*CheMeleon weights unavailable — showing ECFP only.*"
                ).callout(kind="info")

            _view = mo.vstack(
                [
                    _structures,
                    _what,
                    _ecfp_block,
                    mo.md(""),
                    _chem_block,
                    mo.md(
                        "Flip between edits and watch the two panels disagree: a "
                        "**halogen** or **magic methyl** often leaves the strip "
                        "mostly grey, while an **aza-swap** or **bioisostere** "
                        "lights up far more — even though the change is chemically "
                        "'small'. Neither fingerprint knows whether the edit matters "
                        "*biologically*; each just reports its own chosen notion of "
                        "similarity."
                    ),
                ]
            )
    mo.vstack([edit_choice, _view, mo.md("---")])
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

    **Credits.** RDKit · chemprop (D-MPNN) · CheMeleon · Boltz-2 · PLIP ·
    MoleculeACE · 3Dmol.js · Altair · marimo. Thanks to OpenADMET and the
    marimo team for the competition.
    """)
    return


if __name__ == "__main__":
    app.run()
