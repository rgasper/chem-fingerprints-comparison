# Comparing classical and neural molecular fingerprints

Eight fingerprints (six classical RDKit + two neural) probed across six lenses. The point is not to crown a winner — it's to show how different the fingerprints actually are, and where each one's blind spots hide.

## Fingerprints studied

- **Local atom-environment**: Morgan (r=2, 2048 bits)
- **Path-based**: RDKit topological, Avalon, AtomPair, TopTorsion
- **Expert keys**: MACCS (167 bits)
- **Neural**: CheMeleon, MIST-28M

## Investigations

| # | Lens | Question |
|---|------|----------|
| 1 | Pair plots | What does each FP think of hand-picked molecule pairs? |
| 2 | ADME alignment | Does FP geometry track ADME properties? |
| 3 | Agreement matrix | How redundant are the FPs across each other? |
| 4 | Scaffold hopping | Does the FP preserve scaffold structure, and is that bought at the cost of property awareness? |
| 5 | Activity cliffs | For pre-defined activity cliffs (small chemical change, large potency change), what similarity score does each FP assign? |

There's also a clustering experiment under `figures/clustering/` (HDBSCAN over UMAP-reduced fingerprints on a 10k-molecule random ChEMBL sample). The result was that random ChEMBL is too scaffold-diverse to cluster meaningfully — useful as a sanity check that "fingerprint geometry → clusters" is not a free lunch, but not the central story.

## Conclusions

### Fingerprints encode genuinely different similarity geometries

![RV agreement matrix](figures/agreement/01_rv_agreement_n5000.png)

The RV agreement matrix is the clearest evidence. Classical FPs are not redundant with each other — most pairs sit at RV 0.30–0.40. RDKit-topo and Avalon agree most (0.62, both path-based). MIST-28M is the most isolated (RV ~0.15–0.30 with most others). CheMeleon sits closer to the classical block than MIST does (RV 0.40–0.66 with classical FPs vs MIST's 0.15–0.30).

### Both neural FPs are heavily cliff-blind

Activity cliffs are pairs of structurally-similar molecules with very different potency. We define them fingerprint-agnostically: graph_distance ≤ 5 (atoms not shared via the maximum common substructure with strict aromatic/non-aromatic bond matching) AND |ΔpKi| ≥ 2.0. Then for each cliff pair we ask each FP for its similarity score:

![cliff-blind summary](figures/cliffs/03_cliff_blind_summary.png)

The hierarchy is consistent across all three ChEMBL targets, with the actual cliff-blind rates (P(similarity ≥ 0.7)) shown above:

- **Morgan** is the only FP with a cliff-blind rate below 50% on every target (D3 0.08, Thrombin 0.41, GSK-3β 0.20). TopTorsion is second-best on the harder targets (Thrombin 0.51, GSK-3β 0.46).
- **AtomPair** ranges widely (D3 0.24 → Thrombin 0.76) — strong on D3, weak on Thrombin.
- **RDKit-topo and Avalon** sit in the middle on D3 (0.39, 0.43) and become heavily cliff-blind on the harder sets (≥ 0.90 on Thrombin and GSK-3β).
- **MACCS** is cliff-blind on the harder targets (0.96–0.98) — 167 bits don't have the resolution.
- **CheMeleon and MIST-28M score nearly every cliff as similar** (cliff-blind rate 0.99–1.00 across all three datasets).

A caveat the analysis can't separate without more work: this metric uses a fixed 0.7 threshold for both Tanimoto (binary FPs) and cosine (neural FPs). Continuous embeddings with non-negative features have a higher baseline cosine even between unrelated molecules, so part of the neural cliff-blind rate is a scale artifact. There's no clean fix — rank-based metrics (cliff percentile within the FP's own pairwise distribution, PR-AUC of cliff-vs-non-cliff separation) are scale-invariant *within* a dataset, but the threshold they imply is set by the dataset's composition and doesn't transfer. A scaffold-diverse library and a congeneric series will produce very different "top 5%" cutoffs even for the same FP. The honest reading is that for neural FPs there is no single, dataset-portable "is similar" threshold to recommend — you have to calibrate per use case.

The D3 receptor violins make the scale issue stark:

![D3 cliff violins](figures/cliffs/01_cliff_similarity_violins_CHEMBL234_Ki.png)

Median cliff-pair similarity on D3: Morgan 0.43, TopTorsion 0.49, Avalon 0.59, RDKit-topo 0.60, AtomPair 0.60, MACCS 0.77, **CheMeleon 0.90, MIST 0.95**. Even taking the cosine-vs-Tanimoto scale difference into account, the neural FPs leave very little room for similarity to drop on cliffs — their dynamic range over D3's molecule space is roughly 0.7–1.0.

Practical implication for retrieval-style workflows on neural FPs: **a kNN search on CheMeleon or MIST cosine cannot rule out 100× potency differences in its top hits**. Whether that's a fundamental limitation or just a similarity-scale calibration question is the open follow-up (see Gotchas).

### What kinds of changes does each fingerprint actually see?

Per-fingerprint deep-dive figures are in [`figures/cliffs/`](figures/cliffs/) — one per FP showing its top-3 most cliff-blind and top-3 most cliff-aware molecule pairs across all three datasets, with non-MCS atoms highlighted. Reading those figures by eye:

- **Morgan**: shrugs at single-atom changes within a shared core (graph distance 1–2): one ring nitrogen swapped, a methyl shifted, a halogen swapped. It pulls similarity down once enough atom-environment bits change — typically when the substituent pattern around the scaffold reshuffles. Sensitive to: scaffold identity, halogen-vs-methyl swaps, substituent connectivity at radius 2. Insensitive to: changes confined to one or two atoms inside a polycyclic core.
- **TopTorsion**: similar profile to Morgan but with a tighter dynamic range — its "most cliff-aware" floor sits around 0.33–0.41 instead of Morgan's 0.11–0.33. Sensitive to: atom-quadruplet (4-atom path) changes that reshape long aliphatic chains. Insensitive to: localized atom swaps that preserve most 4-atom path patterns.
- **AtomPair**: cliff-blind rate varies widely across targets (0.24 on D3, 0.76 on Thrombin). Sensitive to: changes that alter the distance between key atoms. Insensitive to: positional swaps at the same path distance — a heteroatom shift that preserves atom-pair distances will still register high.
- **RDKit-topo**: scores most cliffs near or above 0.7 even when the molecules look visibly different — its hashed-paths approach over-rewards shared connectivity. Limited dynamic range (Thrombin median 0.86). Sensitive to: scaffold replacement. Insensitive to: heteroatom swaps, substituent decoration, and even some ring-size changes.
- **Avalon**: similar shape to RDKit-topo but with a slightly broader range. Sensitive to: scaffold change, ring fusion changes. Insensitive to: small decorative changes.
- **MACCS**: routinely returns sim = 1.00 for cliff pairs because 167 bits is too coarse to distinguish many small-change pairs — they end up with identical bit vectors. Sensitive to: presence/absence of specific predefined substructures (carboxylic acid, halogens, particular ring systems). Insensitive to: anything that doesn't add or remove a key listed group.
- **CheMeleon**: cliff-aware floor is ~0.71 even on its best examples — it cannot score true cliffs as anything but "similar." Its similarity scale is calibrated for "same chemotype family." Sensitive to: scaffold-class changes. Insensitive to: any structural change short of a full scaffold rewrite.
- **MIST-28M**: the most compressed scale of all 8 — top-3 most cliff-aware on D3 are 0.71–0.75. Sensitive to: gross molecular character. Insensitive to: anything finer than that.

The pattern across the eight: **dynamic range correlates with cliff resolution**. Morgan's similarity distribution spans 0.0–1.0 over diverse pairs and gives it room to drop low on cliffs; MIST's distribution is squeezed into roughly 0.7–1.0 over the same molecule space, so it can't distinguish "structurally similar but functionally different" from "structurally similar and functionally similar."

### CheMeleon and MIST are different in other ways though

Lumping them as "the neural FPs" still misleads on every probe except cliffs:

- **CheMeleon has the highest local property alignment on AqSolDB logS** ([figure](figures/scaffolds/02_purity_vs_property_pareto_k5.png), kNN R² ≈ 0.76 at k=5) at moderate scaffold purity (~0.65). MIST sits in the middle of the pack on R² (~0.69) and has the lowest scaffold purity of all 8 FPs (~0.55). Read together, the figure shows CheMeleon is on the property-aware Pareto frontier; MIST is the most scaffold-hopping FP but doesn't translate that into the strongest property neighborhoods.
- **MIST is the most isolated in the agreement matrix** (RV ~0.15–0.30 with most other FPs); CheMeleon sits closer to the classical block (RV 0.40–0.66).

Both are heavily cliff-blind, but they reach the same problem from different directions: MIST's geometry is genuinely different from classical FPs (and that broad geometry can't resolve cliffs); CheMeleon's geometry is much more chemistry-aware but uses a similarity scale tuned for "this molecule is in the same chemotype family", not "this molecule has the same activity."

### Property gradients are visible in neural FP geometry, blob-like in classical FPs

UMAP of each FP on 9980 AqSolDB molecules, colored by logS:

![logS UMAP per fingerprint](figures/adme/01_umap_solubility_aqsoldb.png)

Read by eye:

- **CheMeleon and MIST** show the clearest property gradients — soluble (yellow) and insoluble (purple) molecules occupy visibly different regions of the embedding, with smooth transitions through green / teal in between. This is consistent with their high kNN R² on logS.
- **Classical FPs** (Morgan, RDKit-topo, Avalon, AtomPair, TopTorsion, MACCS) form scaffold-driven blobs with logS values mixed within each blob. The structural classes are real, but logS is not the axis along which they're separated. RDKit-topo and MACCS in particular fragment into many small islands.
- **MIST shows the most spread-out, less-clustered geometry**, consistent with it having the lowest scaffold purity. CheMeleon's clusters are tighter than MIST's but looser than the classical FPs'.

The lipophilicity (logD7.4) UMAP in [`figures/adme/`](figures/adme/) tells a similar story but with weaker overall gradients — logD has less dynamic range over the AstraZeneca set than logS has over AqSolDB. The BBB binary-classification UMAP is in the same folder for completeness.

### Classical FPs are also not interchangeable

The pair plots already hinted at this — even on four hand-picked pairs the classical FPs disagree visibly:

![drug pair similarity summary](figures/pairs/drug/03_within_pair_distance_summary.png)

- **Morgan** is the most idiosyncratic and the best at recognizing cliffs as different (cliff-blind rate 0.08–0.41 across the three targets; median cliff similarity 0.43 on D3). Its highest off-diagonal in the RV matrix is only 0.57. Atom-environment hashing preserves both scaffold and decoration in a way path-based FPs don't.
- **Avalon** (512 bits) is competitive with the 2048-bit FPs on scaffold purity (~0.68 on AqSolDB) but middling on property R² (~0.64 on logS). On cliffs its median similarity is high (0.59 on D3) and cliff-blind rate ≥ 0.96 on the harder targets — like the other path-based FPs, it has trouble distinguishing methyl-vs-OMe-style cliffs.
- **MACCS** is short (167 bits) and fast. The cliff-blind rate is the highest among the classical FPs (0.81 / 0.98 / 0.96 across D3 / Thrombin / GSK-3β) because 167 bits don't have the resolution to distinguish small-change cliffs.
- **RDKit-topo, AtomPair, TopTorsion** look similar on paper. They aren't (RV 0.28–0.62 with each other). On cliffs their behavior diverges substantially across targets: TopTorsion is the second-best after Morgan on Thrombin and GSK-3β (cliff-blind 0.51 / 0.46) but only third-best on D3. AtomPair ranges from 0.24 on D3 to 0.76 on Thrombin. RDKit-topo is the worst of the three on the harder targets (0.90 / 0.90 cliff-blind on Thrombin / GSK-3β).

## Gotchas

**General:**

- **Don't bake one FP into how you define your test.** An earlier version of the cliff probe used "Morgan ≥ 0.9 with large Δy" or "kNN top-k" definitions; both reduced fingerprint comparison to "agreement with whichever FP we used to define the cliff set." The current cliff probe defines pairs from molecular graph properties only (graph_distance ≤ 5, |ΔpKi| ≥ 2.0).
- **Local geometry ≠ global geometry.** All FPs show useful local kNN signal but only modest global Spearman ρ — see the right panel below, ρ ≤ 0.25 everywhere on logS:

  ![logS alignment](figures/adme/03_alignment_solubility_aqsoldb.png)

  Methods that rely on global geometry (UMAP-then-cluster, Spearman over all pairs) amplify whatever idiosyncrasy each FP has.
- **Scaffold-purity needs scaffold-repeat-rich data.** Random ChEMBL is so scaffold-diverse (4490 unique scaffolds per 5000 molecules) that purity-at-k is mostly noise. AqSolDB has dense repeats and is the right venue.
- **Three ChEMBL targets is a small base for "the cliff hierarchy is consistent."** D3 (n=730 cliffs), Thrombin (n=475), and GSK-3β (n=128) are the three we ran. Per-target cliff-blind rates carry sampling variance, especially on GSK-3β. The aggregated MoleculeACE (~30 targets) sweep is the planned follow-up.

**Per-fingerprint:**

- **MACCS** has only 167 bits — distinct molecules collide more often, and small-change activity cliffs almost always come out high-similarity (cliff-blind rate 0.81 on D3, 0.96–0.98 on Thrombin / GSK-3β).
- **MIST and CheMeleon cosine scores are NOT on the same scale as Tanimoto, and there is no fixed cross-dataset threshold to substitute.** Cosine 0.4 between two MIST embeddings does not mean what 0.4 Tanimoto means. The cliff-pair probe makes this concrete: MIST median similarity over D3 cliff pairs is 0.95 (Morgan's is 0.43). Tanimoto rules of thumb ("≥ 0.7 = similar") simply do not transfer. Rank-based alternatives (percentile within a dataset's own pairwise-similarity distribution, PR-AUC of cliff-vs-non-cliff separation) are scale-invariant within a dataset but their cutoffs are set by that dataset's composition — a scaffold-diverse library and a congeneric series produce very different "top 5%" thresholds. So the practical guidance is: for binary FPs, Tanimoto thresholds carry across datasets reasonably well; for neural FPs, calibrate per use case using a small held-out set with the property you actually care about.
- **CheMeleon similarity is "chemistry-aware" but not biology-aware.** It scores activity cliffs at median similarity 0.90 on D3 — chemically the molecules in a cliff pair really are very similar, and CheMeleon agrees. It just doesn't know that small chemical change can imply huge potency change.
- **Morgan's "two molecules share a key motif" failure mode** is real: two compounds can both light up the same Morgan bit while differing globally. The drug-sized pair heatmap shows this — 0.75 Tanimoto between aniline-quinazoline and gefitinib, driven entirely by their shared core:

  ![Morgan drug-sized pair heatmap](figures/pairs/drug/01_pair_similarity_morgan.png)

  Conversely, on properly-defined cliffs (small change → big activity change) Morgan is the best discriminator we tested — its bit-vector resolution captures small modifications well.

## Practical takeaway

Pick the fingerprint for the question:

- **Threshold portability across datasets** → Tanimoto on binary FPs (Morgan, RDKit-topo, Avalon, AtomPair, TopTorsion, MACCS) is roughly comparable across datasets — a "≥ 0.7 = similar" rule developed on one ChEMBL target carries forward to another with similar meaning. Cosine on neural FPs (CheMeleon, MIST) does not — the same numeric threshold means different things on different datasets, and rank-based fixes (percentiles, PR-AUC) are also dataset-bound. If you need a portable "is this similar?" decision rule, prefer a binary FP. If you need neural FPs, plan to calibrate the threshold per use case.
- **Activity-cliff-sensitive retrieval** on a known target → Morgan first; TopTorsion as a backup on harder targets. Across the three ChEMBL targets we tested, Morgan was the only FP with cliff-blind rate < 50% on every target (0.08 / 0.41 / 0.20). Avoid neural FPs and MACCS for this kind of retrieval.
- **Local property-aware lookup** (find similar molecules, hope their property values are informative) → CheMeleon's geometry shows the strongest property gradient on AqSolDB logS in the UMAP and the highest kNN R². The cliff blindness doesn't hurt as much when the property is smoothly distributed (no single methyl-swap is going to flip logS by 100×). Note: this is a structural-alignment observation, not a benchmarked predictor — if you need a real ADME model, train one on top.
- **Chemotype discovery / clustering** in a drug-like library → if natural clusters exist at all, MACCS and Avalon are most likely to surface them; MIST is the least scaffold-anchored and least likely to. Note the clustering experiment under `figures/clustering/` — random ChEMBL is too scaffold-diverse to cluster meaningfully even with the most scaffold-anchored FP.
- **Feature input to a supervised neural model** → MIST's distinctness from classical FPs may be the point. The downstream model can re-learn cliff structure from labels.
- **One general-purpose FP** → no single winner. Morgan handles cliffs and pair-level similarity well but its UMAP geometry is blob-like with respect to ADME properties. CheMeleon has the smoothest property geometry but is heavily cliff-blind. Use Morgan for retrieval, CheMeleon for property-similarity lookup.
- **Ensembling** → Morgan paired with a neural FP (CheMeleon or MIST) gives the most distinct views in the RV matrix. The two have very different similarity geometries and different failure modes, which is what you want from an ensemble.

## Reproducing

```bash
uv sync
uv run python scripts/figure_pairs.py
uv run python scripts/figure_clustering.py
uv run python scripts/figure_adme.py
uv run python scripts/figure_adme_alignment.py --mode sweep
uv run python scripts/figure_adme_alignment.py --mode bars --k 5
uv run python scripts/figure_adme_families.py
uv run python scripts/figure_agreement.py
uv run python scripts/figure_scaffolds.py
uv run python scripts/figure_cliffs.py
```

Figures land under `figures/<topic>/`.
