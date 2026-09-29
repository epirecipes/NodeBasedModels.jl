# NodeBasedModels.jl Vignettes


- [Overview](#overview)
- [Conventions shared by every page](#conventions-shared-by-every-page)
- [Pages](#pages)
- [Environment](#environment)
- [Rendering](#rendering)
- [References](#references)

## Overview

[NodeBasedModels.jl](https://github.com/epirecipes/NodeBasedModels.jl)
builds node-based deterministic models of epidemics on networks:
population-level pairwise models with a choice of triple closure
([Keeling 1999](#ref-keeling1999); [Eames and Keeling
2002](#ref-eames2002); [House and Keeling 2011](#ref-house2011)),
heterogeneous and S-anchored pairwise models, clustered closures
([Barnard et al. 2019](#ref-barnard2019)), motif and neighbourhood
approximations for SIS dynamics ([Hadjichrysanthou et al.
2012](#ref-hadjichrysanthou2012); [Ritchie et al.
2016](#ref-ritchie2016)), and individual- and pair-based models on a
fixed graph ([Sharkey 2008](#ref-sharkey2008), [2011](#ref-sharkey2011);
[Sharkey et al. 2015](#ref-sharkey2015)). The textbook treatment is Kiss
et al. ([2017](#ref-kiss2017)).

Models are written once, as a reaction network (Catalyst.jl), an ODE
system (ModelingToolkit.jl) or with the direct constructor of
NetworkEpiCore.jl, and turned into a `ContactModel` with
`contact_model`; `node_based` then lifts that model onto a network
descriptor. The same `ContactModel` is lifted to an edge-based model by
EdgeBasedModels.jl and simulated by NetworkOutbreaks.jl, so every page
compares against the same committed stochastic reference summaries
(`scenario_summary(sc)`) as the mirrored EdgeBasedModels.jl page.

## Conventions shared by every page

- Every model is shown twice: first built from the low-level front end,
  then with the factory, and the two vector fields are asserted equal.
- Every simulation comparison loads the committed NetworkOutbreaks
  summary of a registered scenario, draws it with the shared
  `comparisonplot` and tabulates it with `compare`, and states N, the
  number of runs, the conditioning rule and P(major). Spread bands are
  the pointwise q2.5–q97.5 of the runs; the comparison band is the mean
  ±1.96 SE.
- Every number in the prose is printed by code on the same page.
- Lean results are cited only by the theorem names listed in
  `NetworkEpiCore.jl/proofs/CITABLE.txt`.
- The canonical anchors are R₀ = 2, γ = 1/4 and a per-contact rate τ =
  1/6, unless a scenario says otherwise.

## Pages

| \# | Page | Scenarios | Mirrors (EdgeBasedModels.jl) |
|----|----|----|----|
| N01 | [From a reaction network to a pairwise model](N01_reaction_network_to_pairwise/index.md) | `:sir_pois5`, `:sir_reg6` | E01 |
| N02 | [Closures and degree heterogeneity](N02_closures_degree_heterogeneity/index.md) | `:sir_reg6`, `:sir_pois5`, `:sir_nb4`, `:sir_bim`, `:sir_pl`, `:sir_pois5_N1000`, `:sir_pois5_N100000`, `:sir_bim_N1000`, `:sir_bim_N100000`, `:sir_pl_N1000`, `:sir_pl_N100000`, `:seair_pois5` | E03, E06 |
| N03 | [Natural history in pairwise models](N03_natural_history/index.md) | `:seir_pois5`, `:sir_erl3_pois5`, `:seair_pois5`, `:twostrain_pois5`, `:sir_vax_pois5`, `:sirs_pois5` | E04 |
| N04 | [Graph-level models: individual- and pair-based](N04_graph_level/index.md) | `:sir_reg6_fixed`, `:sir_sbm2` | E10 |
| N05 | [Clustering](N05_clustering/index.md) | `:sir_clust_s2t2`, `:sir_clust_pois12`, `:sir_clust_s2t2_N1000`, `:sir_clust_s2t2_N100000`, `:sir_reg6` | E08 |
| N06 | [Back to mass action](N06_mass_action/index.md) | `:sir_wm5`, `:sir_dense_pois5`, `:sir_dense_pois20`, `:sir_dense_pois100` | E05 |
| N07 | [Thresholds and R₀: SIR versus SIS](N07_thresholds/index.md) | `:sir_reg6`, `:sis_reg3` | – |
| N08 | [Why pairwise composition is lax](N08_pairwise_composition/index.md) | `:seir_pois5`, `:sir_sbm2` | E12 |
| N09 | [SIS: pairwise and reinfection counting](N09_sis_reinfection/index.md) | `:sis_reg3` | – |
| N10 | [Motif closures](N10_motif_closures/index.md) | `:sis_reg3` | – |
| N11 | [Neighbourhood model (n = 2)](N11_neighbourhood/index.md) | `:sis_reg3` | – |
| N12 | [All SIS approximations against simulation](N12_sis_comparison/index.md) | `:sis_reg3` | E13 |
| N13 | [Motif catalogue and validation methodology](N13_motif_catalogue_validation/index.md) | `:sir_pois5` | E14 |
| N14 | [Multitype pairwise models](N14_multitype_pairwise/index.md) | `:sir_sbm2`, `:sir_age2` | – |

14 of 14 pages are available. N14 is the stretch page on
population-level multitype pairwise models.

## Environment

| Package             | Version |
|---------------------|---------|
| NetworkEpiCore.jl   | 0.1.0   |
| NodeBasedModels.jl  | 0.2.0   |
| EdgeBasedModels.jl  | 0.2.0   |
| NetworkOutbreaks.jl | 0.2.0   |

Julia 1.12.7.

## Rendering

``` bash
cd NodeBasedModels.jl/vignettes
julia --project=. -e 'using Pkg; Pkg.instantiate()'
_shared/render.sh          # html + pdf, then gfm, with NETEPI_STRICT_CACHE=1; fails on warnings or missing summaries
julia --project=. _shared/check_pages.jl --rendered
```

The rendered pages and the `_freeze/` directory are kept with the
sources, so a page is only re-executed when its source changes. This
index is frozen too, but it reads the page catalogue and the package
versions, which can change while `index.qmd` does not, so re-render it
locally (`quarto render index.qmd`) when pages are added or versions
change. Continuous integration builds the HTML from `_freeze/` without
running Julia.

Render through `_shared/render.sh` rather than a bare `quarto render`. A
single render of all three formats deletes each page’s `index_files/`
directory once the self-contained HTML is written, which also deletes
the figures that the GitHub-Markdown `index.md` links to. The script
therefore renders html and pdf first and gfm last, and
`check_pages.jl --rendered` fails if an image in an `index.md` points to
a missing file.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-barnard2019" class="csl-entry">

Barnard, Rosanna C., Luc Berthouze, Péter L. Simon, and István Z. Kiss.
2019. “Epidemic Threshold in Pairwise Models for Clustered Networks:
Closures and Fast Correlations.” *Journal of Mathematical Biology* 79
(3): 823–60. <https://doi.org/10.1007/s00285-019-01380-1>.

</div>

<div id="ref-eames2002" class="csl-entry">

Eames, Ken T. D., and Matt J. Keeling. 2002. “Modeling Dynamic and
Network Heterogeneities in the Spread of Sexually Transmitted Diseases.”
*Proceedings of the National Academy of Sciences* 99 (20): 13330–35.
<https://doi.org/10.1073/pnas.202244299>.

</div>

<div id="ref-hadjichrysanthou2012" class="csl-entry">

Hadjichrysanthou, Christoforos, Mark Broom, and István Z. Kiss. 2012.
“Approximating Evolutionary Dynamics on Networks Using a Neighbourhood
Configuration Model.” *Journal of Theoretical Biology* 312: 13–21.
<https://doi.org/10.1016/j.jtbi.2012.07.015>.

</div>

<div id="ref-house2011" class="csl-entry">

House, Thomas, and Matt J. Keeling. 2011. “Insights from Unifying Modern
Approximations to Infections on Networks.” *Journal of the Royal Society
Interface* 8 (54): 67–73. <https://doi.org/10.1098/rsif.2010.0179>.

</div>

<div id="ref-keeling1999" class="csl-entry">

Keeling, Matt J. 1999. “The Effects of Local Spatial Structure on
Epidemiological Invasions.” *Proceedings of the Royal Society of London.
Series B* 266 (1421): 859–67. <https://doi.org/10.1098/rspb.1999.0716>.

</div>

<div id="ref-kiss2017" class="csl-entry">

Kiss, István Z., Joel C. Miller, and Péter L. Simon. 2017. *Mathematics
of Epidemics on Networks: From Exact to Approximate Models*. Vol. 46.
Interdisciplinary Applied Mathematics. Springer.
<https://doi.org/10.1007/978-3-319-50806-1>.

</div>

<div id="ref-ritchie2016" class="csl-entry">

Ritchie, Martin, Luc Berthouze, and István Z. Kiss. 2016. “Beyond
Clustering: Mean-Field Dynamics on Networks with Arbitrary Subgraph
Composition.” *Journal of Mathematical Biology* 72 (1–2): 255–81.
<https://doi.org/10.1007/s00285-015-0884-1>.

</div>

<div id="ref-sharkey2008" class="csl-entry">

Sharkey, Kieran J. 2008. “Deterministic Epidemiological Models at the
Individual Level.” *Journal of Mathematical Biology* 57 (3): 311–31.
<https://doi.org/10.1007/s00285-008-0161-7>.

</div>

<div id="ref-sharkey2011" class="csl-entry">

Sharkey, Kieran J. 2011. “Deterministic Epidemic Models on Contact
Networks: Correlations and Unbiological Terms.” *Theoretical Population
Biology* 79 (4): 115–29. <https://doi.org/10.1016/j.tpb.2011.01.004>.

</div>

<div id="ref-sharkey2015" class="csl-entry">

Sharkey, Kieran J., István Z. Kiss, Robert R. Wilkinson, and Péter L.
Simon. 2015. “Exact Equations for SIR Epidemics on Tree Graphs.”
*Bulletin of Mathematical Biology* 77 (4): 614–45.
<https://doi.org/10.1007/s11538-013-9923-5>.

</div>

</div>
