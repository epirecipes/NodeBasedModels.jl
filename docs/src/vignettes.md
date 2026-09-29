# Vignettes

The vignettes are Quarto pages under `vignettes/` in the repository. They are rendered with the
strict scenario cache (`NETEPI_STRICT_CACHE=1`), so a missing or stale reference ensemble is an
error. Every number on a page is printed by code on that page, and Lean is cited only from
NetworkEpiCore's `CITABLE.txt`. N01's shared first cell is the same as EdgeBasedModels' E01,
apart from the back-end lines.

- [From a reaction network to a pairwise model](vignettes/N01_reaction_network_to_pairwise/index.html): SIR written as a Catalyst network, lifted to a heterogeneous pairwise model on Poisson(5) and 6-regular networks, and checked against NetworkOutbreaks
- [Closures and degree heterogeneity](vignettes/N02_closures_degree_heterogeneity/index.html): the constant-K closure, the PGF closure and the S-anchored pairwise model on five degree distributions with the same R₀, against NetworkOutbreaks and the edge-based model
- [Natural history in pairwise models](vignettes/N03_natural_history/index.html): latency, Erlang stages, branching with two infectors, two strains, vaccination and waning immunity, on Poisson(5), against NetworkOutbreaks and the edge-based model
- [Graph-level models: individual- and pair-based](vignettes/N04_graph_level/index.html): individual-based (NIMFA) and pair-based (Kirkwood) SIR models on the very graph the simulation used, their exactness on trees, and a typed stochastic block model instance
- [Clustering](vignettes/N05_clustering/index.html): Keeling's and Barnard's clustered pair closures on networks with triangles, against NetworkOutbreaks and the exact edge-based model of Volz
- [Back to mass action](vignettes/N06_mass_action/index.html): the mean-field closure on a well-mixed population, the individual-based model on the complete graph, and dense Poisson networks approaching mass action
- [Thresholds and R₀: SIR versus SIS](vignettes/N07_thresholds/index.html): the epidemic thresholds of the pairwise SIR and SIS models, the generation-based R₀ and the growth-based threshold ratio
- [Why pairwise composition is lax](vignettes/N08_pairwise_composition/index.html): glue the syntax first, then lift; the pairwise field of a glued model is not the sum of the parts' fields
- [SIS: pairwise and reinfection counting](vignettes/N09_sis_reinfection/index.html): the SIS pairwise model, the reinfection-counting refinement with L = 0…4, and both against NetworkOutbreaks
- [Motif closures](vignettes/N10_motif_closures/index.html): SIS motif closures of order m = 2, 3, 4 on a 3-regular network against NetworkOutbreaks
- [Neighbourhood model (n = 2)](vignettes/N11_neighbourhood/index.html): SIS by the number of infected neighbours of each node, on a 3-regular network, against NetworkOutbreaks
- [All SIS approximations against simulation](vignettes/N12_sis_comparison/index.html): pairwise, reinfection-counting, motif and neighbourhood models of SIS on a 3-regular network against one NetworkOutbreaks ensemble
- [Motif catalogue and validation methodology](vignettes/N13_motif_catalogue_validation/index.html): how the NodeBasedModels pages are validated against NetworkOutbreaks, every declared verdict checked in one table, the N-scaling protocol, and an appendix listing the motif shapes
- [Multitype pairwise models](vignettes/N14_multitype_pairwise/index.html): population-level pairwise models of stratified SIR on typed configuration networks, the typed morphism to the edge-based model, and NetworkOutbreaks

The pages are also rendered as Markdown (`vignettes/<page>/index.md`) and PDF.
