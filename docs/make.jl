using Documenter
using NodeBasedModels

DocMeta.setdocmeta!(NodeBasedModels, :DocTestSetup, :(using NodeBasedModels);
                    recursive = true)

makedocs(
    sitename = "NodeBasedModels.jl",
    modules  = [NodeBasedModels],
    authors  = "Simon Frost",
    repo     = "https://github.com/epirecipes/NodeBasedModels.jl/blob/{commit}{path}#{line}",
    format   = Documenter.HTML(;
        canonical    = "https://epirecip.es/NodeBasedModels.jl/",
        edit_link    = "main",
        assets       = String[],
        prettyurls   = false,
        size_threshold = nothing,
    ),
    pages = [
        "Home"      => "index.md",
        "Vignettes" => "vignettes.md",
        "API"       => "api.md",
    ],
    checkdocs = :none,
    warnonly  = true,
)
