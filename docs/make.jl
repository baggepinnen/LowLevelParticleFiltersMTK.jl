using Documenter
using LowLevelParticleFiltersMTK

makedocs(
    sitename = "LowLevelParticleFiltersMTK Documentation",
    doctest = false,
    modules = [LowLevelParticleFiltersMTK],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
    warnonly = [:docs_block, :missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/baggepinnen/LowLevelParticleFiltersMTK.jl.git",
    push_preview = false,
)
