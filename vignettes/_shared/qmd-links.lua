-- Owner: V-NBM-infra. Rewrite relative links to other vignette pages (`N01_.../index.qmd`, `../index.qmd`) to the
-- rendered file of the current format: `.md` for gfm, `.html` otherwise (the pdf links point at the html pages).
-- A default-type Quarto project does not do this itself (only websites and books do); without the filter the
-- rendered index.md and index.html would link the .qmd sources. Same behaviour as the EdgeBasedModels.jl filter.
function Link(el)
  if el.target:match("^%a[%w+.-]*:") then return nil end   -- absolute URL (http:, mailto:, ...): leave alone
  local ext = quarto.doc.is_format("gfm") and ".md" or ".html"
  local new, n = el.target:gsub("%.qmd(#?.*)$", ext .. "%1")
  if n > 0 then el.target = new; return el end
end
