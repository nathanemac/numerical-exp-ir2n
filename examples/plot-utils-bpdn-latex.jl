ENV["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" * ENV["PATH"]

using PyPlot
using LinearAlgebra, Printf

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", serif="Computer Modern Roman")
PyPlot.rc("text.latex", preamble=raw"\usepackage[T1]{fontenc}\usepackage{amsmath}")

# ======== AJOUT POUR AGRANDIR LES POLICES ========
PyPlot.rc("font", size=16)
PyPlot.rc("axes", titlesize=18, labelsize=16)
PyPlot.rc("xtick", labelsize=14)
PyPlot.rc("ytick", labelsize=14)
PyPlot.rc("legend", fontsize=14)
# ==================================================
using LinearAlgebra
using Printf

using PGFPlotsX
using LaTeXStrings
using Printf

# x ↦ table (i, x_i) pour PGFPlotsX
function vector_to_table(x::AbstractVector)
  n = length(x)
  idx = collect(1:n)
  Table(["x" => idx, "y" => collect(x)])
end


function plot_bpdn_compare_tikz(out_exact, out_inexact, sol; name="bpdn-exact-vs-inexact")
  x_exact = out_exact.solution
  x_inexact = out_inexact.solution

  # erreurs relatives (comme dans ta version PyPlot)
  rel_exact = norm(x_exact .- sol) / norm(sol)
  rel_inexact = norm(x_inexact .- sol) / norm(sol)
  rel_exact_str = @sprintf("%.2e", rel_exact)
  rel_inexact_str = @sprintf("%.2e", rel_inexact)

  # mêmes chaînes que dans la figure PyPlot
  label_exact = "iR2N (error = \\($rel_exact_str\\))"
  label_inexact = "iR2N (error = \\($rel_inexact_str\\))"

  # Tables (x = index, y = valeur) pour PGFPlotsX
  tbl_exact = vector_to_table(x_exact)
  tbl_inexact = vector_to_table(x_inexact)
  tbl_sol = vector_to_table(sol)

  tikz = @pgf GroupPlot(
    {
      group_style = {
        group_size = "2 by 1",
        horizontal_sep = "2.5cm",
      },
      "axis on top",
      enlargelimits = "false",
      title_style = "{font=\\Large}",
      ticklabel_style = "{font=\\large}",
      label_style = "{font=\\large}",
    },
    Axis(
      {
        title = raw"Exact prox (\(\kappa_s = 10^{-7}\))",
        xlabel = "index",
        ylabel = "signal",
        legend_pos = "south west",
        xmin = 1,
        xmax = length(sol),
      },
      Plot(
        {
          "no marks",
          "line width=1.3pt",
          "color=blue",
        },
        tbl_exact,
      ),
      Plot(
        {
          "no marks",
          "line width=1.0pt",
          "color=black",
          "dashed",
        },
        tbl_sol,
      ),
      LegendEntry(label_exact),
      LegendEntry("true solution"),
    ),
    Axis(
      {
        title = "Inexact prox",
        xlabel = "index",
        ylabel = "signal",
        legend_pos = "south west",
        xmin = 1,
        xmax = length(sol),
      },
      Plot(
        {
          "no marks",
          "line width=1.3pt",
          "color=red",
        },
        tbl_inexact,
      ),
      Plot(
        {
          "no marks",
          "line width=1.0pt",
          "color=black",
          "dashed",
        },
        tbl_sol,
      ),
      LegendEntry(label_inexact),
      LegendEntry("true solution"),
    ),
  )
  PGFPlotsX.save("$(name).tikz", tikz)
  println("Saved $(name).tikz")
end

