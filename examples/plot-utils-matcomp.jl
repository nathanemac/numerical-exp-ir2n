using PGFPlots
using TikzPictures
TikzPictures.tikzCommand("/Library/TeX/texbin/lualatex")

function plot_matcomp(outstruct, sol, name="tr-qr")
  #Comp_pg = outstruct.solver_specific[:SubsolverCounter]
  objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
  x = outstruct.solution
  a = Axis(
    [
      Plots.Linear(
        1:length(x),
        x;
        style="ultra thick, color=blue",
        mark="none",
        legendentry="computed",
      ),
      Plots.Linear(
        1:length(sol),
        sol;
        style="densely dashed, color=black",
        mark="none",
        legendentry="exact",
      ),
    ],
    xlabel="index",
    ylabel="signal",
    legendStyle="at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
  )
  save("matcomp-$(name).pdf", a)

  # b = Axis(
  #   Plots.Linear(1:length(Comp_pg), Comp_pg, mark="none"),
  #   xlabel="outer iterations",
  #   ylabel="inner iterations",
  #   ymode="log",
  # )
  # save("bpdn-inner-outer-$(name).pdf", b)

  c = Axis(
    Plots.Linear(1:length(objdec), objdec, mark="none"),
    xlabel="\$ k^{th}\$  \$ \\nabla f \$ Call",
    ylabel="Objective Value",
    ymode="log",
  )
  save("matcomp-objdec-$(name).pdf", c)

  d = Axis(
    Plots.Linear(1:length(x), abs.(x - sol), mark="none"),
    xlabel="index",
    ylabel="error \$|x - x^*|\$",
  )
  save("matcomp-error-$(name).pdf", d)
end