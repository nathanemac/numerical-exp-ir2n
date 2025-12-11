using PGFPlotsX
using LaTeXStrings
using Printf

include("plot-utils-bpdn-latex.jl")

Random.seed!(12345)
function run_bpdn_solvers(compound=1)
  model, nls_model, sol = bpdn_model(compound)
  f = LBFGSModel(model)
  p = 1.1
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 2
  options = ROSolverOptions(ν=1.0, β=1e16, ϵa=1e-6, ϵr=1e-6, verbose=1)
  # EXACT 
  context_exact = ProxTVContext(model.meta.nvar, :lp, p, κs=1e-7, λ=λ, dualGap=eps(), use_absolute_criterion=true)
  hp_exact = NormLp(λ, p, context_exact)
  reset!(f)
  out_exact = iR2N(f, hp_exact, options, x0=f.meta.x0)
  # INEXACT 
  context_inexact = ProxTVContext(model.meta.nvar, :lp, p, κs=1e-7, λ=λ, dualGap=eps())
  hp_inexact = NormLp(λ, p, context_inexact)
  reset!(f)
  out_inexact = iR2N(f, hp_inexact, options, x0=f.meta.x0)
  return out_exact, out_inexact, sol
end

out_exact, out_inexact, sol = run_bpdn_solvers(1)
plot_bpdn_compare_tikz(out_exact, out_inexact, sol; name="bpdn-exact-vs-inexact")
