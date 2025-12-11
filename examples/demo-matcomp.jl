using Random
using LinearAlgebra
using ProximalOperators
using ProxTV
using NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

include("plot-utils-matcomp.jl")

Random.seed!(12345)

function demo_solver(f, sol, h, suffix="")
  options = ROSolverOptions(ν=1.0, β=1e16, ϵa=1e-3, ϵr=1e-3, verbose=1, neg_tol=1e-2)

  @info "using iR2N with $(suffix)"
  reset!(f)
  iR2N_out = iR2N(f, h, options, x0=f.meta.x0)
  @info "iR2N relative error: $(norm(iR2N_out.solution - sol) / norm(sol))"
  plot_matcomp(iR2N_out, sol, "ir2n-$(suffix)")
end

function demo_matcomp(m=8, n=12, r=5)
  model, nls_model, sol = random_matrix_completion_model(m=m, n=n, r=r)
  f = LBFGSModel(model)
  p = 1.1
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 2

  # with "exact" prox, meaning we use the absolute criterion δ ≤ dualGap
  context_tvp = ProxTVContext(model.meta.nvar, :tvp, p, κs=1e-5, λ=λ, dualGap=eps(), use_absolute_criterion=true)
  hp_tvp = NormTVp(λ, p, context_tvp)
  demo_solver(f, sol, hp_tvp, "exact-prox")

  # with "inexact" prox, meaning we use the relative criterion ||s||≥ κs * bound_on_s
  context_tvp = ProxTVContext(model.meta.nvar, :tvp, p, κs=1e-5, λ=λ, dualGap=eps())
  hp_tvp = NormTVp(λ, p, context_tvp)
  demo_solver(f, sol, hp_tvp, "inexact-prox")
end

demo_matcomp()