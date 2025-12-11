using Random
using LinearAlgebra
using ProximalOperators
using ProxTV
using NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems
using DataFrames
using CSV

Random.seed!(12345)

smooth_objective_results = DataFrame(κs=Float64[], smooth_objective=Float64[], relative_error=Float64[])

model, nls_model, sol = random_matrix_completion_model(m=8, n=12, r=5)
f = LBFGSModel(model)
p = 1.1
λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 2
options = ROSolverOptions(ν=1.0, β=1e16, ϵa=1e-3, ϵr=1e-3, verbose=1, neg_tol=1e-2)

# inexact prox
for κs in [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.5, 0.9, 0.99]
  context_tvp = ProxTVContext(model.meta.nvar, :tvp, p, κs=κs, λ=λ, dualGap=eps())
  hp_tvp = NormTVp(λ, p, context_tvp)
  ir2n_out = iR2N(f, hp_tvp, options, x0=f.meta.x0)
  @info "κs = $κs, relative error: $(norm(ir2n_out.solution - sol) / norm(sol))"
  relative_error = norm(ir2n_out.solution - sol) / norm(sol)
  push!(smooth_objective_results, (κs, ir2n_out.solver_specific[:smooth_obj], relative_error))
end

# exact prox
context_tvp = ProxTVContext(model.meta.nvar, :tvp, p, κs=0.99999, λ=λ, dualGap=eps(), use_absolute_criterion=true)
hp_tvp = NormTVp(λ, p, context_tvp)
ir2n_out = iR2N(f, hp_tvp, options, x0=f.meta.x0)
@info "exact prox, relative error: $(norm(ir2n_out.solution - sol) / norm(sol))"
relative_error = norm(ir2n_out.solution - sol) / norm(sol)
push!(smooth_objective_results, (10, ir2n_out.solver_specific[:smooth_obj], relative_error))
CSV.write("smooth-objective-matcomp.csv", smooth_objective_results)
