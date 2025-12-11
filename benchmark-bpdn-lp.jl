using Revise
using NLPModels
using ADNLPModels
# using DifferentialEquations # for fh_model(). removed for compat with PGFPlots
using RegularizedProblems
using ProximalOperators
using NLPModelsModifiers
using Logging
using Random
using DataFrames
using Statistics
using CSV
using LinearAlgebra

using RegularizedOptimization
using ProxTV
using IRBP

using Dates



###################################################################
############## I - f = bpdn(), h = lp ################
###################################################################

# ------------------------------ DataFrame runs -------------------------
stats_runs_I = DataFrame(
  p=Float64[],
  κs=Float64[],
  ok=Bool[],
  it_iR2N=Union{Missing,Int}[],
  it_iR2=Union{Missing,Int}[],
  it_mean_iR2=Union{Missing,Float64}[],
  prox_calls_outer=Union{Missing,Int}[],
  total_iters_prox_outer=Union{Missing,Int}[],
  mean_iters_prox_outer=Union{Missing,Float64}[],
  prox_calls_inner=Union{Missing,Int}[],
  total_iters_prox_inner=Union{Missing,Int}[],
  mean_iters_prox_inner=Union{Missing,Float64}[],
  prox_calls_all=Union{Missing,Int}[],
  total_iters_prox_all=Union{Missing,Int}[],
  mean_iters_prox_all=Union{Missing,Float64}[],
  t_iR2N=Union{Missing,Float64}[]
)

# ------------------------------ Boucles -------------------------------
p_range = [1.1] # 
κs_range = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.5, 0.9, 0.99] # respects convergence conditions
n_runs = 10
for p in p_range
  println("=== p = $p ===")
  for κs in κs_range
    println("  κs = $κs")
    for run in 1:n_runs
      print("    run $run … \n")
      try
        model, nls_model, _ = bpdn_model(1)
        λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 2
        context = ProxTVContext(model.meta.nvar, :lp, p, κs=κs, λ=λ)
        hp = NormLp(λ, p, context)
        options = ROSolverOptions(ν=1.0, β=1e16, verbose=5, ϵa=1e-6, ϵr=1e-6, maxIter=100)

        ir2n = iR2N(LBFGSModel(model), hp, options)

        push!(stats_runs_I, (
          p, κs, true,
          ir2n.iter,
          ir2n.solver_specific[:total_iters_subsolver],
          ir2n.solver_specific[:mean_iters_subsolver],
          ir2n.solver_specific[:prox_calls_outer],
          ir2n.solver_specific[:total_iters_prox_outer],
          ir2n.solver_specific[:mean_iters_prox_outer],
          ir2n.solver_specific[:prox_calls_inner],
          ir2n.solver_specific[:total_iters_prox_inner],
          ir2n.solver_specific[:mean_iters_prox_inner],
          ir2n.solver_specific[:prox_calls_all],
          ir2n.solver_specific[:total_iters_prox_all],
          ir2n.solver_specific[:mean_iters_prox_all],
          ir2n.elapsed_time
        ))
        println("OK")
      catch err
        @warn "échec run=$run p=$p κs=$κs" err
        push!(stats_runs_I, (p, κs, false,
          missing, missing, missing,
          missing, missing, missing,
          missing, missing, missing,
          missing, missing, missing, missing))
        println("FAIL")
      end
    end
  end
end

println("=== run exact criterion ===")
for p in p_range
  println("=== p = $p ===")
  for run in 1:10
    print("    run $run … \n")
    try
      model, nls_model, _ = bpdn_model(1)
      λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 2
      context = ProxTVContext(model.meta.nvar, :lp, p, λ=λ, κs=1e-5, use_absolute_criterion=true)
      hp = NormLp(λ, p, context)
      options = ROSolverOptions(ν=1.0, β=1e16, verbose=5, ϵa=1e-6, ϵr=1e-6, maxIter=100)
      ir2n = iR2N(LBFGSModel(model), hp, options)
      push!(stats_runs_I, (
        p, 10, true,
        ir2n.iter,
        ir2n.solver_specific[:total_iters_subsolver],
        ir2n.solver_specific[:mean_iters_subsolver],
        ir2n.solver_specific[:prox_calls_outer],
        ir2n.solver_specific[:total_iters_prox_outer],
        ir2n.solver_specific[:mean_iters_prox_outer],
        ir2n.solver_specific[:prox_calls_inner],
        ir2n.solver_specific[:total_iters_prox_inner],
        ir2n.solver_specific[:mean_iters_prox_inner],
        ir2n.solver_specific[:prox_calls_all],
        ir2n.solver_specific[:total_iters_prox_all],
        ir2n.solver_specific[:mean_iters_prox_all],
        ir2n.elapsed_time
      ))
      println("OK")
    catch err
      @warn "échec run=$run p=$p" err
      push!(stats_runs_I, (p, 10, false,
        missing, missing, missing,
        missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, missing))
      println("FAIL")
    end
  end
end

# ------------------------------ Agrégation ----------------------------
grouped_stats_I = combine(groupby(stats_runs_I, [:p, :κs])) do df
  n_fail = count(!, df.ok)
  df_ok = df[df.ok, :]
  (; n_runs=size(df, 1),
    n_fail,
    iter_iR2N=mean(skipmissing(df_ok.it_iR2N)),
    iter_iR2=mean(skipmissing(df_ok.it_iR2)),
    moyenne_iter_iR2=mean(skipmissing(df_ok.it_mean_iR2)),
    prox_calls_outer=mean(skipmissing(df_ok.prox_calls_outer)),
    total_iters_prox_outer=mean(skipmissing(df_ok.total_iters_prox_outer)),
    mean_iters_prox_outer=mean(skipmissing(df_ok.mean_iters_prox_outer)),
    prox_calls_inner=mean(skipmissing(df_ok.prox_calls_inner)),
    total_iters_prox_inner=mean(skipmissing(df_ok.total_iters_prox_inner)),
    mean_iters_prox_inner=mean(skipmissing(df_ok.mean_iters_prox_inner)),
    prox_calls_all=mean(skipmissing(df_ok.prox_calls_all)),
    total_iters_prox_all=mean(skipmissing(df_ok.total_iters_prox_all)),
    mean_iters_prox_all=mean(skipmissing(df_ok.mean_iters_prox_all)),
    moyenne_temps=mean(skipmissing(df_ok.t_iR2N)))
end

using Printf

sci2(x) = x isa Missing ? x : @sprintf("%.2e", x)

for c in names(grouped_stats_I)
  if eltype(grouped_stats_I[!, c]) <: Real
    grouped_stats_I[!, c] = [sci2(v) for v in grouped_stats_I[!, c]]
  end
end

println("\nRésumé des statistiques iR2N par (p, κs) :")
show(grouped_stats_I, allrows=true, allcols=true)

date_today = Dates.format(Dates.now(), "yyyy-mm-dd-HHh-MM")
filename = "numerical-results-bpdn-matcomp-tvp/december-2025/ir2n_matcomp_tvp_stats__p_kappa_s_$date_today.csv"
CSV.write(filename, grouped_stats_I)
