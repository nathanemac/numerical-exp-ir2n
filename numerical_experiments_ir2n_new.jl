using Revise
using NLPModels
using ADNLPModels
using DifferentialEquations # for fh_model(). removed for compat with PGFPlots
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

# irbp
data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
model = ADNLPModel(misfit, 1 / 10 * ones(5); matrix_free=true)
options = ROSolverOptions(; verbose=100, ϵa=1e-6, ϵr=1e-6, β=1e16, ν=1.0e+2, σmin=5e2, σk=1e5)
context_irbp_exact = IRBPContext(model.meta.nvar, 0.5, 2.0; κs=1e-7, flag_projLp=1)
context_irbp_inexact = IRBPContext(model.meta.nvar, 0.5, 2.0; κs=1e-7)
hp_exact = ProjLpBall(1.0, 0.5, 2.0, context_irbp_exact)
hp_inexact = ProjLpBall(1.0, 0.5, 2.0, context_irbp_inexact)
res_ir2n_exact = iR2N(LBFGSModel(model), hp_exact, options)
res_ir2n_inexact = iR2N(LBFGSModel(model), hp_inexact, options)

# res_ir2_exact = iR2(model, hp_exact, options)
# res_ir2_inexact = iR2(model, hp_inexact, options)


# lp
model_lp, nls_model_lp, _ = bpdn_mo2del(1) # m, n > 12 => convergence issues
context_lp = ProxTVContext(model_lp.meta.nvar, :lp, 1.2, κs=1e-1, λ=0.1)
hp_lp = NormLp(0.1, 1.2, context_lp)
options_lp = ROSolverOptions(verbose=1, ϵa=1e-3, ϵr=1e-3, maxIter=100)
ir2n_lp = iR2N(LBFGSModel(model_lp), hp_lp, options_lp)

# tvp
model_tvp, nls_model_tvp, _ = random_matrix_completion_model(m=20, n=40, r=10)
context_tvp = ProxTVContext(model_tvp.meta.nvar, :tvp, 2.7, κs=1e-1, λ=0.1)
hp_tvp = NormTVp(0.1, 2.7, context_tvp)
options_tvp = ROSolverOptions(verbose=1, ϵa=1e-1, ϵr=1e-1, maxIter=100)
ir2n_tvp = iR2N(LBFGSModel(model_tvp), hp_tvp, options_tvp)


model, nls_model, sol = bpdn_model(1)
f = LBFGSModel(model)
λ = norm(grad(model, zeros(model.meta.nvar)), Inf)
context_lp = ProxTVContext(model.meta.nvar, :lp, 2.7, κs=1e-5, λ=λ, dualGap=1e-12)
hp_lp = NormLp(λ, 2.7, context_lp)
options = ROSolverOptions(ν=1.0, β=1e16, ϵa=1e-6, ϵr=1e-6, verbose=10)
iR2N_out = iR2N(f, hp_lp, options, x0=f.meta.x0)

# ######################################################################################
# ### 0 - COMPARE R2N, IR2N, TR, R2 on the same problem (fh_model, h = NormL1(10.0) for exact prox and IRBP for inexact prox) ###
# ######################################################################################
# data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
# model = ADNLPModel(misfit, 1 / 10 * ones(5); matrix_free=true)
# options = ROSolverOptions(; verbose=50, ϵa=1e-5, ϵr=1e-5, β=1e16, ν=1.0e+2, σmin=5e2, σk=1e5)
# lbfgs_model = LBFGSModel(model)

# context = IRBPContext(model.meta.nvar, 0.5, 2.0; κs=0.8)
# inexact_h = ProjLpBall(1., 0.5, 2.0, context)
# h = NormL1(10.0)
# χ = NormLinf(1.0)

# TR_out = TR(lbfgs_model, h, χ, options)
# reset!(lbfgs_model)
# R2N_out = R2N(lbfgs_model, h, options)
# reset!(lbfgs_model)
# iR2N_out = iR2N(lbfgs_model, inexact_h, options)
# R2_out = R2(lbfgs_model, h, options)
# LMTR_out = LMTR(model, h, inexact_h, options) # TODO: fix this




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
p_range = [2.7] # 
κs_range = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] # respects convergence conditions
n_runs = 10
for p in p_range
  println("=== p = $p ===")
  for κs in κs_range
    println("  κs = $κs")
    for run in 1:n_runs
      print("    run $run … \n")
      try
        model, nls_model, _ = bpdn_model(300, 600, 200)
        context = ProxTVContext(model.meta.nvar, :lp, p, κs=κs, λ=0.1)
        hp = NormLp(0.1, p, context)
        options = ROSolverOptions(verbose=1, ϵa=1e-1, ϵr=1e-1, maxIter=100)

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
filename = "numerical_results_final/december-2025/ir2n_bpdn_lp_stats__p_kappaxi_$date_today.csv"
CSV.write(filename, grouped_stats_I)



###################################################################
############## II - f = matrix_completion(), h = TV ###############
###################################################################

# ------------------------------ DataFrame runs -------------------------
stats_runs_II = DataFrame(
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
p_range = [2.7] # 
κs_range = [0.25, 0.5, 0.75, 0.9, 0.99] # respects convergence conditions
n_runs = 10
for p in p_range
  println("=== p = $p ===")
  for κs in κs_range
    println("  κs = $κs")
    for run in 1:n_runs
      print("    run $run … \n")
      try
        model, nls_model, _ = random_matrix_completion_model(m=20, n=40, r=10)
        context = ProxTVContext(model.meta.nvar, :tvp, p, κs=κs, λ=0.1)
        hp = NormTVp(0.1, p, context)
        options = ROSolverOptions(verbose=1, ϵa=1e-1, ϵr=1e-1, maxIter=100)

        ir2n = iR2N(LBFGSModel(model), hp, options)

        push!(stats_runs_II, (
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
        push!(stats_runs_II, (p, κs, false,
          missing, missing, missing,
          missing, missing, missing,
          missing, missing, missing,
          missing, missing, missing, missing))
        println("FAIL")
      end
    end
  end
end

# ------------------------------ Agrégation ----------------------------
grouped_stats_II = combine(groupby(stats_runs_II, [:p, :κs])) do df
  n_fail = count(!, df.ok)
  df_ok = df[df.ok, :]
  (; n_runs=size(df, 1), n_fail,
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

for c in names(grouped_stats_II)
  if eltype(grouped_stats_II[!, c]) <: Real
    grouped_stats_II[!, c] = [sci2(v) for v in grouped_stats_II[!, c]]
  end
end

println("\nRésumé des statistiques iR2N par (p, κs) :")
show(grouped_stats_II, allrows=true, allcols=true)

date_today = Dates.format(Dates.now(), "yyyy-mm-dd-HHh-MM")
filename = "numerical_results_final/december-2025/ir2n_matrix_completion_tvp_stats__p_kappaxi_$date_today.csv"
CSV.write(filename, grouped_stats_II)


###################################################################
############## III - f = fh_model(), h = ProjLpBall ###############
###################################################################
# !!! Experiments on FH + IRBP are on Atlas
# n_runs = 10
# p_range = [0.5]
# κs_range = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0 - eps()] # respects convergence conditions

# # ------------------------------ DataFrame runs -------------------------
# stats_runs = DataFrame(
#   p=Float64[],
#   κs=Float64[],
#   ok=Bool[],
#   it_iR2N=Union{Missing,Int}[],
#   it_iR2=Union{Missing,Int}[],
#   it_mean_iR2=Union{Missing,Float64}[],
#   prox_calls_outer=Union{Missing,Int}[],
#   total_iters_prox_outer=Union{Missing,Int}[],
#   mean_iters_prox_outer=Union{Missing,Float64}[],
#   prox_calls_inner=Union{Missing,Int}[],
#   total_iters_prox_inner=Union{Missing,Int}[],
#   mean_iters_prox_inner=Union{Missing,Float64}[],
#   prox_calls_all=Union{Missing,Int}[],
#   total_iters_prox_all=Union{Missing,Int}[],
#   mean_iters_prox_all=Union{Missing,Float64}[],
#   t_iR2N=Union{Missing,Float64}[]
# )

# # ------------------------------ Boucles -------------------------------
# for p in p_range
#   println("=== p = $p ===")
#   for κs in κs_range
#     println("  κs = $κs")
#     for run in 1:n_runs
#       print("    run $run … \n")
#       try
#         # ----- créer le problème FH pour ce run -----
#         data, sim, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
#         model = ADNLPModel(misfit, 0.1 .* ones(5); matrix_free=true)
#         lbfgs_model = LBFGSModel(model)
#         context = IRBPContext(model.meta.nvar, p, 2.0; κs=κs)
#         hp = ProjLpBall(1.0, p, 2.0, context)

#         opts = ROSolverOptions(verbose=50, ϵa=1e-5, ϵr=1e-5,
#           β=1e16, ν=1e2, σmin=5e2, σk=1e5)
#         ir2n = iR2N(lbfgs_model, hp, opts)

#         push!(stats_runs, (
#           p, κs, true,
#           ir2n.iter,
#           ir2n.solver_specific[:total_iters_subsolver],
#           ir2n.solver_specific[:mean_iters_subsolver],
#           ir2n.solver_specific[:prox_calls_outer],
#           ir2n.solver_specific[:total_iters_prox_outer],
#           ir2n.solver_specific[:mean_iters_prox_outer],
#           ir2n.solver_specific[:prox_calls_inner],
#           ir2n.solver_specific[:total_iters_prox_inner],
#           ir2n.solver_specific[:mean_iters_prox_inner],
#           ir2n.solver_specific[:prox_calls_all],
#           ir2n.solver_specific[:total_iters_prox_all],
#           ir2n.solver_specific[:mean_iters_prox_all],
#           ir2n.elapsed_time
#         ))
#         println("OK")
#       catch err
#         @warn "échec run=$run p=$p κs=$κs" err
#         push!(stats_runs, (p, κs, false,
#           missing, missing, missing,
#           missing, missing, missing,
#           missing, missing, missing,
#           missing, missing, missing, missing))
#         println("FAIL")
#       end
#     end
#   end
# end

# # ------------------------------ Agrégation ----------------------------
# grouped_stats = combine(groupby(stats_runs, [:p, :κs])) do df
#   n_fail = count(!, df.ok)              # nombre de faux
#   df_ok = df[df.ok, :]                 # on garde uniquement les succès pour les moyennes
#   (; n_runs=size(df, 1),
#     n_fail,
#     iter_iR2N=mean(skipmissing(df_ok.it_iR2N)),
#     iter_iR2=mean(skipmissing(df_ok.it_iR2)),
#     moyenne_iter_iR2=mean(skipmissing(df_ok.it_mean_iR2)),
#     prox_calls_outer=mean(skipmissing(df_ok.prox_calls_outer)),
#     total_iters_prox_outer=mean(skipmissing(df_ok.total_iters_prox_outer)),
#     mean_iters_prox_outer=mean(skipmissing(df_ok.mean_iters_prox_outer)),
#     prox_calls_inner=mean(skipmissing(df_ok.prox_calls_inner)),
#     total_iters_prox_inner=mean(skipmissing(df_ok.total_iters_prox_inner)),
#     mean_iters_prox_inner=mean(skipmissing(df_ok.mean_iters_prox_inner)),
#     prox_calls_all=mean(skipmissing(df_ok.prox_calls_all)),
#     total_iters_prox_all=mean(skipmissing(df_ok.total_iters_prox_all)),
#     mean_iters_prox_all=mean(skipmissing(df_ok.mean_iters_prox_all)),
#     moyenne_temps=mean(skipmissing(df_ok.t_iR2N)))
# end

# using Printf

# sci2(x) = x isa Missing ? x : @sprintf("%.2e", x)

# for c in names(grouped_stats)
#   if eltype(grouped_stats[!, c]) <: Real
#     grouped_stats[!, c] = [sci2(v) for v in grouped_stats[!, c]]
#   end
# end

# println("\nRésumé des statistiques iR2N par (p, κs) :")
# show(grouped_stats, allrows=true, allcols=true)

# # ------------------------------ Sauvegarde CSV ------------------------
# date_tag = Dates.format(now(), "yyyy-mm-dd-HHh-MM")
# out_dir = "numerical_results_final/december-2025"
# filename = "$out_dir/ir2n_fh_projlp_stats_$date_tag.csv"
# CSV.write(filename, grouped_stats)


# ###################################################################
# ############## IV - f = fh_model(), h = ProjLpBall ###############
# ###################################################################
# # Now we vary the tolerance (abs and rel) for a fixed κs and p.
# # Tolerance directly impacts the precision at which we evaluate the objective, therefore its gradient. 
# # -------------------- paramètres expérimentaux -------------------
# n_runs = 10
# p_range = [0.5 + eps(), 0.6, 0.68, 0.75, 0.83, 0.9, 1 - eps()]
# κs_fixed = 0.9
# tol_range = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14] # même tol en abs/rel

# # -------------------- DataFrame pour chaque run -----------------
# stats_runs_tol = DataFrame(
#   p=Float64[], tol=Float64[], ok=Bool[],
#   it_iR2N=Union{Missing,Int}[],
#   it_iR2=Union{Missing,Int}[],
#   it_mean_iR2=Union{Missing,Float64}[],
#   it_Prox=Union{Missing,Int}[],
#   it_mean_Prox=Union{Missing,Float64}[],
#   t_iR2N=Union{Missing,Float64}[]
# )

# # -------------------- boucles principales -----------------------
# for p in p_range
#   println("=== p = $p ===")
#   for tol in tol_range
#     println("  tol = $tol")
#     for run in 1:n_runs
#       print("    run $run … ")
#       try
#         # --- créer objectif avec tolérance courante ---
#         data, sim, resid, misfit, x0 = RegularizedProblems.FH_smooth_term(
#           abstol=tol, reltol=tol)
#         model = ADNLPModel(misfit, 0.1 .* ones(5); matrix_free=true)
#         lbfgs_model = LBFGSModel(model)
#         ctx = IRBPContext(model.meta.nvar, 0.5, 2.0; κs=κs_fixed)
#         hp = ProjLpBall(1.0, p, 2.0, ctx)

#         opts = ROSolverOptions(verbose=50, ϵa=1e-5, ϵr=1e-5,
#           β=1e16, ν=1e2, σmin=1e2, σk=1e5)
#         ir2n = iR2N(lbfgs_model, hp, opts)

#         push!(stats_runs_tol, (
#           p, tol, true,
#           ir2n.iter,
#           ir2n.solver_specific[:total_iters_subsolver],
#           ir2n.solver_specific[:mean_iters_subsolver],
#           ir2n.solver_specific[:total_iters_prox],
#           ir2n.solver_specific[:mean_iters_prox],
#           ir2n.elapsed_time
#         ))
#         println("OK")
#       catch err
#         @warn "échec run=$run p=$p tol=$tol" err
#         push!(stats_runs_tol, (p, tol, false,
#           missing, missing, missing, missing, missing, missing))
#         println("FAIL")
#       end
#     end
#   end
# end

# # -------------------- agrégation -------------------------------
# grouped_tol = combine(groupby(stats_runs_tol, [:p, :tol])) do df
#   n_fail = count(!, df.ok)
#   df_ok = df[df.ok, :]
#   (; n_runs=size(df, 1), n_fail,
#     iter_iR2N=mean(skipmissing(df_ok.it_iR2N)),
#     iter_iR2=mean(skipmissing(df_ok.it_iR2)),
#     moyenne_iter_iR2=mean(skipmissing(df_ok.it_mean_iR2)),
#     tot_iter_prox=mean(skipmissing(df_ok.it_Prox)),
#     moyenne_iter_prox=mean(skipmissing(df_ok.it_mean_Prox)),
#     moyenne_temps=mean(skipmissing(df_ok.t_iR2N)))
# end

# println("\nRésumé par (p, tol) :")
# show(grouped_tol, allrows=true, allcols=true)

# # -------------------- sauvegarde CSV ---------------------------
# date_tag = Dates.format(now(), "yyyy-mm-dd-HHh-MM")
# out_dir = "numerical_results_final/october-2025"
# filename = "$out_dir/ir2n_fh_projlp_tolerance_stats_$date_tag.csv"
# CSV.write(filename, grouped_tol)