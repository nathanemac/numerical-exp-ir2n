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

using PGFPlotsX
using LaTeXStrings
using Printf

# helper : t ↦ courbe (t, y(t)) pour PGFPlotsX
function ts_to_table(t::AbstractVector, y::AbstractVector)
  @assert length(t) == length(y)
  Table(["t" => collect(t), "y" => collect(y)])
end

include("plot-utils-fh-latex.jl")

function demo_fh()
  data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
  model = ADNLPModel(misfit, 1 / 10 * ones(5); matrix_free=true)
  options = ROSolverOptions(; verbose=100, ϵa=1e-6, ϵr=1e-6, β=1e16, ν=1.0e+2, σmin=5e2, σk=1e5, maxIter=1000)
  solve_ir2n(flag_projLp) = begin
    context = IRBPContext(model.meta.nvar, 0.5, 2.0; κs=1e-7, flag_projLp=flag_projLp)
    hp = ProjLpBall(1.0, 0.5, 2.0, context)
    iR2N(LBFGSModel(model), hp, options)
  end

  out_standard = solve_ir2n(0)
  out_proj = solve_ir2n(1)

  rel_err_standard = norm(out_standard.solution - x0) / norm(x0)
  rel_err_proj = norm(out_proj.solution - x0) / norm(x0)

  @info "iR2N relative error (flag_projLp=0)" rel_err_standard
  @info "iR2N relative error (flag_projLp=1)" rel_err_proj

  plot_fh_compare_tikz(out_standard, out_proj, simulate, data,
    rel_err_standard, rel_err_proj,
    name="ir2n-exact-vs-inexact")

end

demo_fh()