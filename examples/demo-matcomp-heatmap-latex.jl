ENV["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" * get(ENV, "PATH", "")

using Random
using LinearAlgebra
using PyPlot
using ProximalOperators
using ProxTV
using Distributions
using NLPModels, NLPModelsModifiers, RegularizedOptimization, RegularizedProblems

# Configuration LaTeX + Computer Modern
PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", serif="Computer Modern Roman")
PyPlot.rc("text.latex", preamble=raw"\usepackage[T1]{fontenc}\usepackage{amsmath}")


# ======== AJOUT POUR AGRANDIR LES POLICES ========
PyPlot.rc("font", size=18)
PyPlot.rc("axes", titlesize=20, labelsize=18)
PyPlot.rc("xtick", labelsize=16)
PyPlot.rc("ytick", labelsize=16)
PyPlot.rc("legend", fontsize=16)

using PGFPlotsX
using LaTeXStrings

Random.seed!(12345)

function demo_add_gauss(x, σ, μ; clip=false)
  noise = σ .* randn(eltype(x), size(x)) .+ μ
  y = x .+ noise
  clip ? clamp.(y, zero(eltype(y)), one(eltype(y))) : y
end


struct DemoMatrixProjection
  mask::BitVector
  m::Int
  n::Int
end

function DemoMatrixProjection(m::Int, n::Int, ω::AbstractVector{<:Integer})
  mask = falses(m * n)
  mask[ω] .= true
  DemoMatrixProjection(mask, m, n)
end

function demo_mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = rand(Uniform(-0.1, 0.3), m, r)
  xr = rand(Uniform(-0.1, 0.3), n, r)
  xs = xl * xr'
  Ω = findall(<(sr), rand(m, n))  # indices (i,j) retenus
  B = xs[Ω]
  B = (1 - c) * demo_add_gauss(B, va, 0; clip=true) +
      c * demo_add_gauss(B, vb, 0; clip=true)

  ω = zeros(Int64, length(Ω))   # indices linéarisés
  for i = 1:length(Ω)
    # ATTENTION : indexation colonne-major en Julia : row + m * (col - 1)
    ω[i] = Ω[i][1] + m * (Ω[i][2] - 1)
  end

  return xs, B, ω
end
function matrix_to_table(R::AbstractMatrix)
  m, n = size(R)
  rows = Array{Any}(undef, m * n + 1, 3)
  # Header pour PGFPlotsX
  rows[1, :] .= ("x", "y", "v")
  k = 2
  # x = colonne, y = ligne renversée (pour avoir la première ligne en haut)
  for j in 1:n, i in 1:m
    rows[k, 1] = j                     # x
    rows[k, 2] = m - i + 1             # y renversé
    rows[k, 3] = R[i, j]               # valeur
    k += 1
  end
  return Table(rows)
end


function demo_matrix_completion_model(xs, B, ω)
  m, n = size(xs)
  res = vec(fill!(similar(xs), 0))
  projection = DemoMatrixProjection(m, n, ω)

  function resid!(res, x)
    res .= 0
    res[ω] .= x[ω] .- B
    res
  end

  function jprod_resid!(Jv, x, v)
    Jv .= 0
    Jv[ω] .= v[ω]
    Jv
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  grad!(r, x) = resid!(r, x)

  x0 = rand(eltype(B), m * n)
  FirstOrderModel(obj, grad!, x0, name="MATRAND"),
  FirstOrderNLSModel(resid!, jprod_resid!, jprod_resid!, m * n, x0, name="MATRAND-LS"),
  vec(xs),
  projection
end

"""
    model, nls_model, sol, projection = demo_random_matrix_completion_model(; kwargs...)
"""
function demo_random_matrix_completion_model(;
  m::Int=100,
  n::Int=100,
  r::Int=5,
  sr::Float64=0.8,
  va::Float64=1.0e-4,
  vb::Float64=1.0e-2,
  c::Float64=0.2,
)
  xs, B, ω = demo_mat_rand(m, n, r, sr, va, vb, c)
  demo_matrix_completion_model(xs, B, ω)
end

function demo_perturb(I, c=0.8, p=0.8)
  Ω = findall(<(p), rand(256, 256))
  ω = zeros(Int, size(Ω, 1))   # Vectorize Omega
  for i = 1:size(Ω, 1)
    ω[i] = Ω[i][1] + 256 * (Ω[i][2] - 1)
  end
  X = fill!(similar(I), 0)
  B = I[Ω]
  B = c * demo_add_gauss(B, sqrt(0.001), 0) + (1 - c) * demo_add_gauss(B, sqrt(0.1), 0)
  X[Ω] .= B
  X, B, ω
end

function demo_MIT_matrix_completion_model()
  I = ones(256, 256)
  I[:, 1:20] .= 0.1
  I[1:126, 40:60] .= 0
  I[:, 80:100] .= 0
  I[1:40, 120:140] .= 0
  I[80:256, 120:140] .= 0.5
  I[1:40, 160:256] .= 0
  I[80:256, 160:180] .= 0

  X, B, ω = demo_perturb(I, 0.8, 0.8)
  demo_matrix_completion_model(X, B, ω)
end

function demo_projection_mask(projection::DemoMatrixProjection)
  reshape(projection.mask, projection.m, projection.n)
end

function demo_apply_projection_vector(projection::DemoMatrixProjection, x::AbstractVector)
  length(x) == length(projection.mask) || throw(ArgumentError("Vector has length $(length(x)), expected $(length(projection.mask))."))
  projection.mask .* x
end

function demo_apply_projection_matrix(projection::DemoMatrixProjection, X::AbstractMatrix)
  size(X, 1) == projection.m || throw(ArgumentError("Matrix has $(size(X, 1)) rows, expected $(projection.m)."))
  size(X, 2) == projection.n || throw(ArgumentError("Matrix has $(size(X, 2)) columns, expected $(projection.n)."))
  demo_projection_mask(projection) .* X
end

demo_masked_difference(projection::DemoMatrixProjection, X::AbstractMatrix, A::AbstractMatrix) =
  demo_apply_projection_matrix(projection, X .- A)
function plot_projection_experiment(; λ=1e-1, p=1.1, kwargs...)
  # Problème de complétion de matrice
  model, _, sol, projection = demo_random_matrix_completion_model(; kwargs...)
  f = LBFGSModel(model)
  options = ROSolverOptions(ν=1.0, β=1e16, ϵa=1e-3, ϵr=1e-3, verbose=1, neg_tol=1e-2)

  m, n = projection.m, projection.n

  # ===== 1) Critère "inexact" =====
  context_tvp_1 = ProxTVContext(model.meta.nvar, :tvp, p,
    κs=1e-5, λ=λ, dualGap=eps())
  hp_tvp_1 = NormTVp(λ, p, context_tvp_1)
  solver_out_1 = iR2N(f, hp_tvp_1, options, x0=f.meta.x0)
  X1 = reshape(solver_out_1.solution, m, n)

  # ===== 2) Critère "exact" =====
  context_tvp_2 = ProxTVContext(model.meta.nvar, :tvp, p,
    κs=1e-5, λ=λ, dualGap=eps(), use_absolute_criterion=true)
  hp_tvp_2 = NormTVp(λ, p, context_tvp_2)
  solver_out_2 = iR2N(f, hp_tvp_2, options, x0=f.meta.x0)
  X2 = reshape(solver_out_2.solution, m, n)

  # vraie matrice
  A = reshape(sol, m, n)

  # masque Ω
  mask_mat = Float64.(demo_projection_mask(projection))

  # résidus masqués
  R1 = abs.(X1 .- A) .* mask_mat
  R2 = abs.(X2 .- A) .* mask_mat
  R3 = abs.(X1 .- X2) .* mask_mat

  # échelles
  maxR12 = max(maximum(R1), maximum(R2))
  maxR3 = maximum(R3)

  # tables pour PGFPlotsX
  tbl1 = matrix_to_table(R1)
  tbl2 = matrix_to_table(R2)
  tbl3 = matrix_to_table(R3)

  # === Figure PGFPlotsX : 3 heatmaps côte à côte ===
  tikz = @pgf GroupPlot(
    {
      group_style = {
        group_size = "3 by 1",
        horizontal_sep = "1.2cm",
      },
      "axis on top",
      enlargelimits = "false",
      xmin = 0.5, xmax = n + 0.5,
      ymin = 0.5, ymax = m + 0.5,
      xtick = "\\empty",
      ytick = "\\empty",
    }, Axis(
      {
        title = L"|\!X_{\mathrm{inexact}} - A\!| \, (\kappa_s = 10^{-7})",
        "colorbar",
        colormap_name = "magma",
        point_meta_min = 0.0,
        point_meta_max = maxR12,
      },
      Plot(
        {
          "matrix plot*",
          mesh = {cols = n},
          "point meta=explicit",
        },
        tbl1,
      ),
    ), Axis(
      {
        title = L"|\!X_{\mathrm{exact}} - A\!|",
        "colorbar",
        colormap_name = "magma",
        point_meta_min = 0.0,
        point_meta_max = maxR12,
      },
      Plot(
        {
          "matrix plot*",
          mesh = {cols = n},
          "point meta=explicit",
        },
        tbl2,
      ),
    ), Axis(
      {
        title = L"|\!X_{\mathrm{inexact}} - X_{\mathrm{exact}}\!|",
        "colorbar",
        colormap_name = "viridis",
        point_meta_min = 0.0,
        point_meta_max = maxR3,
      },
      Plot(
        {
          "matrix plot*",
          mesh = {cols = n},
          "point meta=explicit",
        },
        tbl3,
      ),
    ),
  )

  outfile = "demo-matcomp-comparison.tikz"
  PGFPlotsX.save(outfile, tikz)
  println("Saved TikZ figure to $(abspath(outfile))")
end

plot_projection_experiment(m=10, n=12, r=5, sr=0.8)
