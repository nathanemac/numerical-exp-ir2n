ENV["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" * get(ENV, "PATH", "")

using PyPlot
using Printf
using Random

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", serif="Computer Modern Roman")
PyPlot.rc("text.latex", preamble="\\usepackage{amsmath}")

PyPlot.rc("font", size=18)
PyPlot.rc("axes", titlesize=20, labelsize=18)
PyPlot.rc("xtick", labelsize=16)
PyPlot.rc("ytick", labelsize=16)
PyPlot.rc("legend", fontsize=16)

Random.seed!(12345)
function quasi_lp_ball!(ax; p::Real=0.5, r::Real=1.0, num_points::Integer=600)
  (0 < p < 1) || throw(ArgumentError("p must satisfy 0 < p < 1, got $p"))
  (r > 0) || throw(ArgumentError("r must be positive, got $r"))

  x_max = r^(1 / p)
  xs = collect(range(-x_max, x_max; length=num_points))
  ys = similar(xs)
  for (i, x) in enumerate(xs)
    radial = max(r - abs(x)^p, 0.0)
    ys[i] = radial == 0 ? 0.0 : radial^(1 / p)
  end

  ax.fill_between(xs, ys, -ys;
    color="lightskyblue", alpha=0.6,
    label="\$\\|y\\|_p^p \\le r\$",
  )

  ax.plot(xs, ys; color="black", linewidth=1.5)
  ax.plot(xs, -ys; color="black", linewidth=1.5)

  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("\$y_1\$")
  ax.set_ylabel("\$y_2\$")
  ax.grid(false)
end

function plot_quasi_lp_ball(; p::Real=0.5, r::Real=1.0, num_points::Integer=600,
  outfile::AbstractString="")
  fig, ax = subplots(figsize=(5, 5))
  quasi_lp_ball!(ax; p=p, r=r, num_points=num_points)

  p_str = replace(replace(string(p), r"0+$" => ""), r"\.$" => "")
  r_str = replace(replace(string(r), r"0+$" => ""), r"\.$" => "")

  title_str = "\$\\|y\\|_p^p \\le r,\\quad p=$(p_str),\\ r=$(r_str)\$"
  ax.set_title(title_str; fontsize=20)

  # --- Annotations LaTeX dans / hors de la boule ---
  ax.text(0.0, 0.0, "\$\\chi_{p,r}(y) = 0\$",
    ha="center", va="center", fontsize=18)

  ax.text(0.05, 0.8, "\$\\chi_{p,r}(y) = +\\infty\$",
    transform=ax.transAxes,
    ha="left", va="top", fontsize=18)

  ax.legend(loc="upper right", frameon=false)
  fig.tight_layout()

  if isempty(outfile)
    sanitized = replace(@sprintf("p%.2f_r%.2f", p, r), "." => "")
    outfile = "lp_quasiball_$sanitized.pdf"
  end

  fig.savefig(outfile, dpi=300)
  println("Saved plot to $(abspath(outfile))")
  return outfile
end

plot_quasi_lp_ball(p=0.5, r=2.0, outfile="lp_quasiball_p0.5_r1.0.pdf")
