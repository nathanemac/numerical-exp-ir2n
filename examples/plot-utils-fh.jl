function plot_fh_compare(out_standard, out_proj, simulate, data,
  rel_err_standard, rel_err_proj,
  name::AbstractString="ir2n-contexts")

  F_standard = simulate(out_standard.solution)
  F_proj = simulate(out_proj.solution)

  fig, ax = subplots(1, 2, figsize=(11, 5.5), constrained_layout=true)

  # -------- panneau 1 : inexact prox --------
  _plot_fh_panel!(ax[1], F_standard, data;
    title_str=raw"Inexact prox (\(\kappa_s = 10^{-7}\))",
    line_styles=("-", "-"),
    add_legend=false,
  )

  # légende principale (V, W, data)
  handles1, labels1 = ax[1].get_legend_handles_labels()
  leg1 = ax[1].legend(handles1, labels1,
    loc="upper right", frameon=false, fontsize="small")

  # encadré pour l'erreur relative (texte + bbox)
  rel_std_str = @sprintf("error = %.2e", rel_err_standard)
  ax[1].text(
    0.03, 0.05, rel_std_str,                # position en coordonnées d'axes
    transform=ax[1].transAxes,
    ha="left", va="bottom",
    bbox=Dict(
      "boxstyle" => "square",   # coins carrés
      "fc" => "white",     # fond transparent
      "ec" => "black",    # bord noir
      "lw" => 0.8,
    ),
    fontsize=12,
  )

  # -------- panneau 2 : exact prox --------
  _plot_fh_panel!(ax[2], F_proj, data;
    title_str=raw"Exact prox",
    line_styles=("--", "--"),
    add_legend=false,
  )

  handles2, labels2 = ax[2].get_legend_handles_labels()
  leg2 = ax[2].legend(handles2, labels2,
    loc="upper right", frameon=false, fontsize="small")

  rel_proj_str = @sprintf("error = %.2e", rel_err_proj)
  ax[2].text(
    0.03, 0.05, rel_proj_str,
    transform=ax[2].transAxes,
    ha="left", va="bottom",
    bbox=Dict(
      "boxstyle" => "square",
      "fc" => "white",
      "ec" => "black",
      "lw" => 0.8,
    ),
    fontsize=12,
  )

  fig.savefig("fh-$(name).pdf", bbox_inches="tight")
  close(fig)
end
