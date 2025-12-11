function plot_fh_compare_tikz(out_standard, out_proj, simulate, data,
  rel_err_standard, rel_err_proj;
  name::AbstractString="ir2n-contexts")

  # ===== sorties simulées (même logique que plot_fh d'origine) =====
  F_standard = vec(simulate(out_standard.solution))
  F_proj = vec(simulate(out_proj.solution))
  data_vec = vec(data)

  @assert length(F_standard) == length(F_proj) == length(data_vec) "F_standard, F_proj et data doivent avoir même longueur"
  @assert iseven(length(data_vec)) "Longueur de data doit être paire (V/W intercalés)"

  N = length(data_vec) ÷ 2
  idx1 = 1:2:(2N-1)
  idx2 = 2:2:(2N)

  # temps = indices, comme dans plot_fh
  t = collect(1:N)

  # --- inexact ---
  V_std = F_standard[idx1]
  W_std = F_standard[idx2]

  data1 = data_vec[idx1]
  data2 = data_vec[idx2]

  # --- exact ---
  V_proj = F_proj[idx1]
  W_proj = F_proj[idx2]

  # ===== erreurs relatives en texte =====
  rel_std_str = @sprintf("error = %.2e", rel_err_standard)
  rel_proj_str = @sprintf("error = %.2e", rel_err_proj)

  # ===== tables PGFPlotsX =====
  tbl_V_std = ts_to_table(t, V_std)
  tbl_W_std = ts_to_table(t, W_std)
  tbl_V_proj = ts_to_table(t, V_proj)
  tbl_W_proj = ts_to_table(t, W_proj)
  tbl_data1 = ts_to_table(t, data1)
  tbl_data2 = ts_to_table(t, data2)

  # ===== encadrés d’erreur =====
  # ===== encadrés d’erreur (en bas à gauche) =====
  extra_std = raw"\node[anchor=south west,font=\scriptsize," *
              raw"draw=black,fill=white,line width=0.8pt,inner sep=2pt]" *
              raw" at (rel axis cs:0.03,0.05) {" * rel_std_str * "};"

  extra_proj = raw"\node[anchor=south west,font=\scriptsize," *
               raw"draw=black,fill=white,line width=0.8pt,inner sep=2pt]" *
               raw" at (rel axis cs:0.03,0.05) {" * rel_proj_str * "};"


  tikz = @pgf GroupPlot(
    {
      group_style = {
        group_size = "2 by 1",
        horizontal_sep = "2.5cm",
      },
      "axis on top",
      enlargelimits = "false",
      title_style = "{font=\\Large}",
      ticklabel_style = "{font=\\large}",
      label_style = "{font=\\large}",
    },
    Axis(
      {
        title = raw"Inexact prox (\(\kappa_s = 10^{-7}\))",
        xlabel = "time",
        ylabel = "voltage",
        legend_style = "{at={(1.0,1.0)},anchor=north east,draw=none,font=\\scriptsize}",
        xmin = minimum(t),
        xmax = maximum(t),
        "extra description" = extra_std,
      },
      # V(t) inexact
      Plot(
        {"no marks", "line width=1.3pt", "color=blue"},
        tbl_V_std,
      ),
      # W(t) inexact
      Plot(
        {"no marks", "line width=1.3pt", "color=red"},
        tbl_W_std,
      ),
      # V data
      Plot(
        {"only marks", "mark=o", "mark size=1.5pt", "color=black"},
        tbl_data1,
      ),
      # W data
      Plot(
        {"only marks", "mark=*", "mark size=1.5pt", "color=black"},
        tbl_data2,
      ),
      LegendEntry("V"),
      LegendEntry("W"),
      LegendEntry("V data"),
      LegendEntry("W data"),
    ),
    Axis(
      {
        title = raw"Exact prox",
        xlabel = "time",
        ylabel = "voltage",
        legend_style = "{at={(1.0,1.0)},anchor=north east,draw=none,font=\\scriptsize}",
        xmin = minimum(t),
        xmax = maximum(t),
        "extra description" = extra_proj,
      },
      # V(t) exact -- tirets
      Plot(
        {"no marks", "line width=1.3pt", "color=blue", "dashed"},
        tbl_V_proj,
      ),
      # W(t) exact -- tirets
      Plot(
        {"no marks", "line width=1.3pt", "color=red", "dashed"},
        tbl_W_proj,
      ),
      # V data
      Plot(
        {"only marks", "mark=o", "mark size=1.5pt", "color=black"},
        tbl_data1,
      ),
      # W data
      Plot(
        {"only marks", "mark=*", "mark size=1.5pt", "color=black"},
        tbl_data2,
      ),
      LegendEntry("V"),
      LegendEntry("W"),
      LegendEntry("V data"),
      LegendEntry("W data"),
    ),
  )

  outname = "fh-$(name).tikz"
  PGFPlotsX.save(outname, tikz)
  println("Saved $(abspath(outname))")
end
