# Changes in version 1.3 (prepared but not released separately; its changes are part of version 1.4)

Version 1.3 changes the presentation of three figures. No data, result or number changes.

Changed
- `src/figures.py`
  - Figure 5 (`fig3_net_saving_vs_radio_cost`): the x-axis label is now "radio cost per report e_r (J)"
    (the swept range is stated in the caption); a crossover at or above 1e-3 J is labelled in the
    ×10⁻³ form (exchange_cpc: "2.2×10⁻³ J" instead of "22.0×10⁻⁴ J"); the crossover labels and the
    anchor label have a white background so that they stay legible where they cross the curves.
  - Figure 8 (`fig4_tolerance_sweep`): the cell values use the minus sign (U+2212) instead of the
    hyphen, as the colour bar and Table 4 do.
- `src/figures_extra.py`
  - Figure 9 (`fig_checkpoints`): panel (a) gains a colour key (lightweight, CNN-LSTM), the same as
    panel (b).
- `figures/`: `fig3_net_saving_vs_radio_cost`, `fig4_tolerance_sweep` and `fig_checkpoints`
  (PNG and PDF) regenerated with `figures_pdf.py` and `figures_extra.py` in a clean run directory,
  following the README. The other five figures are unchanged: their PNG files are byte-identical when
  regenerated, and their PDF files differ only in the embedded creation date, so the shipped copies
  are kept. `checkpoint_stats.json` regenerates byte-identical.

At the time of version 1.3 the figures in the manuscript were these files, byte-identical (SHA-256 in
`SHA256SUMS_v1.3`); version 1.4 replaces them (see Part 4 of `CHANGES_v1.4.md`).
