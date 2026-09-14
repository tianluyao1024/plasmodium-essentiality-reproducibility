"""Render the reviewer-polished v58 version of the directional-context figure.

This wrapper reuses the frozen v57 data assembly and plotting code.  It changes
only the layout allocation and the wording of panel b; no model, score, call or
statistical calculation is recomputed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure

import make_v57_directional_context_figure as v57


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "protozoa_human_sci_analysis" / "reviewer_revision" / "ncomms_submission_v58_final_polish"


def main() -> None:
    v57.OUT = OUT
    v57.FIG_DIR = OUT / "Main_Figures"
    v57.SOURCE_DIR = OUT / "Source_Data"

    original_add_gridspec = matplotlib.figure.Figure.add_gridspec
    original_panel_label = v57.panel_label

    def reviewer_layout(self, nrows=1, ncols=1, **kwargs):
        if nrows == 2 and ncols == 2:
            kwargs["width_ratios"] = [0.66, 1.34]
        return original_add_gridspec(self, nrows=nrows, ncols=ncols, **kwargs)

    def reviewer_panel_label(ax, label: str, title: str) -> None:
        if label == "b":
            title = "Sequence identity of directional and matched E/E/E triplets"
        original_panel_label(ax, label, title)

    matplotlib.figure.Figure.add_gridspec = reviewer_layout
    v57.panel_label = reviewer_panel_label
    try:
        v57.main()
    finally:
        matplotlib.figure.Figure.add_gridspec = original_add_gridspec
        v57.panel_label = original_panel_label


if __name__ == "__main__":
    main()
