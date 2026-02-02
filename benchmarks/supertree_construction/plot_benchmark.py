"""Script for plotting the benchmark of supertree construction algorithms."""

from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

from run_benchmark import CONRACTION_PROBABILITY_VALUES
from run_benchmark import NUM_SUBTREES_VALUES
from run_benchmark import PATH_RESULT_DIR
from run_benchmark import PATH_OUTPUT_FILE as PATH_BENCHMARK

PATH_BOXPLOT = PATH_RESULT_DIR / "benchmark_boxplots.pdf"
PATH_PLOT_MEDIANS = PATH_RESULT_DIR / "benchmark_medians.pdf"

COLUMN2ALGO_NAME = {
    "time_classic_build": "BUILD",
    "time_buildst": "BuildST",
    "time_merge_trees": "Merge_Trees",
    "time_lct": "Loose_Cons",
    "time_lincr": "LinCR",
}

FONTSIZE = 16


# read the execution times
df = pd.read_csv(PATH_BENCHMARK).rename(columns=COLUMN2ALGO_NAME)

# check whether all supertree methods produce the same results
for col in [
    "classic_build_vs_buildst",
    "classic_build_vs_lincr",
    "buildst_vs_lincr",
    "classic_build_vs_merge_trees",
    "classic_build_vs_lct",
]:
    num_true = df[col].sum()
    num_false = (~df[col]).sum()
    print(f"{col}:  {num_true} identical,  {num_false} different")

# --------------------------------------------------------------------------------------------------
#                                    PLOT 1: Boxplots
# --------------------------------------------------------------------------------------------------

algos_to_plot = ["BUILD", "BuildST", "Merge_Trees", "LinCR"]

fig, axs = plt.subplots(3, 4, sharex=False, sharey=True)

fig.set_size_inches(16, 11)

boxplot_props = {
    "boxprops": dict(linewidth=1, edgecolor="black"),
    "flierprops": dict(markerfacecolor="black", markeredgecolor="black", markersize=2),
    "medianprops": dict(linewidth=1.5, color="black", solid_capstyle="butt"),
    "whiskerprops": dict(linewidth=1, color="black", solid_capstyle="butt"),
    "capprops": dict(linewidth=1, color="black"),
}

for i, p in enumerate(CONRACTION_PROBABILITY_VALUES):
    df1 = df[(df["contraction_probability"] == p)]
    df2 = pd.melt(
        df1,
        id_vars=["num_leaves", "num_subtrees"],
        value_vars=algos_to_plot,
        var_name="time",
        value_name="value",
    )

    for j, algo in enumerate(algos_to_plot):
        ax = axs[i, j]

        df3 = df2[df2["time"] == algo]

        sns.boxplot(
            x="num_leaves",
            y="value",
            hue="num_subtrees",
            data=df3,
            ax=ax,
            order=sorted(df3["num_leaves"].unique()),
            **boxplot_props,
        )

        if j == 0:
            ax.set_ylabel(f"p = {p}\ntime [s]", fontsize=FONTSIZE)
        else:
            ax.set_ylabel("")

        if i == 2:
            ax.set_xlabel("num. of leaves", fontsize=FONTSIZE - 2)
        else:
            ax.set_xlabel("")

        ax.set_yscale("log")
        ax.tick_params(axis="both", which="both", labelsize=FONTSIZE - 4)
        ax.legend().set_visible(False)

        if i == 0:
            ax.set_title(algos_to_plot[j], fontsize=FONTSIZE)

        if i != 2:
            ax.set_xticklabels([])

        ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
        ax.grid(True, axis="y", color="lightgrey")

        for x in range(6):
            ax.axvline(x + 0.5, linewidth=0.75, color="black")

        for x, k in enumerate(sorted(df3["num_subtrees"].unique())):
            ax.text(
                -0.27 + x * 0.3,
                ax.get_ylim()[1] * 0.8,
                f"{k} trees",
                rotation=90,
                fontsize=11,
                horizontalalignment="center",
                verticalalignment="top",
            )

plt.tight_layout()
plt.savefig(PATH_BOXPLOT)


# --------------------------------------------------------------------------------------------------
#                                   PLOT 2: Median values
# --------------------------------------------------------------------------------------------------

algos_to_plot = ["BUILD", "BuildST", "Merge_Trees", "Loose_Cons", "LinCR"]

ratio = 0.75
fig, axs = plt.subplots(
    3,
    4,
    sharex=False,
    sharey=True,
    gridspec_kw={"width_ratios": [ratio, ratio, ratio, 1], "wspace": 0, "hspace": 0},
)

fig.set_size_inches(9, 9.5)
colors = ["firebrick", "lightcoral", "navy", "cornflowerblue", "limegreen"]
linestyle = "-"
linewidth = 1
markersize = 4

for i, p in enumerate(CONRACTION_PROBABILITY_VALUES):
    for j, k in enumerate(NUM_SUBTREES_VALUES):
        ax = axs[i, j]

        df1 = df[(df["contraction_probability"] == p) & (df["num_subtrees"] == k)]

        df_median = df1.groupby("num_leaves").median().reset_index()
        for algo_idx, algo in enumerate(algos_to_plot):
            ax.plot(
                df_median["num_leaves"],
                df_median[algo],
                marker="o",
                linestyle="-",
                linewidth=linewidth,
                markersize=markersize,
                color=colors[algo_idx],
            )

        if i == 2:
            ax.set_xlabel("num. of leaves", fontsize=FONTSIZE - 2)
        else:
            ax.set_xlabel("")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 5000)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", which="both", labelsize=FONTSIZE - 4)
        ax.set_xticks([10, 100, 1000])
        ax.grid(True, color="lightgrey")

        if i == 0:
            ax.set_title(f"{k} trees", fontsize=FONTSIZE)
        if i != 2:
            ax.set_xticklabels([])
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])

        if j > 0:
            ax.yaxis.set_ticks_position("none")

    axs[i, 0].set_ylabel(f"p = {p}\ntime [s]", fontsize=FONTSIZE)
    axs[i, -1].axis("off")

legend_elements = [
    Line2D(
        [0],
        [0],
        color=colors[i],
        marker="o",
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        label=algos_to_plot[i],
    )
    for i in range(len(algos_to_plot))
]

axs[0, -1].legend(
    handles=legend_elements, handlelength=1.0, fontsize=FONTSIZE - 2, title="", loc="upper right"
)

plt.tight_layout()
plt.savefig(PATH_PLOT_MEDIANS)
