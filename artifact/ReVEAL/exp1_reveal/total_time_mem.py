import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba, to_rgb

categories = ['dc2_64', 'resyn3_64']
solvers = ['fraig', '&cec', 'kissat', 'cec', 'fraig', '&cec', '&fraig']
values = [12, 46, 6, 1, 8, 50, 5]

grouped_values = [
    values[:3],  # dc2_64 
    values[3:]   # resyn3_64 
]

original_colors = ['#d3d3d3', '#f4b183', '#ff9999', '#9bbb59', '#d3d3d3', '#f4b183', '#77dd77']

#  ()
def desaturate_color(color, factor):
    rgb = np.array(to_rgb(color))  #  RGB
    return tuple(rgb * factor) + (1.0,)  #  RGB  alpha 

colors = [desaturate_color(c, 0.3) for c in original_colors]  #  30%

fig, ax = plt.subplots(figsize=(10, 5))  # 
bar_width = 0.2  # 
group_spacing = 0.8  # 

for group_idx, group in enumerate(categories):
    x_base = group_idx * group_spacing  #  x 
    for idx, value in enumerate(grouped_values[group_idx]):
        x = x_base + idx * bar_width  #  x 
        ax.bar(
            x, 
            value, 
            bar_width, 
            color=colors[idx + (group_idx * 3)],  # 
            edgecolor="black",
            linewidth=0.8
        )
        ax.text(
            x,
            value + 1,
            str(value),
            ha="center",
            va="bottom",
            fontsize=12,
            color="#2f2f2f",
        )

#  x 
group_centers = [group_spacing * i + (bar_width * len(grouped_values[i]) / 2) for i in range(len(categories))]
ax.set_xticks(group_centers)
ax.set_xticklabels(categories, fontsize=16, fontweight="bold")
ax.set_ylabel("Optimal Count", fontsize=16, fontweight="bold", labelpad=10)
ax.set_ylim(0, 60)  #  y 
ax.yaxis.set_tick_params(labelsize=12)

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=colors[0], edgecolor="black", linewidth=0.8, label="fraig"),
    plt.Rectangle((0, 0), 1, 1, fc=colors[1], edgecolor="black", linewidth=0.8, label="&cec"),
    plt.Rectangle((0, 0), 1, 1, fc=colors[2], edgecolor="black", linewidth=0.8, label="kissat"),
    plt.Rectangle((0, 0), 1, 1, fc=colors[3], edgecolor="black", linewidth=0.8, label="cec"),
    plt.Rectangle((0, 0), 1, 1, fc=colors[4], edgecolor="black", linewidth=0.8, label="fraig"),
    plt.Rectangle((0, 0), 1, 1, fc=colors[5], edgecolor="black", linewidth=0.8, label="&cec"),
    plt.Rectangle((0, 0), 1, 1, fc=colors[6], edgecolor="black", linewidth=0.8, label="&fraig")
]
ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#404040",
    fontsize=10,
    title="Solvers",
    title_fontsize=11,
)

plt.tight_layout(pad=1.5)

#  PDF 
output_pdf = "solver_optimal_count_comparison.pdf"
with PdfPages(output_pdf) as pdf:
    pdf.savefig(fig, bbox_inches="tight")

print(f": {output_pdf}")
