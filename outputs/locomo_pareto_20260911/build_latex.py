"""Generate editable PGFPlots token figures from the audited CSV (no image includes)."""
from __future__ import annotations

import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "latex"

STYLE = r"""% Add this file in the document preamble (before \begin{document}).
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{arrows.meta,plotmarks}
\definecolor{locomoText}{HTML}{182438}
\definecolor{locomoMuted}{HTML}{5C6B7D}
\definecolor{locomoGrid}{HTML}{DCE3EC}
\definecolor{locomoBase}{HTML}{25779E}
\definecolor{locomoOur}{HTML}{6941C6}
\definecolor{locomoInternal}{HTML}{9370B2}
\definecolor{locomoVariant}{HTML}{718096}
\definecolor{locomoRef}{HTML}{B77824}
\definecolor{locomoFront}{HTML}{157D71}
\pgfplotsset{locomo axis/.style={
    axis lines=left, axis line style={locomoMuted,-},
    grid=major, major grid style={locomoGrid,line width=0.3pt},
    tick align=outside, tick style={locomoMuted},
    tick label style={font=\scriptsize,text=locomoMuted},
    label style={font=\small,text=locomoText},
    title style={font=\small\bfseries,text=locomoText},
    clip=false, minor x tick num=0,
    scaled ticks=false, enlargelimits=false,
}}
\tikzset{
    locomo label/.style={font=\scriptsize,text=locomoText,align=left,inner sep=1pt},
    locomo leader/.style={draw=locomoMuted,line width=0.3pt},
}
"""

CAPTIONS = {
    "tokens_main": r"Quality--token trade-off for the nine main-table configurations on the same 1,540 LoCoMo questions with Qwen3.5-9B. The reversed logarithmic axis shows selected build/preparation plus QA LLM input and output tokens, excluding embeddings. $\dagger$ denotes reconstructed preparation plus cache-normalized logical reader usage. $\geq$ denotes a recorded token lower bound; arrows point toward higher cost, and dashed segments involve a lower-bound coordinate. The frontier describes the plotted accounting figures under native pipeline conditions.",
    "tokens_all": r"Quality--token trade-off across all 15 verified configurations, including internal variants and alternative LightMem pipelines. Under the displayed selected-path accounting, R40, Seed, and E-Mem are nondominated. Internal costs ($\dagger$) reconstruct required preparation plus full logical reader usage, including reused answers; embedding tokens are excluded. E-Mem and HiGMem show recorded token lower bounds ($\geq$). The connecting line joins observed configurations and does not imply achievable interpolated performance.",
    "latency": r"Quality and recorded latency for ten configurations with available timing. The left panel shows native question timers; LightMem is shown separately as the sum of recorded retrieval and answer-API components. The right panel uses observed LLM-plus-embedding call time divided by 1,540 for four internal variants and a zoomed F1 axis. Batching and cache reuse affect this amortized metric, which is not end-to-end query latency. No frontier is shared across panels or baseline timing scopes.",
}


def nondominated(rows: list[dict]) -> list[dict]:
    return sorted([
        row for row in rows if not any(
            other["tokens"] <= row["tokens"] and other["f1"] >= row["f1"]
            and (other["tokens"] < row["tokens"] or other["f1"] > row["f1"])
            for other in rows
        )
    ], key=lambda row: row["tokens"])


def xy(row: dict) -> str:
    return f"({row['tokens'] / 1e6:.6f},{row['f1']:.12f})"


def axis_xy(row: dict) -> str:
    return "(axis cs:" + xy(row)[1:]


def figure(rows: list[dict], expanded: bool) -> str:
    lines = [
        "% Generated from token_points.csv; coordinates retain unrounded F1.",
        "% All drawing is native PGFPlots/TikZ; no image or runtime CSV dependency.",
        r"\begin{tikzpicture}", r"\begin{axis}[locomo axis,",
        r"  width=\linewidth, height=9.3cm,",
        r"  xmode=log, log basis x=10, x dir=reverse,",
        r"  xmin=3.1, xmax=90, ymin=26, ymax=62,",
        r"  xtick={5,10,20,40,80}, xticklabels={5,10,20,40,80},",
        r"  ytick={30,35,40,45,50,55,60},",
        r"  xlabel={Selected build + QA LLM tokens (millions; fewer $\rightarrow$)},",
        r"  ylabel={LoCoMo F1 (\%)},",
        "  title={" + ("All 15 verified configurations" if expanded else "Main-table comparison: 9 configurations") + "},",
        r"]",
    ]
    front = nondominated(rows)
    front_names = {row["method"] for row in front}
    lines.append("% Pareto frontier: " + " -> ".join(row["method"] for row in front))
    for a, b in zip(front, front[1:]):
        dashed = ",dashed" if "lower_bound" in {a["cost_status"], b["cost_status"]} else ""
        lines.append(r"\addplot[locomoFront,line width=0.9pt,mark=none" + dashed + "] coordinates {" + xy(a) + " " + xy(b) + "};")
    offsets = {
        "Naive": (0, -18, "north"),
        "Mem0": (20, -9, "west"), "A-MEM": (-18, 10, "east"),
        "LangMem": (14, -7, "west"), "SimpleMem": (13, 9, "west"),
        "LightMem_official": (-7, -13, "north east"),
        "HiGMem": (-9, 12, "east"), "E-Mem": (6, 16, "south west"),
        "s_parent_single_2000": (10, 15, "south west"),
        "LightMem_pipeline": (-9, 12, "south east"),
        "LightMem_direct": (12, 13, "south west"),
        "CertMem_v15": (0, 20, "south"),
        "recursive_source_rehearsal_v1": (-18, 24, "south east"),
        "seed": (-3, 18, "south"), "r40_fused_four_turn": (-10, -14, "north east"),
    }
    if expanded:
        offsets.update({
            "Mem0": (-13, -19, "north east"),
            "A-MEM": (-26, 9, "east"),
            "SimpleMem": (12, -20, "north west"),
            "LightMem_official": (7, -15, "north west"),
            "s_parent_single_2000": (10, -18, "north west"),
        })
    for i, row in enumerate(rows):
        key = row["method"]
        x, y = row["tokens"] / 1e6, row["f1"]
        color = {"baseline": "locomoBase", "ours": "locomoOur", "reference": "locomoRef", "internal": "locomoInternal", "variant": "locomoVariant"}[row["group"]]
        marker = {"ours": "star", "reference": "square*", "internal": "diamond*"}.get(row["group"], "*")
        size = "3.2pt" if row["group"] == "ours" else "1.9pt"
        lines.append("% point: " + key)
        if key in front_names:
            lines.append(r"\addplot[only marks,mark=o,mark size=3.8pt,locomoFront,thin] coordinates {" + xy(row) + "};")
        lines.append(r"\addplot[only marks,mark=" + marker + ",mark size=" + size + ",color=" + color + ",mark options={solid,fill=" + color + "}] coordinates {" + xy(row) + "};")
        if row["cost_status"] == "lower_bound":
            lines.append(r"\draw[-{Stealth[length=3pt]}," + color + ",thin] (axis cs:" + f"{x*1.025:.6f},{y:.12f}) -- (axis cs:{x*1.2:.6f},{y:.12f});")
        dx, dy, anchor = offsets[key]
        label = row["label"] + (r"$^{\dagger}$" if row["cost_status"] == "derived" else "")
        if key in front_names or row["group"] == "ours":
            label = r"\textbf{" + label + "}"
        bound = r"$\geq$" if row["cost_status"] == "lower_bound" else ""
        text = label + r"\\" + f"{y:.2f}" + r"\,\textperiodcentered\," + bound + f"{x:.2f}" + r"\,M"
        extra = ",draw=locomoOur,rounded corners=2pt,fill=white,inner sep=3pt,text=locomoOur" if row["group"] == "ours" else ""
        lines.append(r"\node[locomo label,anchor=" + anchor + extra + "] (lp" + str(i) + ") at ([xshift=" + str(dx) + "pt,yshift=" + str(dy) + "pt]axis cs:" + f"{x:.6f},{y:.12f}) {{{text}}};")
        lines.append(r"\draw[locomo leader] " + axis_xy(row) + " -- (lp" + str(i) + ");")
    lines.extend([
        r"\draw[locomoFront,line width=0.9pt] (axis description cs:0.025,0.07) -- (axis description cs:0.080,0.07);",
        r"\node[locomo label,text=locomoMuted,anchor=west] at (axis description cs:0.092,0.07) {Pareto frontier at accounted token totals};",
        r"\end{axis}", r"\end{tikzpicture}", "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with (BASE / "token_points.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["tokens"] = int(row["llm_tokens"])
        row["f1"] = float(row["f1_percent"])
    selected = [row for row in rows if row["main_table"] == "True"]
    assert len(rows) == 15 and len(selected) == 9
    assert {row["method"] for row in nondominated(rows)} == {"r40_fused_four_turn", "seed", "E-Mem"}
    assert {row["method"] for row in nondominated(selected)} == {"LightMem_official", "s_parent_single_2000", "E-Mem"}
    (OUT / "locomo_style.tex").write_text(STYLE, encoding="utf-8")
    for name, subset, expanded in [("tokens_main", selected, False), ("tokens_all", rows, True)]:
        (OUT / f"{name}.tikz.tex").write_text(figure(subset, expanded), encoding="utf-8")
    document = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[a4paper,margin=16mm]{geometry}",
        r"\input{locomo_style.tex}",
        r"\pagestyle{empty}", r"\begin{document}",
    ]
    for name in ("tokens_all", "tokens_main", "latency"):
        document.extend([
            r"\begin{figure}[p]", r"\centering", r"\input{" + name + ".tikz.tex}",
            r"\caption{" + CAPTIONS[name] + "}",
            r"\label{fig:locomo-" + name.replace("_", "-") + "}",
            r"\end{figure}", r"\clearpage",
        ])
    document.extend([r"\end{document}", ""])
    (OUT / "main.tex").write_text("\n".join(document), encoding="utf-8")
    print("Generated editable PGFPlots files: main9 and all15; standalone main.tex includes latency fragment.")


if __name__ == "__main__":
    main()