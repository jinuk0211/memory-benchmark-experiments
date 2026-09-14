"""Reproduce LoCoMo token/F1 figures from audited scores and token accounting."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCORES = ROOT / "outputs/locomo_comparability_audit_20260911/scores_audit.json"
USAGE = ROOT / "experiments/refinement_usage_20260909/reports/current_verified/USAGE_REPORT.json"

# Token totals exclude embeddings. All source paths are relative to ROOT.
# These are selected-path accounting figures, not uniformly measured cold runs.
SPECS = [
    ("Naive", "Full-context reference", 54106911, "reference", "reported", "certmem_primary_results_r1/primary_results_summary.md", "Full-context input + output tokens"),
    ("Mem0", "Mem0", 27370629, "baseline", "reported", "results/locomo/mem0/mem0-final-report-50156979-r5.json", "usage.by_request_kind.chat_completion.total_tokens"),
    ("A-MEM", "A-MEM", 27285662, "baseline", "reported", "amem_final_results_r26/results_summary.md", "Reported LLM input + output tokens"),
    ("LangMem", "LangMem", 25942379, "baseline", "reported", "results/locomo/langmem/langmem-final-report-50156979-r1.json", "usage.by_request_kind.chat_completion.total_tokens"),
    ("SimpleMem", "SimpleMem", 19777906, "baseline", "reported", "simplemem-final-report-20260908-r25.json", "usage.by_request_kind.chat_completion.total_tokens"),
    ("LightMem_official", "LightMem (official)", 5300706, "baseline", "reported", "results/comparisons/all_methods_20260909/lightmem_official/FINAL_METRICS.json", "selected_build_and_qa_usage.total_tokens"),
    ("HiGMem", "HiGMem", 38514830, "baseline", "lower_bound", "results/locomo/higmem/higmem-final-report-20260907.json", "server_tokens.construction.total_tokens + server_tokens.qa.total_tokens; directory ledger includes prior invocations/retries; failed requests not fully metered"),
    ("E-Mem", "E-Mem", 60802774, "baseline", "lower_bound", "results/locomo/emem/emem-final-report-50156979-r11.json", "usage.by_request_kind.chat_completion.total_tokens; one request unmetered; includes metered failed attempts"),
    ("s_parent_single_2000", "Our method", 10356854, "ours", "derived", str(USAGE.relative_to(ROOT)), "831791 seed build + 795965 source QA + 5445573 utility + 3283525 logical reader (including cache reuse)"),
    ("LightMem_pipeline", "LightMem (pipeline)", 5582404, "variant", "reported", "results/locomo/lightmem/lightmem-final-report-20260907.json", "usage.by_request_kind.chat_completion.total_tokens"),
    ("LightMem_direct", "LightMem (direct)", 15267609, "variant", "reported", "results/locomo/lightmem_direct/report.json", "usage.by_request_kind.chat_completion.total_tokens"),
    ("CertMem_v15", "CertMem v15", 27163164, "variant", "reported", "certmem_primary_results_r1/primary_results_summary.md", "Reported full/certified LLM input + output tokens"),
    ("recursive_source_rehearsal_v1", "Recursive v1", 13149892, "internal", "derived", str(USAGE.relative_to(ROOT)), "7073329 shared preparation + 2779047 recursive preparation + 3297516 logical reader"),
]

PALETTES = {
    "dark": dict(bg="#10141c", fg="#edf2fb", muted="#a5b2c6", grid="#293546", base="#69bdd6", ours="#b2a2ff", internal="#cfb5ef", variant="#b0bacb", ref="#f2b56c", frontier="#80decc"),
    "light": dict(bg="#ffffff", fg="#182438", muted="#5c6b7d", grid="#dce3ec", base="#25779e", ours="#6941c6", internal="#9370b2", variant="#718096", ref="#b77824", frontier="#157d71"),
}


def frontier(rows: list[dict]) -> list[dict]:
    """Minimize accounted tokens, maximize F1; retain exact ties."""
    return sorted([
        row for row in rows
        if not any(
            other["llm_tokens"] <= row["llm_tokens"]
            and other["f1_percent"] >= row["f1_percent"]
            and (other["llm_tokens"] < row["llm_tokens"]
                 or other["f1_percent"] > row["f1_percent"])
            for other in rows
        )
    ], key=lambda row: row["llm_tokens"])


def build_rows() -> list[dict]:
    scores = json.loads(SCORES.read_text(encoding="utf-8-sig"))["methods"]
    specs = list(SPECS)
    additional = OUT / "additional_internal_tokens.json"
    if additional.exists():
        specs.extend(json.loads(additional.read_text(encoding="utf-8-sig")))
    rows = []
    for key, label, tokens, group, status, source, detail in specs:
        assert (ROOT / source).is_file(), source
        rows.append(dict(
            method=key, label=label, questions=1540,
            f1_percent=scores[key]["f1_percent"], llm_tokens=tokens,
            llm_tokens_m=tokens / 1e6, group=group, cost_status=status,
            main_table=key in {spec[0] for spec in SPECS[:9]},
            token_source=source, token_definition=detail,
            score_source=str(SCORES.relative_to(ROOT)),
        ))
    assert len({r["method"] for r in rows}) == len(rows)
    assert 831791 + 795965 + 5445573 + 3283525 == 10356854
    assert {r["method"] for r in frontier([r for r in rows if r["main_table"]])} == {
        "LightMem_official", "s_parent_single_2000", "E-Mem"
    }
    return rows


def style_axes(ax, palette: dict) -> None:
    ax.set_facecolor(palette["bg"])
    ax.tick_params(colors=palette["muted"], labelsize=11, length=0, pad=9)
    for name, spine in ax.spines.items():
        spine.set_color(palette["grid"])
        spine.set_visible(name in {"left", "bottom"})
    ax.grid(True, which="major", color=palette["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_xlim(90, 3.1)
    ax.set_ylim(26, 62)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([30, 35, 40, 45, 50, 55, 60])
    ax.set_xlabel("Selected build + QA path: LLM tokens (millions)   |   Fewer tokens →",
                  color=palette["fg"], fontsize=12, labelpad=17)
    ax.set_ylabel("LoCoMo F1 (%)   |   Higher is better", color=palette["fg"], fontsize=12, labelpad=17)


def plot(rows: list[dict], theme: str, expanded: bool) -> None:
    p = PALETTES[theme]
    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor(p["bg"])
    fig.subplots_adjust(left=.085, right=.965, bottom=.20, top=.84)
    style_axes(ax, p)
    title = "LoCoMo: quality vs. token usage"
    subtitle = f"{'All verified configurations' if expanded else 'Main-table comparison'}  ·  {len(rows)} points  ·  1,540 identical questions  ·  Qwen3.5-9B"
    fig.text(.085, .946, title, fontsize=23, weight="bold", color=p["fg"])
    fig.text(.085, .903, subtitle, fontsize=12, color=p["muted"])
    boundary = frontier(rows)
    for left, right in zip(boundary, boundary[1:]):
        bounded = "lower_bound" in {left["cost_status"], right["cost_status"]}
        ax.plot([left["llm_tokens_m"], right["llm_tokens_m"]],
                [left["f1_percent"], right["f1_percent"]],
                color=p["frontier"], linewidth=1.8,
                linestyle=(0, (4, 3)) if bounded else "-", zorder=2)
    offsets = {
        "Naive": (0, -30, "center"), "Mem0": (39, -18, "left"),
        "A-MEM": (-30, 19, "right"), "LangMem": (25, -12, "left"),
        "SimpleMem": (21, 13, "left"), "LightMem_official": (-12, -25, "right"),
        "HiGMem": (-13, 17, "right"), "E-Mem": (7, 22, "left"),
        "s_parent_single_2000": (16, 24, "left"),
        "LightMem_pipeline": (-16, 15, "right"),
        "LightMem_direct": (20, 20, "left"),
        "CertMem_v15": (13, 15, "left"),
        "recursive_source_rehearsal_v1": (-30, 42, "right"),
        "seed": (-3, 22, "center"), "r40_fused_four_turn": (-18, -29, "right"),
    }
    if expanded:
        offsets.update({
            "s_parent_single_2000": (18, -33, "left"),
            "LightMem_official": (10, -26, "left"),
            "SimpleMem": (16, -34, "left"),
            "Mem0": (-25, -30, "right"), "A-MEM": (-50, 13, "right"),
            "CertMem_v15": (0, 33, "center"),
        })
    frontier_keys = {r["method"] for r in boundary}
    for row in rows:
        key = row["method"]
        group = row["group"]
        color = p[{"baseline": "base", "ours": "ours", "internal": "internal",
                   "variant": "variant", "reference": "ref"}[group]]
        marker = "*" if group == "ours" else ("s" if group == "reference" else ("D" if group == "internal" else "o"))
        size = 240 if group == "ours" else 68
        x, y = row["llm_tokens_m"], row["f1_percent"]
        if key in frontier_keys:
            ax.scatter(x, y, s=size + 160, facecolors="none", edgecolors=p["frontier"], linewidths=.9, alpha=.6, zorder=3)
        ax.scatter(x, y, s=size, marker=marker, color=color,
                   edgecolor=p["bg"], linewidth=.8, zorder=5)
        if row["cost_status"] == "lower_bound":
            ax.annotate("", xy=(x * 1.20, y), xytext=(x * 1.025, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.1), zorder=4)
        dx, dy, ha = offsets[key]
        suffix = "†" if row["cost_status"] == "derived" else ""
        bound = "≥" if row["cost_status"] == "lower_bound" else ""
        text = f"{row['label']}{suffix}\n{y:.2f}  ·  {bound}{x:.2f} M"
        options = dict(
            xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
            va="center", fontsize=10.5 if expanded else 11,
            color=color if group == "ours" else p["fg"],
            fontweight="bold" if group == "ours" or key in frontier_keys else "normal",
            arrowprops=dict(arrowstyle="-", color=p["muted"], lw=.6, alpha=.65),
            zorder=6,
        )
        if group == "ours":
            options["bbox"] = dict(boxstyle="round,pad=.5", facecolor=p["bg"], edgecolor=p["ours"], linewidth=1.1)
        ax.annotate(text, **options)
    # The key remains in a quiet part of the plot for either point population.
    ax.plot([.028, .087], [.085, .085], transform=ax.transAxes, color=p["frontier"], lw=1.8)
    ax.text(.099, .085, "Pareto frontier at accounted token totals", transform=ax.transAxes,
            va="center", fontsize=10, color=p["muted"])
    fig.text(.085, .091,
             "† Reconstructed preparation + full logical reader tokens (including reused answers). Embedding tokens excluded.",
             fontsize=10, color=p["muted"])
    fig.text(.085, .063,
             "≥ Token lower bound; arrow points toward higher cost. Dashed frontier uses a lower-bound coordinate.",
             fontsize=10, color=p["muted"])
    fig.text(.085, .035,
             "Native pipeline accounting differs; the line joins observed configurations and does not imply interpolated performance.",
             fontsize=10, color=p["muted"])
    stem = f"tokens_{'all' if expanded else 'main'}_{theme}"
    fig.savefig(OUT / f"{stem}.png", dpi=180, facecolor=fig.get_facecolor())
    if theme == "light":
        fig.savefig(OUT / f"{stem}.pdf", facecolor=fig.get_facecolor())
        fig.savefig(OUT / f"{stem}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    rows = build_rows()
    main_rows = [r for r in rows if r["main_table"]]
    for theme in PALETTES:
        plot(main_rows, theme, expanded=False)
        plot(rows, theme, expanded=True)
    for selection, selected in [("main", main_rows), ("all", rows)]:
        keys = {r["method"] for r in frontier(selected)}
        for row in rows:
            row[f"frontier_{selection}"] = row["method"] in keys
    with (OUT / "token_points.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(
        metric="Official LoCoMo token-overlap F1 multiplied by 100",
        population="Same 1540 questions, categories 1-4, 10 conversations",
        cost_axis="Accounted selected build/preparation + QA LLM input/output tokens; not USD or end-to-end latency",
        frontier_definition="No other plotted point has both <= accounted tokens and >= F1, with at least one strict inequality",
        uncertainty="No confidence interval is shown; frontier is descriptive for selected runs. E-Mem and HiGMem coordinates are token lower bounds.",
        points=rows,
    )
    (OUT / "token_provenance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"main_points": len(main_rows), "all_points": len(rows),
                      "frontier_main": [r["label"] for r in frontier(main_rows)],
                      "frontier_all": [r["label"] for r in frontier(rows)]}, indent=2))


if __name__ == "__main__":
    main()