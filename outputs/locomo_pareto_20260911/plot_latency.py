"""Plot recorded LoCoMo latency without mixing incompatible timing scopes."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SCORES = ROOT / "outputs/locomo_comparability_audit_20260911/scores_audit.json"
INTERNAL = ROOT / "experiments/refinement_usage_20260909/CURRENT_METHODS.csv"


@dataclass(frozen=True)
class Point:
    method: str
    label: str
    seconds: float
    f1_percent: float
    timing_scope: str
    source: str


def load_points() -> list[Point]:
    scores = json.loads(SCORES.read_text(encoding="utf-8-sig"))["methods"]
    native = [
        ("Mem0", "Mem0", 0.3200398810498126,
         "results/locomo/mem0/mem0-final-report-50156979-r5.json"),
        ("LangMem", "LangMem", 0.4096584146673029,
         "results/locomo/langmem/langmem-final-report-50156979-r1.json"),
        ("A-MEM", "A-MEM", 0.6904868276088268,
         "amem_final_results_r26/amem-r26-latency-addendum-r1.json"),
        ("SimpleMem", "SimpleMem", 13.778145232757964,
         "simplemem-final-report-20260908-r25.json"),
        ("E-Mem", "E-Mem", 36.39571874373919,
         "results/locomo/emem/emem-final-report-50156979-r11.json"),
    ]
    points = [
        Point(method, label, seconds, scores[method]["f1_percent"],
              "saved_native_question_timer", source)
        for method, label, seconds, source in native
    ]
    points.append(Point(
        "LightMem_official", "LightMem", 1.080323854039496,
        scores["LightMem_official"]["f1_percent"],
        "saved_retrieval_plus_answer_api_component_sum",
        "upload_preparation_20260910/extracted_baseline_results/lightmem_official"))
    labels = {
        "seed": "Seed",
        "r40_fused_four_turn": "R40",
        "s_parent_single_2000": "Our method",
        "recursive_source_rehearsal_v1": "Recursive v1",
    }
    with INTERNAL.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            assert row["questions"] == "1540"
            assert abs(float(row["official_f1_100"]) -
                       scores[method]["f1_percent"]) < 1e-10
            points.append(Point(
                method, labels[method],
                float(row["amortized_native_inference_seconds_per_question"]),
                scores[method]["f1_percent"],
                "observed_batch_amortized_llm_and_embedding_call_time",
                str(INTERNAL.relative_to(ROOT))))
    assert len(points) == 10
    return points


def nondominated(points: list[Point]) -> list[Point]:
    """Minimize time and maximize F1; retain ties only if not strictly dominated."""
    return sorted(
        [point for point in points if not any(
            other.seconds <= point.seconds
            and other.f1_percent >= point.f1_percent
            and (other.seconds < point.seconds
                 or other.f1_percent > point.f1_percent)
            for other in points)],
        key=lambda point: point.seconds,
        reverse=True,
    )


def draw(points: list[Point], *, dark: bool) -> None:
    colors = {
        "bg": "#10141c" if dark else "#ffffff",
        "fg": "#edf2fb" if dark else "#172438",
        "muted": "#9dacbf" if dark else "#5f6d80",
        "grid": "#273344" if dark else "#e1e7ef",
        "baseline": "#69bdd6" if dark else "#257b98",
        "ours": "#ad9cff" if dark else "#6750ae",
        "frontier": "#69d2bb" if dark else "#21876f",
        "light": "#ecb875" if dark else "#b07528",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": colors["fg"], "axes.labelcolor": colors["fg"],
        "xtick.color": colors["muted"], "ytick.color": colors["muted"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.65, 1.2]})
    fig.set_facecolor(colors["bg"])
    fig.subplots_adjust(left=.07, right=.965, top=.77, bottom=.28, wspace=.23)

    for ax in axes:
        ax.set_facecolor(colors["bg"])
        ax.grid(True, color=colors["grid"], linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(colors["grid"])
        ax.tick_params(length=0, pad=8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))

    left, right = axes
    left.set_xlim(70, .18)
    left.set_ylim(27.5, 60)
    left.set_yticks([30, 35, 40, 45, 50, 55, 60])
    left.xaxis.set_major_locator(FixedLocator([.2, .5, 1, 2, 5, 10, 20, 50]))
    left.set_ylabel("LoCoMo F1 (%)", labelpad=10)
    left.set_xlabel("Recorded mean seconds / question  →  faster", labelpad=13)
    left.set_title("BASELINES  ·  6 recorded configurations",
                   fontsize=11, fontweight="bold", loc="left", pad=21)
    offsets = {
        "Mem0": (-4, 18, "center"), "LangMem": (0, 15, "center"),
        "A-MEM": (-24, -14, "right"), "SimpleMem": (0, 16, "center"),
        "E-Mem": (13, 4, "left"), "LightMem_official": (0, 16, "center"),
    }
    for point in points[:6]:
        component = point.method == "LightMem_official"
        color = colors["light"] if component else colors["baseline"]
        left.scatter(point.seconds, point.f1_percent, s=75 if component else 60,
                     marker="D" if component else "o", color=color,
                     edgecolors=colors["bg"], linewidths=1.1, zorder=4)
        dx, dy, align = offsets[point.method]
        suffix = " (official)" if component else ""
        left.annotate(
            f"{point.label}{suffix}\n{point.f1_percent:.2f}  ·  {point.seconds:.3f}s",
            (point.seconds, point.f1_percent), xytext=(dx, dy),
            textcoords="offset points", ha=align,
            va="bottom" if dy > 0 else "top", fontsize=9.2,
            color=colors["fg"], linespacing=1.5)
    left.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=6,
               color=colors["baseline"], label="Native question timer"),
        Line2D([], [], linestyle="none", marker="D", markersize=6,
               color=colors["light"], label="Retrieval + answer API components"),
    ], loc="upper right", frameon=False, fontsize=8.5, labelcolor=colors["muted"])

    internal = points[6:]
    frontier = nondominated(internal)
    assert [point.method for point in frontier] == [
        "seed", "recursive_source_rehearsal_v1", "s_parent_single_2000"]
    right.set_xlim(.445, .276)
    right.set_ylim(55.22, 56.59)
    right.set_yticks([55.25, 55.5, 55.75, 56, 56.25, 56.5])
    right.xaxis.set_major_locator(FixedLocator([.28, .30, .32, .35, .38, .42]))
    right.set_ylabel("LoCoMo F1 (%) · zoomed scale", labelpad=10)
    right.set_xlabel("Amortized model-call seconds / question  →  faster", labelpad=13)
    right.set_title("OUR VARIANTS  ·  4 recorded configurations",
                    fontsize=11, fontweight="bold", loc="left", pad=21)
    right.plot([point.seconds for point in frontier],
               [point.f1_percent for point in frontier],
               color=colors["frontier"], linewidth=1.8, zorder=2)
    inner_offsets = {
        "seed": (8, 15, "left"),
        "r40_fused_four_turn": (0, -14, "center"),
        "s_parent_single_2000": (0, -16, "center"),
        "recursive_source_rehearsal_v1": (8, 11, "left"),
    }
    for point in internal:
        is_ours = point.method == "s_parent_single_2000"
        color = colors["ours"] if is_ours else colors["baseline"]
        right.scatter(point.seconds, point.f1_percent, s=180 if is_ours else 65,
                      marker="*" if is_ours else "o", color=color,
                      edgecolors=colors["bg"], linewidths=1, zorder=4)
        dx, dy, align = inner_offsets[point.method]
        right.annotate(
            f"{point.label}\n{point.f1_percent:.2f}  ·  {point.seconds:.3f}s",
            (point.seconds, point.f1_percent), xytext=(dx, dy),
            textcoords="offset points", ha=align,
            va="bottom" if dy > 0 else "top",
            fontsize=9.5, fontweight="bold" if is_ours else "normal",
            color=color if is_ours else colors["fg"], linespacing=1.5)
    right.legend(handles=[
        Line2D([], [], color=colors["frontier"], linewidth=1.8,
               label="Frontier of these observed measurements"),
    ], loc="upper right", frameon=False, fontsize=8.1, labelcolor=colors["muted"])

    fig.text(.07, .94, "LoCoMo  /  Quality and recorded latency",
             fontsize=21, fontweight="bold", va="top")
    fig.text(.07, .882,
             "Qwen3.5-9B  ·  Same 1,540 questions  ·  10 configurations with available timing",
             fontsize=11, color=colors["muted"])
    fig.text(.07, .174,
             "Timing scopes differ, so the panels do not share a latency frontier.",
             fontsize=10.4, fontweight="bold")
    fig.text(.07, .133,
             "Baseline times follow each saved implementation; LightMem sums recorded retrieval and answer API components.",
             fontsize=9.1, color=colors["muted"])
    fig.text(.07, .099,
             "Our variants use observed LLM + embedding call time / 1,540, with batching and cache reuse; this is not end-to-end query latency.",
             fontsize=9.1, color=colors["muted"])
    fig.text(.07, .065,
             "End-to-end question times are unavailable for Our method, Full context and HiGMem. F1 ranges differ between panels.",
             fontsize=9.1, color=colors["muted"])

    stem = "latency_dark" if dark else "latency_paper"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, facecolor=colors["bg"])
    if not dark:
        fig.savefig(OUTPUT / f"{stem}.pdf", facecolor=colors["bg"])
        fig.savefig(OUTPUT / f"{stem}.svg", facecolor=colors["bg"])
    plt.close(fig)


def main() -> None:
    points = load_points()
    with (OUTPUT / "latency_points.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0])))
        writer.writeheader()
        writer.writerows(asdict(point) for point in points)
    draw(points, dark=True)
    draw(points, dark=False)
    print(json.dumps({
        "configurations": len(points),
        "baseline_configurations": 6,
        "internal_configurations": 4,
        "internal_frontier": [point.label for point in nondominated(points[6:])],
        "outputs": ["latency_dark.png", "latency_paper.png", "latency_paper.pdf",
                    "latency_paper.svg", "latency_points.csv"],
    }, indent=2))


if __name__ == "__main__":
    main()
