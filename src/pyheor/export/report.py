"""
Markdown report generation for PyHEOR models.

Generates a self-contained analysis report with parameter tables,
base case results, OWSA tornado diagrams, and PSA summary.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file saving
import matplotlib.pyplot as plt


def generate_report(
    model,
    filepath: str = "report.md",
    *,
    wtp: float = 50000,
    n_sim: int = 1000,
    psa_seed: Optional[int] = 42,
    max_params: int = 10,
    run_psa: Optional[bool] = None,
    dpi: int = 150,
) -> str:
    """Generate a Markdown analysis report from a configured model.

    Runs base case, OWSA, and (optionally) PSA, then writes a ``.md``
    file with embedded tables and image references.

    Parameters
    ----------
    model : MarkovModel, PSMModel, MicroSimModel, or DESModel
        A fully configured model (parameters, transitions, costs, utilities).
    filepath : str
        Output path for the Markdown file (default: ``"report.md"``).
    wtp : float
        Willingness-to-pay threshold (default: 50,000).
    n_sim : int
        Number of PSA Monte Carlo simulations (default: 1,000).
    psa_seed : int, optional
        Random seed for PSA reproducibility (default: 42).
    max_params : int
        Maximum parameters shown in OWSA tornado (default: 10).
    run_psa : bool, optional
        Whether to run PSA. ``None`` (default) auto-detects: runs PSA
        only when at least one parameter has a ``dist`` defined.
    dpi : int
        Image resolution (default: 150).

    Returns
    -------
    str
        Absolute path of the generated report file.
    """
    from ..models.des import DESModel

    out = Path(filepath).resolve()
    img_dir = out.parent / f"{out.stem}_files"
    img_dir.mkdir(parents=True, exist_ok=True)
    rel_img = f"{out.stem}_files"  # relative path for markdown refs

    is_des = isinstance(model, DESModel)
    model_type = type(model).__name__
    has_dist = any(p.dist is not None for p in model.params.values())
    do_psa = has_dist if run_psa is None else run_psa

    # ── Run analyses ────────────────────────────────────────────────
    print(f"[report] Running base case...")
    base_result = model.run() if is_des else model.run_base_case()

    owsa_result = None
    if not is_des and model.params:
        print(f"[report] Running OWSA...")
        owsa_result = model.run_owsa(wtp=wtp)

    psa_result = None
    if do_psa:
        print(f"[report] Running PSA ({n_sim} simulations)...")
        psa_result = model.run_psa(n_sim=n_sim, seed=psa_seed)

    # ── Build sections ──────────────────────────────────────────────
    sections = []
    sections.append(f"# 卫生经济学分析报告\n")
    sections.append(
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"| 模型类型: {model_type}\n"
    )

    sections.append(_build_overview(model, is_des))
    sections.append(_build_params(model))
    sections.append(_build_base_case(base_result))

    if owsa_result is not None:
        sections.append(
            _build_owsa(owsa_result, img_dir, rel_img, max_params, dpi)
        )

    if psa_result is not None:
        sections.append(
            _build_psa(psa_result, img_dir, rel_img, wtp, dpi)
        )

    # ── Write ───────────────────────────────────────────────────────
    report = "\n".join(sections)
    out.write_text(report, encoding="utf-8")
    print(f"[report] Done → {out}")
    return str(out)


# =====================================================================
# Section builders
# =====================================================================

def _build_overview(model, is_des: bool) -> str:
    """Model overview section."""
    rows = [
        ("模型类型", type(model).__name__),
        ("健康状态", ", ".join(model.states)),
        ("治疗策略", ", ".join(
            f"{k} ({v})" if k != v else k
            for k, v in model.strategy_labels.items()
        )),
    ]
    if is_des:
        rows.append(("时间范围", f"{model.time_horizon} 年"))
    else:
        rows.append(("模拟周期", f"{model.n_cycles} × {model.cycle_length} 年"))
        hcc = getattr(model, '_hcc_method', None)
        rows.append(("半周期校正", hcc or "无"))

    rows.append(("费用贴现率", f"{model.dr_cost:.1%}"))
    rows.append(("效用贴现率", f"{model.dr_qaly:.1%}"))
    rows.append(("参数数量", str(len(model.params))))

    lines = ["## 1. 模型概述\n"]
    lines.append("| 属性 | 值 |")
    lines.append("|------|------|")
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def _build_params(model) -> str:
    """Parameter table section."""
    if not model.params:
        return "## 2. 参数表\n\n*无参数*\n"

    lines = ["## 2. 参数表\n"]
    lines.append("| 参数 | 标签 | 基准值 | 下限 | 上限 | 分布 |")
    lines.append("|------|------|--------|------|------|------|")
    for name, p in model.params.items():
        dist_str = repr(p.dist) if p.dist else "Fixed"
        lines.append(
            f"| {name} | {p.label} | {p.base} | "
            f"{p.low} | {p.high} | {dist_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_base_case(result) -> str:
    """Base case results section."""
    lines = ["## 3. 基础分析\n"]
    lines.append("### 结果汇总\n")
    lines.append(result.summary().to_markdown(index=False))
    lines.append("")

    try:
        icer_df = result.icer()
        lines.append("### 增量成本效果比 (ICER)\n")
        lines.append(icer_df.to_markdown(index=False))
        lines.append("")
    except Exception:
        pass

    return "\n".join(lines)


def _build_owsa(owsa_result, img_dir, rel_img, max_params, dpi) -> str:
    """OWSA section with tornado diagram."""
    lines = ["## 4. 单因素敏感性分析 (OWSA)\n"]

    # Tornado plot
    try:
        fig = owsa_result.plot_tornado(
            outcome="nmb", max_params=max_params,
        )
        tornado_path = img_dir / "tornado.png"
        fig.savefig(str(tornado_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        lines.append(f"### 龙卷风图 (Top {max_params})\n")
        lines.append(f"![tornado]({rel_img}/tornado.png)\n")
    except Exception:
        pass

    # Summary table
    try:
        summary = owsa_result.summary(outcome="nmb")
        # Show top N
        if len(summary) > max_params:
            summary = summary.head(max_params)
        lines.append("### 参数敏感性排序\n")
        lines.append(summary.to_markdown(index=False))
        lines.append("")
    except Exception:
        pass

    return "\n".join(lines)


def _build_psa(psa_result, img_dir, rel_img, wtp, dpi) -> str:
    """PSA section with CE plane and CEAC."""
    n = getattr(psa_result, 'n_sim', None) or getattr(psa_result, 'n_outer', 0)
    lines = [f"## 5. 概率敏感性分析 (PSA, n={n})\n"]

    # Summary
    try:
        lines.append("### 汇总统计\n")
        lines.append(psa_result.summary().to_markdown(index=False))
        lines.append("")
    except Exception:
        pass

    # Incremental analysis
    try:
        icer_df = psa_result.icer()
        lines.append("### 增量分析\n")
        lines.append(icer_df.to_markdown(index=False))
        lines.append("")
    except Exception:
        pass

    # CE plane
    try:
        fig = psa_result.plot_scatter(wtp=wtp)
        scatter_path = img_dir / "ce_plane.png"
        fig.savefig(str(scatter_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        lines.append("### 成本效果平面\n")
        lines.append(f"![ce_plane]({rel_img}/ce_plane.png)\n")
    except Exception:
        pass

    # CEAC
    try:
        fig = psa_result.plot_ceac()
        ceac_path = img_dir / "ceac.png"
        fig.savefig(str(ceac_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        lines.append("### 成本效果可接受曲线 (CEAC)\n")
        lines.append(f"![ceac]({rel_img}/ceac.png)\n")
    except Exception:
        pass

    return "\n".join(lines)
