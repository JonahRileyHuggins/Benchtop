"""Read-only ASCII preview of a PEtab-style experiment design.

Loads benchmark YAML via FileLoader and prints the condition graph.
Does not construct Experiment, create a cache, or run simulations.
"""

from __future__ import annotations

import os
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd

from benchtop.file_loader import FileLoader

_SEPARATOR = "=" * 64
_NAME_WIDTH = 36


def render_design(yaml_path: str, verbose: bool = False) -> str:
    """Return an ASCII experiment-design preview for ``yaml_path``."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"{yaml_path} is not a valid benchmark")

    loader = FileLoader(yaml_path)
    loader._petab_files()

    param_name = os.path.basename(loader.config.parameter_file)
    lines = [
        f"Design: {yaml_path}",
        f"problems: {len(loader.problems)}  |  parameters: {param_name}",
        "(preview only -- not simulating)",
        "",
    ]

    total = 0
    for problem in loader.problems:
        block, n_sims = _render_problem(problem, verbose=verbose)
        lines.append(block)
        total += n_sims

    lines.append(f"TOTAL: {total} simulations")
    return "\n".join(lines) + "\n"


def print_design(yaml_path: str, verbose: bool = False) -> None:
    """Print :func:`render_design` to stdout."""
    print(render_design(yaml_path, verbose=verbose), end="")


def _render_problem(problem: SimpleNamespace, verbose: bool = False) -> tuple[str, int]:
    measurements = _first_frame(getattr(problem, "measurement_files", None))
    conditions = _first_frame(getattr(problem, "condition_files", None))
    children, roots, all_nodes = _build_graph(measurements)
    cell_count = problem.cell_count
    n_conds = len(all_nodes)
    n_sims = n_conds * cell_count
    cell_word = "cell" if cell_count == 1 else "cells"

    sbml_files = getattr(problem, "sbml_files", None) or []
    sbml = ", ".join(os.path.basename(p) for p in sbml_files) or "(none)"
    simulator = problem.simulator or "tellurium"

    lines = [
        _SEPARATOR,
        f"PROBLEM  {problem.name}",
        f"  simulator  {simulator}",
        f"  cells      {cell_count}",
        f"  sbml       {sbml}",
        "",
    ]

    node_info = {
        name: _node_info(name, measurements, children, cell_count)
        for name in all_nodes
    }
    if roots:
        lines.extend(_render_forest(roots, children, node_info))
    else:
        lines.append("  (no simulation conditions)")

    lines.append("")
    lines.append(
        f"  simulations: {n_conds} conditions x {cell_count} {cell_word} = {n_sims}"
    )

    for warning in _condition_warnings(conditions, all_nodes):
        lines.append(f"  warning: {warning}")

    if verbose and not conditions.empty:
        lines.append("")
        lines.append(conditions.to_string(index=False))

    lines.append("")
    return "\n".join(lines), n_sims


def _first_frame(files) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    first = files[0]
    if isinstance(first, pd.DataFrame):
        return first
    return pd.DataFrame()


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _build_graph(measurements: pd.DataFrame) -> tuple[dict, list, list]:
    """Return ``(children, roots, all_nodes)`` in first-appearance order."""
    children: dict[str, list] = defaultdict(list)
    has_parent: set[str] = set()
    sim_ids: list[str] = []
    pre_ids: list[str] = []

    if measurements.empty or "simulationConditionId" not in measurements.columns:
        return children, [], []

    for sid in measurements["simulationConditionId"]:
        if _is_missing(sid):
            continue
        if sid not in sim_ids:
            sim_ids.append(sid)

    if "preequilibrationConditionId" in measurements.columns:
        for _, row in measurements.iterrows():
            pre = row["preequilibrationConditionId"]
            sim = row["simulationConditionId"]
            if _is_missing(pre) or _is_missing(sim):
                continue
            if sim not in children[pre]:
                children[pre].append(sim)
            has_parent.add(sim)
            if pre not in pre_ids:
                pre_ids.append(pre)

    all_nodes = list(sim_ids)
    for pre in pre_ids:
        if pre not in all_nodes:
            all_nodes.append(pre)

    roots = [name for name in all_nodes if name not in has_parent]
    return children, roots, all_nodes


def _node_info(
    name: str,
    measurements: pd.DataFrame,
    children: dict,
    cell_count: int,
) -> dict:
    return {
        "role": "preeq" if children.get(name) else "sim",
        "tspan": _time_span(measurements, name),
        "cells": cell_count,
        "obs_lines": _observable_lines(measurements, name),
        "label": _truncate(str(name), _NAME_WIDTH),
    }


def _fmt_num(value) -> str:
    if _is_missing(value):
        return "?"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _time_span(measurements: pd.DataFrame, condition: str) -> str:
    if measurements.empty or "time" not in measurements.columns:
        return "t=?"
    rows = measurements[measurements["simulationConditionId"] == condition]
    if rows.empty:
        return "t=?"
    max_time = rows["time"].max()
    if _is_missing(max_time):
        return "t=?"
    return f"t=0..{_fmt_num(max_time)}"


def _observable_lines(measurements: pd.DataFrame, condition: str) -> list[str]:
    if measurements.empty or "observableId" not in measurements.columns:
        return []
    rows = measurements[measurements["simulationConditionId"] == condition]
    if rows.empty:
        return []

    lines = []
    for obs_id in rows["observableId"].drop_duplicates():
        if _is_missing(obs_id):
            continue
        obs_rows = rows[rows["observableId"] == obs_id]
        times = [
            _fmt_num(t) for t in obs_rows["time"].tolist() if not _is_missing(t)
        ]
        unique_times = list(dict.fromkeys(times))
        label = _truncate(str(obs_id), 40)
        if unique_times:
            lines.append(f"{label} @ {', '.join(unique_times)}")
        else:
            lines.append(label)
    return lines


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _render_forest(roots: list, children: dict, node_info: dict) -> list[str]:
    lines: list[str] = []
    for index, root in enumerate(roots):
        if index:
            lines.append("")
        _walk_node(
            name=root,
            children=children,
            node_info=node_info,
            prefix="  ",
            is_last=True,
            is_root=True,
            lines=lines,
            seen=set(),
        )
    return lines


def _walk_node(
    name: str,
    children: dict,
    node_info: dict,
    prefix: str,
    is_last: bool,
    is_root: bool,
    lines: list[str],
    seen: set,
) -> None:
    if name in seen:
        lines.append(f"{prefix}+-- {name}  (cycle)")
        return
    seen.add(name)

    info = node_info[name]
    kids = children.get(name, [])

    if is_root:
        line_prefix = prefix
        child_prefix = prefix
        obs_prefix = prefix + ("|  " if kids else "   ")
        stem = prefix + "|"
    else:
        line_prefix = prefix + "+-- "
        child_prefix = prefix + ("    " if is_last else "|   ")
        obs_prefix = child_prefix + "  "
        stem = prefix + "|"

    name_width = _NAME_WIDTH + 4 if is_root else _NAME_WIDTH
    padded = f"{info['label']:<{name_width}}"
    lines.append(
        f"{line_prefix}{padded}{info['role']:<6} {info['tspan']}  x{info['cells']}"
    )

    for i, obs in enumerate(info["obs_lines"]):
        if i == 0:
            lines.append(f"{obs_prefix}obs: {obs}")
        else:
            lines.append(f"{obs_prefix}     {obs}")

    if not kids:
        return

    lines.append(stem)
    for i, child in enumerate(kids):
        if i:
            lines.append(stem)
        _walk_node(
            name=child,
            children=children,
            node_info=node_info,
            prefix=child_prefix,
            is_last=i == len(kids) - 1,
            is_root=False,
            lines=lines,
            seen=seen,
        )


def _condition_warnings(conditions: pd.DataFrame, all_nodes: list) -> list[str]:
    cond_ids: set[str] = set()
    if not conditions.empty and "conditionId" in conditions.columns:
        cond_ids = {
            cid for cid in conditions["conditionId"].tolist() if not _is_missing(cid)
        }

    warnings = []
    missing = [cid for cid in all_nodes if cid not in cond_ids]
    unused = sorted(cond_ids - set(all_nodes))
    if missing:
        warnings.append(
            "measurements reference unknown conditions: " + ", ".join(missing)
        )
    if unused:
        warnings.append(
            "conditions unused in measurements: " + ", ".join(unused)
        )
    return warnings
