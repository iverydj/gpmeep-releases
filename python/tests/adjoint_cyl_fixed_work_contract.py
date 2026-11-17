"""Pure output contract for the fixed-work cylindrical-adjoint validation."""

from __future__ import annotations

import json
from typing import Any, Iterable


def fixed_work_metrics_payload(
    rows: Iterable[dict[str, Any]],
    *,
    is_master: bool,
    expected_case_count: int,
    prefix: str,
) -> str | None:
    """Validate every rank's census and emit one master-rank JSON payload."""
    ordered = sorted(rows, key=lambda row: row["case_index"])
    if [row["case_index"] for row in ordered] != list(range(expected_case_count)):
        raise RuntimeError("fixed-work cylindrical adjoint metric census differs")
    if not is_master:
        return None
    return prefix + json.dumps(
        {
            "schema_version": 1,
            "fixed_work": 1,
            "case_count": len(ordered),
            "cases": ordered,
        },
        sort_keys=True,
        allow_nan=False,
    )
