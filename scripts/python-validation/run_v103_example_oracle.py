#!/usr/bin/env python3
"""Run the generic oracle only after proving the frozen gpmeep v1.0.3 runtime."""

from __future__ import annotations

import run_example_oracle as oracle


if __name__ == "__main__":
    raise SystemExit(
        oracle.main(
            _frozen_v103_runtime_token=oracle._FROZEN_V103_RUNTIME_TOKEN
        )
    )
