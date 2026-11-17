#!/usr/bin/env python3
"""Run a Meep example with deterministic physical and result observables.

The driver wraps one explicitly selected synchronous Simulation operation and
records scalar observables immediately after every completed operation, before
an example can reset or discard its fields.  The default is ``Simulation.run``;
``Simulation.run_k_point`` is also supported for examples whose public
``run_k_points`` loop bypasses ``run`` after its recommendation guard.  The
driver does not alter the timestep loop.  Selected top-level numerical result
vectors can also be captured after the example completes, allowing examples'
published spectra, coefficients, or other derived observables to be compared
without parsing presentation-oriented stdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import operator
import pathlib
import runpy
import sys
import time
import unittest
from typing import Any


SCRIPTS_DIRECTORY = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
import gpmeep_backend_guard as qualification_guard  # noqa: E402


METRIC_PREFIX = "gpmeep-generic-example-metrics:"
AUDIT_PREFIX = "gpmeep-generic-example-audit:"
CHECKPOINT_PREFIX = "gpmeep-generic-example-run-checkpoint:"
AUXILIARY_GUARD_PREFIX = "gpmeep-generic-example-auxiliary-backend-guards:"
V103_IDENTITY_PREFIX = "gpmeep-frozen-v103-process-identity:"
GUARDED_ENTRYPOINTS = ("run", "run_k_point", "solve_cw", "solve_eigfreq")
MAX_RESULT_VALUES = 1_000_000
_FROZEN_V103_RUNTIME_TOKEN = object()


def backend_recommendation_mismatch_type(mp: Any) -> type[BaseException]:
    """Resolve the guard exception from the real installed submodule.

    ``backend_policy.py`` is installed as a Meep submodule, but the generated
    SWIG package does not promise a ``meep.backend_policy`` root attribute
    before that submodule has been imported.  Performance validation must use
    the same lazy installed-package layout as users, rather than depending on
    an attribute accidentally injected by a test double or earlier import.
    """

    module_name = f"{mp.__name__}.backend_policy"
    try:
        policy = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Retain compatibility with deliberately minimal test doubles which
        # expose the policy object directly but are not importable packages.
        if exc.name != module_name:
            raise
        policy = getattr(mp, "backend_policy", None)
    exception_type = getattr(policy, "BackendRecommendationMismatch", None)
    if (
        not isinstance(exception_type, type)
        or not issubclass(exception_type, BaseException)
    ):
        raise RuntimeError(
            "gpmeep backend policy exception type is unavailable"
        )
    return exception_type


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"generic example oracle produced non-numeric {name}"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError(f"generic example oracle produced non-finite {name}")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def positive_range(value: str) -> tuple[int, int]:
    raw_minimum, separator, raw_maximum = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("must use MIN:MAX")
    try:
        minimum = positive_int(raw_minimum)
        maximum = positive_int(raw_maximum)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "range endpoints must be positive integers"
        ) from exc
    if maximum < minimum:
        raise argparse.ArgumentTypeError("range maximum must be >= minimum")
    return minimum, maximum


def positive_finite(value: str) -> float:
    try:
        result = finite_float(value, "signal threshold")
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def result_name(value: str) -> str:
    if not value.isidentifier() or value.startswith("_"):
        raise argparse.ArgumentTypeError(
            "result name must be a public Python identifier"
        )
    return value


def result_threshold(value: str) -> tuple[str, float]:
    name, separator, raw_threshold = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must use NAME=POSITIVE_THRESHOLD")
    return result_name(name), positive_finite(raw_threshold)


def acknowledgement_reason(value: str) -> str:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("acknowledgement reason must be text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise argparse.ArgumentTypeError(
            "acknowledgement reason must be valid UTF-8 text"
        ) from exc
    if not 8 <= len(encoded) <= 512:
        raise argparse.ArgumentTypeError(
            "acknowledgement reason must occupy 8 to 512 UTF-8 bytes"
        )
    return value


def result_shape_contract(value: str) -> tuple[str, tuple[int, ...]]:
    name, separator, raw_shape = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("must use NAME=scalar or NAME=DIM[,DIM...]")
    name = result_name(name)
    if raw_shape == "scalar":
        return name, ()
    if not raw_shape:
        raise argparse.ArgumentTypeError("result shape must not be empty")
    dimensions: list[int] = []
    for raw_dimension in raw_shape.split(","):
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "result dimensions must be positive integers"
            ) from exc
        if dimension <= 0:
            raise argparse.ArgumentTypeError(
                "result dimensions must be positive integers"
            )
        dimensions.append(dimension)
    if math.prod(dimensions) > MAX_RESULT_VALUES:
        raise argparse.ArgumentTypeError(
            f"result shape exceeds the {MAX_RESULT_VALUES}-value limit"
        )
    return name, tuple(dimensions)


def numeric_result_vector(value: Any, name: str) -> dict[str, Any]:
    """Convert a rectangular scalar/vector result into stable JSON metrics."""

    if hasattr(value, "tolist"):
        shape = getattr(value, "shape", None)
        size = getattr(value, "size", None)
        if shape is not None and size is not None:
            try:
                dimensions = tuple(operator.index(item) for item in shape)
                item_count = operator.index(size)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"example result {name!r} has an invalid array shape or size"
                ) from exc
            if isinstance(size, bool) or any(
                isinstance(item, bool) for item in shape
            ):
                raise RuntimeError(
                    f"example result {name!r} has an invalid array shape or size"
                )
            if item_count < 0 or any(dimension < 0 for dimension in dimensions):
                raise RuntimeError(
                    f"example result {name!r} has an invalid array shape or size"
                )
            if item_count > MAX_RESULT_VALUES:
                raise RuntimeError(
                    f"example result {name!r} exceeds the "
                    f"{MAX_RESULT_VALUES}-value limit"
                )
            expected_count = math.prod(dimensions)
            if expected_count != item_count:
                raise RuntimeError(
                    f"example result {name!r} has inconsistent array shape and size"
                )
            if item_count == 0:
                raise RuntimeError(f"example result {name!r} is empty")
        try:
            value = value.tolist()
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"example result {name!r} could not be converted to a list"
            ) from exc

    flat: list[complex] = []

    def visit(item: Any, path: str) -> tuple[int, ...]:
        if hasattr(item, "item") and not isinstance(
            item, (str, bytes, list, tuple)
        ):
            try:
                item = item.item()
            except (TypeError, ValueError):
                pass
        if isinstance(item, (list, tuple)):
            if not item:
                raise RuntimeError(f"example result {name!r} is empty at {path}")
            child_shapes = [
                visit(child, f"{path}[{index}]")
                for index, child in enumerate(item)
            ]
            if any(shape != child_shapes[0] for shape in child_shapes[1:]):
                raise RuntimeError(
                    f"example result {name!r} is ragged at {path}"
                )
            return (len(item), *child_shapes[0])
        if isinstance(item, (str, bytes, bool)):
            raise RuntimeError(
                f"example result {name!r} contains a non-numeric value at {path}"
            )
        try:
            number = complex(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"example result {name!r} contains a non-numeric value at {path}"
            ) from exc
        if not math.isfinite(number.real) or not math.isfinite(number.imag):
            raise RuntimeError(
                f"example result {name!r} contains a non-finite value at {path}"
            )
        flat.append(number)
        if len(flat) > MAX_RESULT_VALUES:
            raise RuntimeError(
                f"example result {name!r} exceeds the "
                f"{MAX_RESULT_VALUES}-value limit"
            )
        return ()

    shape = visit(value, name)
    if not flat:
        raise RuntimeError(f"example result {name!r} contains no scalar values")
    real = [number.real for number in flat]
    imag = [number.imag for number in flat]
    magnitudes = [abs(number) for number in flat]
    result = {
        "count": len(flat),
        "real": real,
        "imag": imag,
        "l2": math.sqrt(math.fsum(value * value for value in magnitudes)),
        "max_abs": max(magnitudes),
        "weighted_real": math.fsum(
            (index + 1) * value for index, value in enumerate(real)
        ),
        "weighted_imag": math.fsum(
            (index + 1) * value for index, value in enumerate(imag)
        ),
    }
    if shape:
        result["shape"] = list(shape)
    else:
        result["rank"] = 0
    return result


def collect_example_results(
    namespace: dict[str, Any] | None, args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    if not args.result_vector:
        return {}
    if namespace is None:
        raise RuntimeError(
            "example exited through SystemExit before result vectors could be captured"
        )
    results: dict[str, dict[str, Any]] = {}
    minimum_l2 = dict(getattr(args, "min_result_l2", []))
    maximum_abs = dict(getattr(args, "max_result_abs", []))
    expected_shapes = dict(getattr(args, "expected_result_shape", []))
    for name in args.result_vector:
        if name not in namespace:
            raise RuntimeError(f"example did not define requested result {name!r}")
        result = numeric_result_vector(namespace[name], name)
        actual_shape = tuple(result.get("shape", ()))
        expected_shape = expected_shapes.get(name)
        if expected_shape is None:
            raise RuntimeError(
                f"example result {name!r} has no independent shape contract"
            )
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"example result {name!r} has shape {actual_shape}, expected "
                f"{expected_shape}"
            )
        threshold = minimum_l2.get(name)
        if threshold is not None and result["l2"] < threshold:
            raise RuntimeError(
                f"example result {name!r} L2 norm {result['l2']} is below "
                f"required signal {threshold}"
            )
        threshold = maximum_abs.get(name)
        if threshold is not None and result["max_abs"] > threshold:
            raise RuntimeError(
                f"example result {name!r} maximum absolute value "
                f"{result['max_abs']} exceeds required limit {threshold}"
            )
        results[name] = result
    return results


def phase_call_totals(statistics: Any) -> dict[str, int]:
    """Return backend phase-call totals without counting transfer/readback data."""

    totals = {"cpu": 0, "cuda": 0}

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key, value in item.items():
            if isinstance(value, dict):
                visit(value)
            elif key.endswith("_calls"):
                for backend in totals:
                    if key.startswith(f"{backend}_"):
                        count = int(value)
                        if count < 0:
                            raise RuntimeError(
                                f"negative {backend} phase-call counter {key}"
                            )
                        totals[backend] += count

    visit(statistics)
    return totals


def counter_delta(
    before: dict[str, int], after: dict[str, int], label: str
) -> dict[str, int]:
    result = {backend: after[backend] - before[backend] for backend in before}
    for backend, value in result.items():
        if value < 0:
            raise RuntimeError(
                f"GPU statistics reset during {label}: negative {backend} delta"
            )
    return result


def statistics_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, int]]:
    """Subtract every integral gpmeep counter without hiding resets."""

    result: dict[str, dict[str, int]] = {}
    for group_name in sorted(set(before) | set(after)):
        before_group = before.get(group_name)
        after_group = after.get(group_name)
        if before_group is None:
            before_group = {}
        if after_group is None:
            after_group = {}
        if not isinstance(before_group, dict) or not isinstance(after_group, dict):
            if isinstance(before_group, dict) == isinstance(after_group, dict):
                continue
            raise RuntimeError(f"GPU statistics group changed type: {group_name}")
        group_delta: dict[str, int] = {}
        for counter_name in sorted(set(before_group) | set(after_group)):
            before_value = before_group.get(counter_name)
            after_value = after_group.get(counter_name)
            before_integral = not isinstance(before_value, bool) and isinstance(
                before_value, int
            )
            after_integral = not isinstance(after_value, bool) and isinstance(
                after_value, int
            )
            if not before_integral and not after_integral:
                continue
            if before_integral and not after_integral:
                raise RuntimeError(
                    "GPU statistics counter disappeared or changed type: "
                    f"{group_name}.{counter_name}"
                )
            if after_integral and counter_name in before_group and not before_integral:
                raise RuntimeError(
                    "GPU statistics counter changed type: "
                    f"{group_name}.{counter_name}"
                )
            if not before_integral:
                before_value = 0
            assert isinstance(before_value, int) and isinstance(after_value, int)
            delta = after_value - before_value
            if delta < 0 and not counter_name.startswith("live_"):
                raise RuntimeError(
                    "GPU statistics reset during Simulation.run: "
                    f"{group_name}.{counter_name}"
                )
            group_delta[counter_name] = delta
        result[group_name] = group_delta
    return result


def field_timing_snapshot(mp: Any, simulation: Any) -> dict[str, list[float]] | None:
    if simulation.fields is None or not hasattr(simulation, "get_timing_data"):
        return None
    sink_names = (
        ("connecting_chunks", "Connecting"),
        ("time_stepping", "Stepping"),
        ("boundaries", "Boundaries"),
        ("mpi_all_to_all", "MpiAllTime"),
        ("mpi_one_to_one", "MpiOneTime"),
        ("field_output", "FieldOutput"),
        ("fourier_transform", "FourierTransforming"),
        ("mpb", "MPBTime"),
        ("get_farfields", "GetFarfieldsTime"),
        ("other", "Other"),
        ("field_update_b", "FieldUpdateB"),
        ("field_update_h", "FieldUpdateH"),
        ("field_update_d", "FieldUpdateD"),
        ("field_update_e", "FieldUpdateE"),
        ("boundary_stepping_b", "BoundarySteppingB"),
        ("boundary_stepping_wh", "BoundarySteppingWH"),
        ("boundary_stepping_ph", "BoundarySteppingPH"),
        ("boundary_stepping_h", "BoundarySteppingH"),
        ("boundary_stepping_d", "BoundarySteppingD"),
        ("boundary_stepping_we", "BoundarySteppingWE"),
        ("boundary_stepping_pe", "BoundarySteppingPE"),
        ("boundary_stepping_e", "BoundarySteppingE"),
    )
    timing_data = simulation.get_timing_data()
    snapshot: dict[str, list[float]] = {}
    for name, attribute in sink_names:
        sink = getattr(mp, attribute, None)
        if sink is None or sink not in timing_data:
            raise RuntimeError(f"Meep timing sink is unavailable: {attribute}")
        values = [finite_float(item, f"{name} timing") for item in timing_data[sink]]
        if not values or any(value < 0.0 for value in values):
            raise RuntimeError(f"Meep timing sink is empty or negative: {attribute}")
        snapshot[name] = values
    return snapshot


def field_timing_delta(
    before: dict[str, list[float]] | None,
    after: dict[str, list[float]] | None,
) -> dict[str, dict[str, Any]] | None:
    if after is None:
        return None
    if before is not None and set(before) != set(after):
        raise RuntimeError("Meep timing sink inventory changed during Simulation.run")
    result: dict[str, dict[str, Any]] = {}
    for name, after_values in after.items():
        before_values = [0.0] * len(after_values) if before is None else before[name]
        if len(before_values) != len(after_values):
            raise RuntimeError("Meep timing rank count changed during Simulation.run")
        values = [
            after_value - before_value
            for before_value, after_value in zip(before_values, after_values)
        ]
        if any(value < -1e-12 for value in values):
            raise RuntimeError(f"Meep timing reset during Simulation.run: {name}")
        values = [max(0.0, value) for value in values]
        result[name] = {
            "per_rank_seconds": values,
            "maximum_seconds": max(values),
            "mean_seconds": math.fsum(values) / len(values),
        }
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("example completed without calling Simulation.run")

    def scalar_summary(key: str) -> dict[str, float]:
        values = [float(record[key]) for record in records]
        return {
            "final": values[-1],
            "sum": math.fsum(values),
            "l2": math.sqrt(math.fsum(value * value for value in values)),
            "weighted_checksum": math.fsum(
                (index + 1) * value for index, value in enumerate(values)
            ),
        }

    timesteps = [int(record["timestep"]) for record in records]
    timestep_deltas = [int(record["timestep_delta"]) for record in records]
    return {
        "run_count": len(records),
        "timestep": {
            "final": timesteps[-1],
            "sum": sum(timesteps),
            "weighted_checksum": sum(
                (index + 1) * value
                for index, value in enumerate(timesteps)
            ),
        },
        "timestep_delta": {
            "final": timestep_deltas[-1],
            "sum": sum(timestep_deltas),
            "weighted_checksum": sum(
                (index + 1) * value
                for index, value in enumerate(timestep_deltas)
            ),
        },
        "meep_time": scalar_summary("meep_time"),
        "field_energy": scalar_summary("field_energy"),
        "dft_norm": scalar_summary("dft_norm"),
        "per_run": {
            "timestep": timesteps,
            "timestep_delta": timestep_deltas,
            "meep_time": [float(record["meep_time"]) for record in records],
            "field_energy": [
                float(record["field_energy"]) for record in records
            ],
            "dft_norm": [float(record["dft_norm"]) for record in records],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--observe-operation",
        choices=("run", "run_k_point"),
        default="run",
        help=(
            "synchronous Simulation operation to observe; run_k_point treats "
            "each operation's intentional field restart as a zero timestep "
            "accounting baseline"
        ),
    )
    parser.add_argument("--expected-run-count", required=True, type=positive_int)
    parser.add_argument("--expected-final-timestep", type=nonnegative_int)
    parser.add_argument("--expected-total-timestep-delta", type=positive_int)
    parser.add_argument(
        "--expected-run-timestep-range",
        action="append",
        default=[],
        type=positive_range,
    )
    parser.add_argument(
        "--expected-run-timestep-delta-range",
        action="append",
        default=[],
        type=positive_range,
    )
    parser.add_argument("--min-field-energy", type=positive_finite)
    parser.add_argument("--min-dft-norm", type=positive_finite)
    parser.add_argument("--min-each-field-energy", type=positive_finite)
    parser.add_argument("--min-each-dft-norm", type=positive_finite)
    parser.add_argument(
        "--result-vector", action="append", default=[], type=result_name
    )
    parser.add_argument(
        "--expected-result-shape",
        action="append",
        default=[],
        type=result_shape_contract,
    )
    parser.add_argument(
        "--min-result-l2", action="append", default=[], type=result_threshold
    )
    parser.add_argument(
        "--max-result-abs", action="append", default=[], type=result_threshold
    )
    parser.add_argument(
        "--acknowledge-backend-mismatch-reason",
        type=acknowledgement_reason,
        help=(
            "explicitly acknowledge each exact nonrecommended CPU/CUDA run "
            "after the guard rejects its first mutation-free attempt"
        ),
    )
    parser.add_argument(
        "--acknowledgement-scope",
        choices=("simulation", "process"),
        default="simulation",
        help=(
            "reuse scope for an explicitly acknowledged mismatch; process "
            "scope is an opt-in for a reviewed equivalent simulation batch"
        ),
    )
    parser.add_argument(
        "--disable-field-timing-snapshots",
        action="store_true",
        help=(
            "do not call Simulation.get_timing_data before or after each run; "
            "use for numerical trajectories where the added MPI collectives "
            "are themselves an intrusive observer"
        ),
    )
    parser.add_argument(
        "--upstream-cpu-baseline",
        action="store_true",
        help=(
            "run the unchanged physical oracle against a separately built "
            "upstream Meep CPU package which intentionally has no gpmeep.gpu "
            "instrumentation; this mode is for M3 comparison evidence only"
        ),
    )
    parser.add_argument(
        "--unittest-discover",
        action="store_true",
        help=(
            "load the selected Python file through verbose unittest discovery "
            "while retaining Simulation instrumentation and canonical test "
            "identities"
        ),
    )
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("example_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if (
        args.upstream_cpu_baseline
        and args.acknowledge_backend_mismatch_reason is not None
    ):
        parser.error(
            "--upstream-cpu-baseline cannot acknowledge a gpmeep backend mismatch"
        )
    if args.example_args[:1] == ["--"]:
        args.example_args = args.example_args[1:]
    args.example = args.example.resolve()
    if not args.example.is_file() or args.example.suffix != ".py":
        parser.error(f"example is not a Python file: {args.example}")
    if args.unittest_discover and args.example_args:
        parser.error("--unittest-discover does not accept example arguments")
    if all(
        value is None
        for value in (
            args.expected_final_timestep,
            args.expected_total_timestep_delta,
            args.expected_run_timestep_range or None,
            args.expected_run_timestep_delta_range or None,
            args.min_field_energy,
            args.min_dft_norm,
            args.min_each_field_energy,
            args.min_each_dft_norm,
        )
    ):
        parser.error(
            "at least one timestep or physical-signal contract is required"
        )
    for option_name, ranges in (
        ("--expected-run-timestep-range", args.expected_run_timestep_range),
        (
            "--expected-run-timestep-delta-range",
            args.expected_run_timestep_delta_range,
        ),
    ):
        if ranges and len(ranges) != args.expected_run_count:
            parser.error(
                f"{option_name} must appear exactly --expected-run-count times"
            )
    if args.expected_final_timestep is not None and args.expected_run_timestep_range:
        parser.error(
            "--expected-final-timestep and --expected-run-timestep-range are "
            "mutually exclusive"
        )
    if (
        args.expected_total_timestep_delta is not None
        and args.expected_run_timestep_delta_range
    ):
        parser.error(
            "--expected-total-timestep-delta and "
            "--expected-run-timestep-delta-range are mutually exclusive"
        )
    if len(set(args.result_vector)) != len(args.result_vector):
        parser.error("--result-vector names must be unique")
    shape_names = [name for name, _ in args.expected_result_shape]
    if len(set(shape_names)) != len(shape_names):
        parser.error("--expected-result-shape names must be unique")
    missing_shapes = sorted(set(args.result_vector) - set(shape_names))
    unknown_shapes = sorted(set(shape_names) - set(args.result_vector))
    if missing_shapes or unknown_shapes:
        details = []
        if missing_shapes:
            details.append("missing for: " + ", ".join(missing_shapes))
        if unknown_shapes:
            details.append("unknown for: " + ", ".join(unknown_shapes))
        parser.error(
            "--expected-result-shape must match every --result-vector exactly ("
            + "; ".join(details)
            + ")"
        )
    threshold_names = [name for name, _ in args.min_result_l2]
    if len(set(threshold_names)) != len(threshold_names):
        parser.error("--min-result-l2 names must be unique")
    unknown_thresholds = sorted(set(threshold_names) - set(args.result_vector))
    if unknown_thresholds:
        parser.error(
            "--min-result-l2 requires a matching --result-vector for: "
            + ", ".join(unknown_thresholds)
        )
    maximum_names = [name for name, _ in args.max_result_abs]
    if len(set(maximum_names)) != len(maximum_names):
        parser.error("--max-result-abs names must be unique")
    unknown_maximums = sorted(set(maximum_names) - set(args.result_vector))
    if unknown_maximums:
        parser.error(
            "--max-result-abs requires a matching --result-vector for: "
            + ", ".join(unknown_maximums)
        )
    if args.unittest_discover and args.result_vector:
        parser.error("--unittest-discover does not expose result vectors")
    return args


def run_verbose_unittest_discovery(example: pathlib.Path) -> None:
    """Run one exact file as a canonical, verbose unittest discovery suite."""
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(example.parent),
        pattern=example.name,
        top_level_dir=str(example.parent),
    )
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    if result.testsRun <= 0:
        raise RuntimeError("unittest discovery completed without running tests")
    if not result.wasSuccessful():
        raise RuntimeError("unittest discovery reported failures or errors")


def validate_contract(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not records:
        raise RuntimeError("example completed without calling Simulation.run")
    if len(records) != args.expected_run_count:
        raise RuntimeError(
            f"expected {args.expected_run_count} Simulation.run calls, got "
            f"{len(records)}"
        )
    timestep_ranges = getattr(args, "expected_run_timestep_range", [])
    timestep_delta_ranges = getattr(
        args, "expected_run_timestep_delta_range", []
    )
    for label, key, ranges in (
        ("timestep", "timestep", timestep_ranges),
        ("timestep delta", "timestep_delta", timestep_delta_ranges),
    ):
        for index, (record, bounds) in enumerate(zip(records, ranges)):
            value = int(record[key])
            minimum, maximum = bounds
            if not minimum <= value <= maximum:
                raise RuntimeError(
                    f"run {index} {label} {value} is outside required range "
                    f"[{minimum}, {maximum}]"
                )
    timestep_deltas = [int(record["timestep_delta"]) for record in records]
    zero_work = [index for index, value in enumerate(timestep_deltas) if value <= 0]
    if zero_work:
        raise RuntimeError(
            "Simulation.run advanced zero timesteps at record indices "
            f"{zero_work}"
        )
    total_delta = sum(timestep_deltas)
    if (
        args.expected_total_timestep_delta is not None
        and total_delta != args.expected_total_timestep_delta
    ):
        raise RuntimeError(
            f"expected total timestep delta {args.expected_total_timestep_delta}, "
            f"got {total_delta}"
        )
    final_timestep = int(records[-1]["timestep"])
    if (
        args.expected_final_timestep is not None
        and final_timestep != args.expected_final_timestep
    ):
        raise RuntimeError(
            f"expected final timestep {args.expected_final_timestep}, got "
            f"{final_timestep}"
        )
    if args.min_field_energy is not None:
        maximum = max(abs(float(record["field_energy"])) for record in records)
        if maximum < args.min_field_energy:
            raise RuntimeError(
                f"maximum field energy {maximum} is below required signal "
                f"{args.min_field_energy}"
            )
    if args.min_dft_norm is not None:
        maximum = max(abs(float(record["dft_norm"])) for record in records)
        if maximum < args.min_dft_norm:
            raise RuntimeError(
                f"maximum DFT norm {maximum} is below required signal "
                f"{args.min_dft_norm}"
            )
    for key, threshold in (
        ("field_energy", args.min_each_field_energy),
        ("dft_norm", args.min_each_dft_norm),
    ):
        if threshold is None:
            continue
        values = [abs(float(record[key])) for record in records]
        below = [index for index, value in enumerate(values) if value < threshold]
        if below:
            raise RuntimeError(
                f"{key} is below required per-run signal {threshold} at "
                f"record indices {below}"
            )


def invoke_original_run_with_pristine_class(
    mp: Any,
    method_name: str,
    instrumented_methods: dict[str, tuple[Any, Any]],
    simulation: Any,
    step_funcs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Invoke Meep while its policy observes the unmodified class definition.

    The oracle intercepts the selected Simulation method and every guarded
    auxiliary entrypoint at the class boundary. gpmeep's recommendation guard
    intentionally binds the exact Simulation class and every package method
    reachable from it. Leaving even one observer installed during that binding
    makes the validation harness look like a package mutation. Restore the
    complete instrumented method set for the synchronous invocation, then
    reinstall every observer before collecting measurements.

    Any third-party mutation while the original method is active fails closed;
    the oracle must never silently overwrite such a mutation.
    """

    if method_name not in instrumented_methods or not instrumented_methods:
        raise RuntimeError("Simulation instrumentation inventory is incomplete")
    instrumentation = tuple(
        (name, original, observer)
        for name, (original, observer) in instrumented_methods.items()
    )
    for name, _original, observer in instrumentation:
        if getattr(mp.Simulation, name) is not observer:
            raise RuntimeError(
                f"Simulation.{name} instrumentation changed before original dispatch"
            )
    for name, original, _observer in instrumentation:
        setattr(mp.Simulation, name, original)
    original_run = instrumented_methods[method_name][0]
    try:
        result = original_run(simulation, *step_funcs, **kwargs)
    except BaseException as error:
        changed = [
            name
            for name, original, _observer in instrumentation
            if getattr(mp.Simulation, name) is not original
        ]
        for name, _original, observer in instrumentation:
            setattr(mp.Simulation, name, observer)
        if changed:
            raise RuntimeError(
                "Simulation package instrumentation changed while the original "
                f"implementation was active: {', '.join(changed)}"
            ) from error
        raise
    changed = [
        name
        for name, original, _observer in instrumentation
        if getattr(mp.Simulation, name) is not original
    ]
    for name, _original, observer in instrumentation:
        setattr(mp.Simulation, name, observer)
    if changed:
        raise RuntimeError(
            "Simulation package instrumentation changed while the original "
            f"implementation was active: {', '.join(changed)}"
        )
    return result


def main(
    argv: list[str] | None = None,
    *,
    _frozen_v103_runtime_token: object | None = None,
) -> int:
    args = parse_args(argv)
    frozen_v103_runtime = _frozen_v103_runtime_token is _FROZEN_V103_RUNTIME_TOKEN
    if _frozen_v103_runtime_token is not None and not frozen_v103_runtime:
        raise RuntimeError("invalid frozen-v1.0.3 oracle capability token")

    import meep as mp

    frozen_v103_identity = None
    if frozen_v103_runtime:
        if args.upstream_cpu_baseline:
            raise RuntimeError("frozen v1.0.3 and upstream baseline modes conflict")
        if args.acknowledge_backend_mismatch_reason is not None:
            raise RuntimeError("frozen v1.0.3 cannot use the v1.1 acknowledgement API")
        import v103_runtime_identity

        frozen_v103_identity = v103_runtime_identity.verify_current_process(mp)

    if args.upstream_cpu_baseline and hasattr(mp, "gpu"):
        raise RuntimeError(
            "upstream CPU baseline mode refuses a package exposing gpmeep.gpu"
        )
    if not args.upstream_cpu_baseline and not hasattr(mp, "gpu"):
        raise RuntimeError(
            "gpmeep oracle mode requires the gpmeep.gpu instrumentation API"
        )

    is_master = bool(mp.am_master()) if hasattr(mp, "am_master") else True

    records: list[dict[str, Any]] = []
    auxiliary_guard_events: list[dict[str, Any]] = []
    observed_method = args.observe_operation
    operation_label = f"Simulation.{observed_method}"
    original_run = getattr(mp.Simulation, observed_method)
    instrumented_methods: dict[str, tuple[Any, Any]] = {}

    def audited_run(simulation: Any, *step_funcs: Any, **kwargs: Any) -> Any:
        pre_fields = simulation.fields
        pre_state_timestep = (
            int(simulation.fields.t) if simulation.fields is not None else 0
        )
        # run_k_point intentionally restarts the fields at zero for every k.
        # Keep the real pre-state for the fail-closed acknowledgement check,
        # while accounting each completed k-point run from its zero baseline.
        pre_timestep = (
            0 if observed_method == "run_k_point" else pre_state_timestep
        )
        pre_statistics = (
            {} if args.upstream_cpu_baseline else mp.gpu.statistics()
        )
        pre_calls = phase_call_totals(pre_statistics)
        pre_field_timing = (
            None
            if args.disable_field_timing_snapshots
            else field_timing_snapshot(mp, simulation)
        )
        if args.upstream_cpu_baseline or frozen_v103_runtime:
            started_ns = time.monotonic_ns()
            result = invoke_original_run_with_pristine_class(
                mp,
                observed_method,
                instrumented_methods,
                simulation,
                step_funcs,
                kwargs,
            )
            ended_ns = time.monotonic_ns()
            run_wall_seconds = (ended_ns - started_ns) / 1_000_000_000.0
            explicit_backend_guard = None
            legacy_backend_guard = {
                "attempts": 1,
                "acknowledgements": 0,
                "warning_wall_seconds": 0.0,
                "warning_telemetry": None,
                "final_telemetry": None,
            }
        else:
            result, explicit_backend_guard = (
                qualification_guard.invoke_with_acknowledgement(
                    mp,
                    simulation,
                    lambda: invoke_original_run_with_pristine_class(
                        mp,
                        observed_method,
                        instrumented_methods,
                        simulation,
                        step_funcs,
                        kwargs,
                    ),
                    args.acknowledge_backend_mismatch_reason,
                    args.acknowledgement_scope,
                )
            )
            run_wall_seconds = explicit_backend_guard[
                "execution_wall_seconds"
            ]
            legacy_backend_guard = {
                "attempts": explicit_backend_guard["attempts"],
                "acknowledgements": explicit_backend_guard[
                    "acknowledgements"
                ],
                "warning_wall_seconds": explicit_backend_guard[
                    "warning_wall_seconds"
                ],
                "warning_telemetry": explicit_backend_guard[
                    "warning_telemetry"
                ],
                "final_telemetry": explicit_backend_guard[
                    "final_telemetry"
                ],
            }
        if not math.isfinite(run_wall_seconds) or run_wall_seconds <= 0:
            raise RuntimeError(
                f"{operation_label} wall time is not positive and finite"
            )
        post_timestep = int(simulation.timestep())
        post_statistics = (
            {} if args.upstream_cpu_baseline else mp.gpu.statistics()
        )
        post_calls = phase_call_totals(post_statistics)
        post_field_timing = (
            None
            if args.disable_field_timing_snapshots
            else field_timing_snapshot(mp, simulation)
        )
        run_calls = counter_delta(pre_calls, post_calls, operation_label)
        if args.upstream_cpu_baseline:
            active_backend = "cpu"
            other_backend = "cuda"
        else:
            active_backend = str(mp.gpu.active_backend)
            if active_backend not in ("cpu", "cuda"):
                raise RuntimeError(
                    f"{operation_label} selected unsupported backend "
                    f"{active_backend!r}"
                )
            other_backend = "cuda" if active_backend == "cpu" else "cpu"
            if post_timestep > pre_timestep and run_calls[active_backend] <= 0:
                raise RuntimeError(
                    f"{operation_label} advanced timesteps without selected-backend "
                    f"phase dispatch ({active_backend})"
                )
            if run_calls[other_backend] != 0:
                raise RuntimeError(
                    f"{operation_label} used backend fallback: "
                    f"{other_backend} calls={run_calls[other_backend]}"
                )
        record = {
            "timestep": post_timestep,
            "timestep_delta": post_timestep - pre_timestep,
            "meep_time": finite_float(simulation.round_time(), "Meep time"),
            "run_wall_seconds": run_wall_seconds,
            "field_energy": finite_float(
                simulation.field_energy_in_box(
                    box=simulation.fields.total_volume()
                ),
                "field energy",
            ),
            "dft_norm": finite_float(
                simulation.fields.dft_norm(), "DFT norm"
            ),
            "active_backend": active_backend,
            "requested_backend": str(
                active_backend
                if args.upstream_cpu_baseline
                else getattr(mp.gpu, "requested_backend", active_backend)
            ),
            "execution_diagnostic": str(
                "upstream-meep-cpu-baseline"
                if args.upstream_cpu_baseline
                else (
                    simulation.fields.gpu_execution_diagnostic()
                    if hasattr(simulation.fields, "gpu_execution_diagnostic")
                    else "unavailable"
                )
            ),
            "runtime_counters": {
                str(name): int(value)
                for name, value in post_statistics.get("runtime", {}).items()
            },
            "run_phase_calls": run_calls,
            "statistics_delta": statistics_delta(pre_statistics, post_statistics),
            "field_timing_delta": field_timing_delta(
                pre_field_timing, post_field_timing
            ),
            "backend_guard": legacy_backend_guard,
            "explicit_backend_guard": explicit_backend_guard,
        }
        measurement_calls = counter_delta(
            post_calls,
            phase_call_totals(
                {} if args.upstream_cpu_baseline else mp.gpu.statistics()
            ),
            "oracle measurement",
        )
        record["measurement_phase_calls"] = measurement_calls
        if record["timestep"] < 0 or record["timestep_delta"] < 0:
            raise RuntimeError(
                "generic example oracle produced a negative timestep or delta"
            )
        records.append(record)
        if is_master:
            print(
                CHECKPOINT_PREFIX
                + json.dumps(
                    {
                        "schema": "gpmeep-generic-example-run-checkpoint-v1",
                        "run_index": len(records) - 1,
                        "record": record,
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return result

    auxiliary_observers: dict[str, tuple[Any, Any]] = {}
    instrumented_methods[observed_method] = (original_run, audited_run)
    if not args.upstream_cpu_baseline and not frozen_v103_runtime:
        for auxiliary_name in GUARDED_ENTRYPOINTS:
            if auxiliary_name == observed_method:
                continue
            auxiliary_original = getattr(mp.Simulation, auxiliary_name, None)
            if not callable(auxiliary_original):
                raise RuntimeError(
                    f"Simulation.{auxiliary_name} guarded entrypoint is unavailable"
                )

            def make_auxiliary_observer(method_name: str) -> Any:
                def auxiliary_observer(
                    simulation: Any, *method_args: Any, **method_kwargs: Any
                ) -> Any:
                    result, evidence = qualification_guard.invoke_with_acknowledgement(
                        mp,
                        simulation,
                        lambda: invoke_original_run_with_pristine_class(
                            mp,
                            method_name,
                            instrumented_methods,
                            simulation,
                            method_args,
                            method_kwargs,
                        ),
                        args.acknowledge_backend_mismatch_reason,
                        args.acknowledgement_scope,
                    )
                    auxiliary_guard_events.append(
                        {"entrypoint": method_name, "guard": evidence}
                    )
                    return result

                return auxiliary_observer

            auxiliary_observer = make_auxiliary_observer(auxiliary_name)
            auxiliary_observers[auxiliary_name] = (
                auxiliary_original,
                auxiliary_observer,
            )
            instrumented_methods[auxiliary_name] = (
                auxiliary_original,
                auxiliary_observer,
            )
            setattr(mp.Simulation, auxiliary_name, auxiliary_observer)

    previous_argv = sys.argv
    previous_argv_contents = list(previous_argv)
    previous_path = list(sys.path)
    setattr(mp.Simulation, observed_method, audited_run)
    # Preserve the interpreter-owned argv list identity. Trusted Meep methods
    # reference sys.argv, so replacing the list object after package import
    # correctly looks like a dependency mutation to the backend policy.
    previous_argv[:] = [str(args.example), *args.example_args]
    sys.path.insert(0, str(args.example.parent))
    namespace: dict[str, Any] | None = None
    try:
        if args.unittest_discover:
            run_verbose_unittest_discovery(args.example)
        else:
            try:
                namespace = runpy.run_path(str(args.example), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        instrumentation_errors = []
        if getattr(mp.Simulation, observed_method) is not audited_run:
            instrumentation_errors.append(operation_label)
        setattr(mp.Simulation, observed_method, original_run)
        for auxiliary_name, (
            auxiliary_original,
            auxiliary_observer,
        ) in auxiliary_observers.items():
            if getattr(mp.Simulation, auxiliary_name) is not auxiliary_observer:
                instrumentation_errors.append(f"Simulation.{auxiliary_name}")
            setattr(mp.Simulation, auxiliary_name, auxiliary_original)
        # An example may rebind sys.argv as well as mutate its contents. Put
        # back the exact interpreter-owned list object first, then restore the
        # values it held before the oracle invocation.
        sys.argv = previous_argv
        previous_argv[:] = previous_argv_contents
        sys.path[:] = previous_path
        if instrumentation_errors and not active_exception:
            raise RuntimeError(
                "generic oracle instrumentation changed before restoration: "
                + ", ".join(instrumentation_errors)
            )
    validate_contract(records, args)
    example_results = collect_example_results(namespace, args)
    if not is_master:
        return 0
    if frozen_v103_identity is not None:
        print(
            V103_IDENTITY_PREFIX
            + json.dumps(
                frozen_v103_identity,
                sort_keys=True,
                allow_nan=False,
            )
        )
    print(
        AUXILIARY_GUARD_PREFIX
        + json.dumps(
            {"events": auxiliary_guard_events},
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(AUDIT_PREFIX + json.dumps({"runs": records}, sort_keys=True, allow_nan=False))
    metrics = aggregate(records)
    if example_results:
        metrics["example_results"] = example_results
    print(METRIC_PREFIX + json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
