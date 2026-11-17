"""Explicit CPU/CUDA recommendation guard for gpmeep v1.1.0.

The policy advises and guards; it never selects or changes the backend.
"""

from __future__ import annotations

import ast
import abc
import builtins
import dis
import collections
import collections.abc
import contextvars
import enum
import functools
import hashlib
import inspect
import json
import math
import numbers
import os
import re
import secrets
import struct
import sys
import threading
import textwrap
import types
import typing
import unittest
import weakref
from pathlib import Path
from typing import Any


_NAMEDTUPLE_AUTHENTICATION_TEMPLATE = collections.namedtuple(
    "_GpmeepNamedtupleAuthenticationTemplate",
    ("first", "second"),
    defaults=(None,),
)
_ORIGINAL_COLLECTIONS_NAMEDTUPLE = collections.namedtuple
_ORIGINAL_COLLECTIONS_NAMEDTUPLE_GLOBALS = collections.namedtuple.__globals__
_ORIGINAL_BUILTINS_PRINT = builtins.print
_ORIGINAL_NAMEDTUPLE_FORMAT_CODES = types.MappingProxyType(
    {
        "_make": (
            _NAMEDTUPLE_AUTHENTICATION_TEMPLATE.__dict__["_make"]
            .__func__
            .__code__
        ),
        "_replace": _NAMEDTUPLE_AUTHENTICATION_TEMPLATE.__dict__[
            "_replace"
        ].__code__,
    }
)
del _NAMEDTUPLE_AUTHENTICATION_TEMPLATE


_ORIGINAL_CONCRETE_PATH_TYPE = type(Path())
_ORIGINAL_RE_NAMED_INT_CONSTANT_TYPE = type(re._constants.LITERAL)
_ORIGINAL_ABC_META = abc.ABCMeta
_ORIGINAL_ABC_GET_DUMP = abc._get_dump
_ORIGINAL_ABC_IMPL_TYPE = type(
    collections.abc.Sequence.__dict__["_abc_impl"]
)
_TRUSTED_STDLIB_HEAP_CLASS_MEMBERS = {
    abc.ABCMeta: dict(abc.ABCMeta.__dict__),
    enum.Enum: dict(enum.Enum.__dict__),
    type(enum.Enum): dict(type(enum.Enum).__dict__),
    unittest.TestCase: dict(unittest.TestCase.__dict__),
}
_TRUSTED_STDLIB_CLASSES_REQUIRING_PROJECTION = frozenset(
    {unittest.TestCase}
)
_SAFE_BUILTIN_ITERATOR_TYPES = frozenset(
    {
        type(iter([])),
        type(iter(())),
        type(iter("")),
        type(iter(b"")),
        type(iter(range(0))),
        type(reversed([])),
        type(enumerate(())),
        type(zip()),
        type(map(int, ())),
        type(filter(None, ())),
    }
)
_ORIGINAL_TYPING_EXPORTED_OBJECT_NAMES: dict[int, tuple[str, ...]] = {}
for _typing_name, _typing_value in tuple(vars(typing).items()):
    if (
        type(_typing_name) is str
        and not _typing_name.startswith("_")
        and type(_typing_value).__module__ == "typing"
        and not isinstance(_typing_value, type)
    ):
        _typing_identity = id(_typing_value)
        _ORIGINAL_TYPING_EXPORTED_OBJECT_NAMES[_typing_identity] = tuple(
            sorted(
                {
                    *_ORIGINAL_TYPING_EXPORTED_OBJECT_NAMES.get(
                        _typing_identity, ()
                    ),
                    _typing_name,
                }
            )
        )
del _typing_name, _typing_value, _typing_identity
_ORIGINAL_TYPING_TYPEVAR_TYPE = type(typing.TypeVar("_GpmeepTypeVar"))
_ORIGINAL_TYPING_FORWARD_REF_TYPE = type(typing.ForwardRef("_GpmeepForward"))
_NO_TYPING_ANNOTATION_TOKEN = object()


try:
    from meep import _meep as _NATIVE_MEEP_EXTENSION
except (ImportError, AttributeError):
    _NATIVE_EXTERNAL_SOURCE_COLLECTOR = None
else:
    _NATIVE_EXTERNAL_SOURCE_COLLECTOR = getattr(
        _NATIVE_MEEP_EXTENSION,
        "_gpmeep_fields_external_source_states",
        None,
    )
    if not isinstance(
        _NATIVE_EXTERNAL_SOURCE_COLLECTOR, types.BuiltinFunctionType
    ):
        _NATIVE_EXTERNAL_SOURCE_COLLECTOR = None


POLICY_VERSION = "gpmeep-v110-explicit-backend-guard-v24"
PROFILER_OR_ABLATION_ID = "v110-m1-m2-aunp-crossover-calibration"
MINIMUM_RANK_LOCAL_CELLS = 524_288
MINIMUM_EXPECTED_TIMESTEPS = 256
BYTES_PER_LOCAL_CELL = 1024
FIXED_DEVICE_BYTES = 64 * 1024 * 1024
DEVICE_CAPACITY_FRACTION = 0.80
MINIMUM_REASON_BYTES = 8
MAXIMUM_REASON_BYTES = 512
MAXIMUM_CALLABLE_DEPENDENCIES = 2048
MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES = 32
_ACKNOWLEDGEMENT_BINDING_KEYS = (
    "selected_backend",
    "recommended_backend",
    "workload_fingerprint",
    "recommendation_policy_version",
    "mpi_topology_digest",
    "native_state_digest",
    "process_nonce",
    "run_nonce",
)

_IMMUTABLE_DESCRIPTOR_TYPES = tuple(
    descriptor_type
    for descriptor_type in (
        getattr(types, "WrapperDescriptorType", None),
        getattr(types, "MethodDescriptorType", None),
        getattr(types, "ClassMethodDescriptorType", None),
        getattr(types, "MethodWrapperType", None),
    )
    if descriptor_type is not None
)


class BackendRecommendationError(RuntimeError):
    """A recommendation contract or acknowledgement was malformed."""


class BackendRecommendationMismatch(BackendRecommendationError):
    """The selected backend differs from the guarded recommendation."""

    def __init__(self, message: str, challenge: str, telemetry: dict[str, Any]):
        super().__init__(message)
        self.challenge = challenge
        self.telemetry = dict(telemetry)


class BackendEnvironmentError(BackendRecommendationError):
    """The selected CUDA execution has a non-acknowledgeable environment defect."""

    def __init__(self, message: str, telemetry: dict[str, Any]):
        super().__init__(message)
        self.telemetry = dict(telemetry)


def _numpy_hidden_dispatcher_snapshot(implementation: Any) -> dict[str, Any] | None:
    """Locate and freeze the Python dispatcher named by NumPy's decorator."""
    if not isinstance(implementation, types.FunctionType):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(implementation)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return None
    hidden_name = None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != implementation.__name__:
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function = decorator.func
            decorator_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute) else ""
            )
            if decorator_name != "array_function_dispatch":
                continue
            if isinstance(decorator.args[0], ast.Name):
                hidden_name = decorator.args[0].id
                break
        if hidden_name is not None:
            break
    hidden = (
        implementation.__globals__.get(hidden_name)
        if hidden_name is not None
        else None
    )
    if not isinstance(hidden, types.FunctionType):
        return None
    return {
        "name": hidden_name,
        "function": hidden,
        "code": hidden.__code__,
        "defaults": hidden.__defaults__,
        "kwdefaults": hidden.__kwdefaults__,
        "closure": hidden.__closure__,
    }


def _snapshot_numpy_array_function_dispatchers() -> tuple[
    type | None,
    dict[int, dict[str, Any]],
    str,
]:
    """Freeze already-loaded NumPy dispatchers before user code can replace them."""
    try:
        import numpy as np
    except ImportError:
        return None, {}, ""
    dispatcher_type = type(getattr(np, "zeros_like", None))
    if (
        dispatcher_type.__module__ != "numpy"
        or dispatcher_type.__qualname__ != "_ArrayFunctionDispatcher"
    ):
        return None, {}, str(np.__version__)
    mutable: dict[int, dict[str, Any]] = {}
    for module_name, module in sorted(tuple(sys.modules.items())):
        if module_name != "numpy" and not module_name.startswith("numpy."):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for export_name, candidate in sorted(tuple(namespace.items())):
            if type(candidate) is not dispatcher_type:
                continue
            identity = id(candidate)
            implementation = getattr(candidate, "_implementation", None)
            existing = mutable.get(identity)
            if existing is None:
                mutable[identity] = {
                    "dispatcher": candidate,
                    "exports": [(module_name, export_name)],
                    "implementation": implementation,
                    "implementation_code": getattr(implementation, "__code__", None),
                    "implementation_defaults": getattr(
                        implementation, "__defaults__", None
                    ),
                    "implementation_kwdefaults": getattr(
                        implementation, "__kwdefaults__", None
                    ),
                    "implementation_closure": getattr(
                        implementation, "__closure__", None
                    ),
                    "hidden": _numpy_hidden_dispatcher_snapshot(implementation),
                }
            else:
                existing["exports"].append((module_name, export_name))
    frozen = {}
    for identity, snapshot in mutable.items():
        frozen[identity] = {
            **snapshot,
            "exports": tuple(sorted(snapshot["exports"])),
        }
    return dispatcher_type, frozen, str(np.__version__)


(
    _ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHER_TYPE,
    _ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHERS,
    _ORIGINAL_NUMPY_VERSION,
) = _snapshot_numpy_array_function_dispatchers()


def _snapshot_numpy_primitive_types() -> tuple[
    Any, Any, Any, Any, Any, Any, frozenset[type], frozenset[type], dict[type, Any]
]:
    try:
        import numpy as np
    except ImportError:
        return (
            None, None, None, None, None, None,
            frozenset(), frozenset(), {},
        )
    dtype_types = set()
    scalar_types = set()
    candidates = list(getattr(np, "sctypeDict", {}).values())
    candidates.extend(
        (
            "?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
            "f2", "f4", "f8", "c8", "c16", "S1", "U1", "V1", "O",
            "M8[ns]", "m8[ns]", [("field", "i4")], ("f8", (2,)),
        )
    )
    for candidate in candidates:
        try:
            dtype = np.dtype(candidate)
            dtype_types.add(type(dtype))
            scalar_types.add(dtype.type)
        except (TypeError, ValueError):
            continue
    scalar_tobytes = {
        scalar_type: getattr(scalar_type, "tobytes", None)
        for scalar_type in scalar_types
    }
    return (
        np.ndarray,
        np.dtype,
        np.generic,
        np.ufunc,
        np.ndarray.tobytes,
        np.ndarray.tolist,
        frozenset(dtype_types),
        frozenset(scalar_types),
        scalar_tobytes,
    )


(
    _ORIGINAL_NUMPY_NDARRAY_TYPE,
    _ORIGINAL_NUMPY_DTYPE_TYPE,
    _ORIGINAL_NUMPY_GENERIC_TYPE,
    _ORIGINAL_NUMPY_UFUNC_TYPE,
    _ORIGINAL_NUMPY_NDARRAY_TOBYTES,
    _ORIGINAL_NUMPY_NDARRAY_TOLIST,
    _ORIGINAL_NUMPY_BUILTIN_DTYPE_TYPES,
    _ORIGINAL_NUMPY_BUILTIN_SCALAR_TYPES,
    _ORIGINAL_NUMPY_SCALAR_TOBYTES,
) = _snapshot_numpy_primitive_types()


def _snapshot_numpy_rng_types() -> tuple[Any, ...]:
    """Freeze the exact NumPy RNG surface supported by workload hashing."""

    try:
        import numpy as np
    except ImportError:
        return (None,) * 16
    random_module = getattr(np, "random", None)
    namespace = getattr(random_module, "__dict__", None)
    if not isinstance(namespace, dict):
        return (None,) * 16
    generator = namespace.get("Generator")
    bit_generator = namespace.get("BitGenerator")
    pcg64 = namespace.get("PCG64")
    seed_sequence = namespace.get("SeedSequence")
    if not all(
        isinstance(item, type)
        for item in (generator, bit_generator, pcg64, seed_sequence)
    ) or type(np.__version__) is not str:
        return (None,) * 16
    generator_namespace = type.__getattribute__(generator, "__dict__")
    bit_generator_namespace = type.__getattribute__(bit_generator, "__dict__")
    pcg64_namespace = type.__getattribute__(pcg64, "__dict__")
    seed_sequence_namespace = type.__getattribute__(seed_sequence, "__dict__")
    generate_state = seed_sequence_namespace.get("generate_state")
    if not isinstance(generate_state, types.FunctionType):
        return (None,) * 16
    closure = generate_state.__closure__
    if type(closure) is not tuple or not closure:
        return (None,) * 16
    try:
        closure_values = tuple(cell.cell_contents for cell in closure)
    except ValueError:
        return (None,) * 16
    call_target = closure_values[0]
    try:
        errstate_owner = closure_values[
            generate_state.__code__.co_freevars.index("self")
        ]
        errstate_owner_state = tuple(
            (name, getattr(errstate_owner, name))
            for name in (
                "_call",
                "_all",
                "_divide",
                "_over",
                "_under",
                "_invalid",
            )
        )
    except (AttributeError, ValueError):
        return (None,) * 16
    target_namespace = getattr(call_target, "__dict__", None)
    if not isinstance(target_namespace, dict):
        return (None,) * 16
    generate_state_integrity = {
        "code": generate_state.__code__,
        "defaults": generate_state.__defaults__,
        "kwdefaults": generate_state.__kwdefaults__,
        "closure": closure,
        "closure_cells": tuple(closure),
        "closure_values": closure_values,
        "errstate_owner": errstate_owner,
        "errstate_owner_type": type(errstate_owner),
        "errstate_owner_state": errstate_owner_state,
        "globals": tuple(
            (name, generate_state.__globals__[name])
            for name in generate_state.__code__.co_names
            if name in generate_state.__globals__
        ),
        "call_target": call_target,
        "call_target_type": type(call_target),
        "call_target_code": getattr(call_target, "__code__", None),
        "call_target_defaults": getattr(call_target, "__defaults__", None),
        "call_target_kwdefaults": getattr(
            call_target, "__kwdefaults__", None
        ),
        "call_target_closure": getattr(call_target, "__closure__", None),
        "call_target_namespace": tuple(target_namespace.items()),
    }
    return (
        random_module,
        generator,
        bit_generator,
        pcg64,
        seed_sequence,
        generator_namespace.get("bit_generator"),
        generator_namespace.get("standard_normal"),
        bit_generator_namespace.get("seed_seq"),
        bit_generator_namespace.get("spawn"),
        pcg64_namespace.get("state"),
        seed_sequence_namespace.get("state"),
        seed_sequence_namespace.get("pool"),
        seed_sequence_namespace.get("spawn"),
        generate_state,
        generate_state_integrity,
        np.__version__,
    )


(
    _ORIGINAL_NUMPY_RANDOM_MODULE,
    _ORIGINAL_NUMPY_GENERATOR_TYPE,
    _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE,
    _ORIGINAL_NUMPY_PCG64_TYPE,
    _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE,
    _ORIGINAL_NUMPY_GENERATOR_BIT_GENERATOR_DESCRIPTOR,
    _ORIGINAL_NUMPY_GENERATOR_STANDARD_NORMAL,
    _ORIGINAL_NUMPY_BIT_GENERATOR_SEED_SEQUENCE_DESCRIPTOR,
    _ORIGINAL_NUMPY_BIT_GENERATOR_SPAWN,
    _ORIGINAL_NUMPY_PCG64_STATE_DESCRIPTOR,
    _ORIGINAL_NUMPY_SEED_SEQUENCE_STATE_DESCRIPTOR,
    _ORIGINAL_NUMPY_SEED_SEQUENCE_POOL_DESCRIPTOR,
    _ORIGINAL_NUMPY_SEED_SEQUENCE_SPAWN,
    _ORIGINAL_NUMPY_SEED_SEQUENCE_GENERATE_STATE,
    _ORIGINAL_NUMPY_SEED_SEQUENCE_GENERATE_STATE_INTEGRITY,
    _ORIGINAL_NUMPY_RNG_VERSION,
) = _snapshot_numpy_rng_types()


def _snapshot_numpy_ufuncs() -> dict[int, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError:
        return {}
    mutable: dict[int, dict[str, Any]] = {}
    for module_name, module in sorted(tuple(sys.modules.items())):
        if module_name != "numpy" and not module_name.startswith("numpy."):
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for export_name, candidate in sorted(tuple(namespace.items())):
            if type(candidate) is not _ORIGINAL_NUMPY_UFUNC_TYPE:
                continue
            identity = id(candidate)
            snapshot = mutable.setdefault(
                identity,
                {"ufunc": candidate, "exports": []},
            )
            snapshot["exports"].append((module_name, export_name))
    return {
        identity: {
            "ufunc": snapshot["ufunc"],
            "exports": tuple(sorted(snapshot["exports"])),
        }
        for identity, snapshot in mutable.items()
    }


_ORIGINAL_NUMPY_UFUNCS = _snapshot_numpy_ufuncs()


def _snapshot_numpy_ufunc_context() -> tuple[
    Any, Any, str, str, Any, Any, Any, Any, Any, Any
]:
    module_name = "numpy._core._ufunc_config"
    export_name = "_extobj_contextvar"
    module = sys.modules.get(module_name)
    namespace = getattr(module, "__dict__", None)
    value = namespace.get(export_name) if isinstance(namespace, dict) else None
    if type(value) is not contextvars.ContextVar:
        return (
            None, None, module_name, export_name,
            None, None, None, None, None, None,
        )
    try:
        default = contextvars.Context().run(value.get)
    except LookupError:
        return (
            None, None, module_name, export_name,
            None, None, None, None, None, None,
        )
    numpy_module = sys.modules.get("numpy")
    numpy_namespace = getattr(numpy_module, "__dict__", None)
    geterr = (
        numpy_namespace.get("geterr")
        if isinstance(numpy_namespace, dict)
        else None
    )
    geterrcall = (
        numpy_namespace.get("geterrcall")
        if isinstance(numpy_namespace, dict)
        else None
    )
    get_extobj_dict = (
        geterr.__globals__.get("_get_extobj_dict")
        if isinstance(geterr, types.FunctionType)
        else None
    )
    return (
        value,
        default,
        module_name,
        export_name,
        geterr,
        getattr(geterr, "__code__", None),
        geterrcall,
        getattr(geterrcall, "__code__", None),
        get_extobj_dict,
        getattr(get_extobj_dict, "__code__", None),
    )


(
    _ORIGINAL_NUMPY_UFUNC_CONTEXT,
    _ORIGINAL_NUMPY_UFUNC_CONTEXT_DEFAULT,
    _ORIGINAL_NUMPY_UFUNC_CONTEXT_MODULE,
    _ORIGINAL_NUMPY_UFUNC_CONTEXT_EXPORT,
    _ORIGINAL_NUMPY_GETERR,
    _ORIGINAL_NUMPY_GETERR_CODE,
    _ORIGINAL_NUMPY_GETERRCALL,
    _ORIGINAL_NUMPY_GETERRCALL_CODE,
    _ORIGINAL_NUMPY_GET_EXTOBJ_DICT,
    _ORIGINAL_NUMPY_GET_EXTOBJ_DICT_CODE,
) = _snapshot_numpy_ufunc_context()


def _numpy_ufunc_context_token(seen: set[int]) -> Any:
    if _ORIGINAL_NUMPY_UFUNC_CONTEXT is None:
        return None
    return _canonical(_ORIGINAL_NUMPY_UFUNC_CONTEXT, seen)


_LOCK = threading.RLock()
_PROCESS_PID = os.getpid()
_PROCESS_NONCE = secrets.token_hex(32)
_PENDING_BY_SIMULATION: weakref.WeakKeyDictionary[Any, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_FILE_INPUT_SEALS_BY_SIMULATION: weakref.WeakKeyDictionary[
    Any, dict[str, dict[str, Any]]
] = weakref.WeakKeyDictionary()
_PENDING_BY_CHALLENGE: dict[str, weakref.ReferenceType[Any]] = {}
_CONSUMED_RUN_NONCES: set[str] = set()
_ACKNOWLEDGED_POLICY_SCOPES: weakref.WeakKeyDictionary[
    Any, dict[str, Any]
] = weakref.WeakKeyDictionary()
_ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE = 0
_ACKNOWLEDGED_PROCESS_POLICY_SCOPES: dict[str, dict[str, Any]] = {}
_ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE = 0
_LAST_TELEMETRY: dict[str, Any] | None = None
_DEVICE_INVENTORY_CACHE: dict[str, Any] | None = None
_DEVICE_INVENTORY_ENVIRONMENT_NAMES = (
    "MEEP_GPU_BACKEND",
    "MEEP_GPU_DEVICE",
    "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
    "MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
)


def _reset_after_fork_child() -> None:
    """Discard every inherited acknowledgement capability in a fork child."""
    global _LOCK, _PROCESS_PID, _PROCESS_NONCE
    global _PENDING_BY_SIMULATION, _FILE_INPUT_SEALS_BY_SIMULATION
    global _PENDING_BY_CHALLENGE, _CONSUMED_RUN_NONCES
    global _ACKNOWLEDGED_POLICY_SCOPES, _ACKNOWLEDGED_PROCESS_POLICY_SCOPES
    global _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE
    global _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE, _LAST_TELEMETRY
    global _DEVICE_INVENTORY_CACHE
    _LOCK = threading.RLock()
    _PROCESS_PID = os.getpid()
    _PROCESS_NONCE = secrets.token_hex(32)
    _PENDING_BY_SIMULATION = weakref.WeakKeyDictionary()
    _FILE_INPUT_SEALS_BY_SIMULATION = weakref.WeakKeyDictionary()
    _PENDING_BY_CHALLENGE = {}
    _CONSUMED_RUN_NONCES = set()
    _ACKNOWLEDGED_POLICY_SCOPES = weakref.WeakKeyDictionary()
    _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE = 0
    _ACKNOWLEDGED_PROCESS_POLICY_SCOPES = {}
    _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE = 0
    _LAST_TELEMETRY = None
    _DEVICE_INVENTORY_CACHE = None


def _ensure_process_local_state() -> None:
    """Provide a lazy fork barrier on platforms without an at-fork callback."""
    if _PROCESS_PID != os.getpid():
        _reset_after_fork_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork_child)

_SIMULATION_NATIVE_RUNTIME_FIELDS = {
    "absorber_vols",
    "dft_data_list",
    "fields",
    "fragment_stats",
    "geps",
    "gv",
    "pml_vols1",
    "pml_vols2",
    "pml_vols3",
    "structure",
}
_SIMULATION_UNBOUND_RUNTIME_FIELDS = _SIMULATION_NATIVE_RUNTIME_FIELDS - {
    "fields",
    "structure",
}
_SIMULATION_FILE_INPUT_FIELDS = {
    "epsilon_input_file",
    "load_fields_file",
    "load_structure_file",
}
_SIMULATION_RANK_LOCAL_LOAD_FIELDS = {
    "load_fields_file",
    "load_structure_file",
}
_SIMULATION_CHUNK_LAYOUT_FIELDS = {
    "chunk_layout",
    "_chunk_layout_original",
}
_EPHEMERAL_OBJECT_FIELDS = {
    "fields",
    "structure",
    "swigobj",
    "swig_obj",
    "progress",
    "interactive",
    "run_index",
    "filename_prefix",
}

_CLASS_METADATA_FIELDS = {
    "__dict__",
    "__weakref__",
    "__doc__",
    "__module__",
    "__qualname__",
}
_PY_TPFLAGS_HEAPTYPE = 1 << 9
_LATE_TRUSTED_MEEP_CLASS_INVENTORY = {
    ("meep.adjoint.filter_source", "FilteredSource"): {
        "relative_path": ("adjoint", "filter_source.py"),
        "class_doc": None,
        "base_global": "CustomSource",
        "base_export": ("meep", "CustomSource"),
        "module_globals": {
            "linalg": "scipy.linalg",
            "np": "numpy",
        },
        "defaults": {"__init__": (None,)},
        "annotations": {
            "__init__": {
                "center_frequency": float,
                "frequencies": ("np", "ndarray"),
            }
        },
        "methods": frozenset(
            {
                "__call__",
                "__init__",
                "__setattr__",
                "cos_window_fd",
                "cos_window_td",
                "dtft",
                "estimate_impulse_response",
                "func",
                "hann",
                "hann_dtft",
                "nuttall",
                "nuttall_bandwidth",
                "nuttall_dtft",
                "rect",
                "sinc",
            }
        ),
    }
}


class _CanonicalTraversal(set):
    """Cycle state plus a per-fingerprint callable dependency memo."""

    def __init__(self, *, memoize_trusted_meep_dependency_graph: bool = False):
        super().__init__()
        if type(memoize_trusted_meep_dependency_graph) is not bool:
            raise BackendRecommendationError(
                "trusted Meep dependency memoization mode is malformed"
            )
        self.callable_dependencies: dict[tuple[int, int], Any] = {}
        self.callable_dependency_count = 0
        self.object_references: dict[int, int] = {}
        # Keep every referenced object alive for the complete traversal.
        # Storing only ``id(value)`` lets CPython recycle the address of a
        # short-lived intermediate object and makes an unrelated later value
        # look like an alias.  Besides corrupting graph topology, allocator
        # timing can then produce different workload fingerprints on MPI
        # ranks with identical semantic inputs.
        self.object_values: dict[int, Any] = {}
        self.class_tokens: dict[int, Any] = {}
        self.active_classes: set[int] = set()
        self.native_simulation_roots: set[int] = set()
        self.reject_unprojected_native_roots = False
        self.trusted_stdlib_function_integrity: dict[int, bool] = {}
        # Disassembly is reusable only for code whose constants cannot run
        # user __repr__/__str__ hooks or change underneath an Instruction.
        # Keep the code itself alive; this memo never contains canonical
        # values, workload tokens, or mutable function/runtime state.
        self.bytecode_instructions: dict[
            int, tuple[types.CodeType, tuple[dis.Instruction, ...]]
        ] = {}
        # Exact workload fingerprints repeatedly encounter the same immutable
        # package class/function graph through Simulation, sources, monitors,
        # and callbacks.  A whole-graph check is stronger than the old
        # per-root checks and may be shared only inside one bounded canonical
        # traversal.  The caller must perform the matching final check before
        # using the token so an intervening mutation still fails closed.
        self.memoize_trusted_meep_dependency_graph = (
            memoize_trusted_meep_dependency_graph
        )
        self.trusted_meep_dependency_graph_checked = False


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _trusted_meep_class_members(
    candidate: type, package_root: Path
) -> dict[str, Any] | None:
    """Return an immutable-member snapshot for one package-defined class."""
    definition_is_trusted = True
    for member in candidate.__dict__.values():
        functions: tuple[Any, ...]
        if isinstance(member, (staticmethod, classmethod)):
            functions = (member.__func__,)
        elif isinstance(member, property):
            functions = (member.fget, member.fset, member.fdel)
        elif isinstance(member, types.FunctionType):
            functions = (member,)
        else:
            functions = ()
        for function in functions:
            code = getattr(function, "__code__", None)
            if code is None:
                continue
            try:
                code_path = Path(code.co_filename).resolve(strict=True)
            except OSError:
                definition_is_trusted = False
                break
            if not _path_is_within(code_path, package_root):
                definition_is_trusted = False
                break
        if not definition_is_trusted:
            break
    return dict(candidate.__dict__) if definition_is_trusted else None


def _late_class_definition_matches_source(
    candidate: type,
    source_path: Path,
    inventory: dict[str, Any],
    module: types.ModuleType,
) -> bool:
    """Require a late class to be the exact implementation shipped on disk."""
    expected_methods = inventory["methods"]
    try:
        source = source_path.read_bytes()
        module_code = compile(
            source,
            str(source_path),
            "exec",
            dont_inherit=True,
            optimize=-1,
        )
    except (OSError, SyntaxError, ValueError):
        return False
    class_codes = [
        item
        for item in module_code.co_consts
        if isinstance(item, types.CodeType)
        and item.co_name == candidate.__qualname__
        and item.co_qualname == candidate.__qualname__
    ]
    if len(class_codes) != 1:
        return False
    expected_codes = {
        item.co_name: item
        for item in class_codes[0].co_consts
        if isinstance(item, types.CodeType)
        and item.co_name in expected_methods
        and item.co_qualname
        == f"{candidate.__qualname__}.{item.co_name}"
    }
    actual_codes = {
        name: member.__code__
        for name, member in candidate.__dict__.items()
        if name in expected_methods and isinstance(member, types.FunctionType)
    }
    if (
        set(expected_codes) != expected_methods
        or set(actual_codes) != expected_methods
        or set(candidate.__dict__) - _CLASS_METADATA_FIELDS - expected_methods
        or type(candidate) is not type
        or candidate.__name__ != candidate.__qualname__.rpartition(".")[2]
        or candidate.__doc__ != inventory["class_doc"]
    ):
        return False
    base = module.__dict__.get(inventory["base_global"])
    base_module_name, base_export_name = inventory["base_export"]
    base_module = sys.modules.get(base_module_name)
    if (
        not isinstance(base, type)
        or not isinstance(base_module, types.ModuleType)
        or base_module.__dict__.get(base_export_name) is not base
        or not _is_trusted_meep_class(base)
        or candidate.__bases__ != (base,)
    ):
        return False
    if candidate.__mro__ != (candidate, *base.__mro__):
        return False
    for global_name, dependency_name in inventory["module_globals"].items():
        dependency = sys.modules.get(dependency_name)
        if (
            not isinstance(dependency, types.ModuleType)
            or module.__dict__.get(global_name) is not dependency
            or getattr(getattr(dependency, "__spec__", None), "name", None)
            != dependency_name
        ):
            return False
    expected_defaults = inventory["defaults"]
    expected_annotations = inventory["annotations"]
    for name in expected_methods:
        function = candidate.__dict__[name]
        if (
            actual_codes[name] != expected_codes[name]
            or function.__globals__ is not module.__dict__
            or function.__name__ != name
            or function.__module__ != candidate.__module__
            or function.__qualname__ != f"{candidate.__qualname__}.{name}"
            or function.__doc__
            != (
                expected_codes[name].co_consts[0]
                if expected_codes[name].co_consts
                and isinstance(expected_codes[name].co_consts[0], str)
                else None
            )
            or function.__defaults__ != expected_defaults.get(name)
            or function.__kwdefaults__ is not None
            or function.__dict__
        ):
            return False
        annotations = expected_annotations.get(name, {})
        resolved_annotations: dict[str, Any] = {}
        for key, value in annotations.items():
            if isinstance(value, tuple):
                owner = module.__dict__.get(value[0])
                value = getattr(owner, value[1], None)
            resolved_annotations[key] = value
        if function.__annotations__ != resolved_annotations:
            return False
        closure = function.__closure__
        if function.__code__.co_freevars == ("__class__",):
            if (
                closure is None
                or len(closure) != 1
                or closure[0].cell_contents is not candidate
            ):
                return False
        elif closure is not None:
            return False
    return True


def _late_class_global_snapshot(
    candidate: type, module: types.ModuleType
) -> dict[str, Any]:
    """Freeze globals that the trusted package implementation can resolve."""
    referenced: set[str] = set()
    for member in candidate.__dict__.values():
        if not isinstance(member, types.FunctionType):
            continue
        for code in _nested_code_objects(member.__code__):
            referenced.update(code.co_names)
    namespace = module.__dict__
    return {
        name: namespace[name]
        for name in sorted(referenced)
        if name != "__builtins__" and name in namespace
    }


def _is_stable_module_attribute_dependency(value: Any) -> bool:
    bound_self = getattr(value, "__self__", None)
    return (
        type(value) is types.FunctionType
        or isinstance(value, (type, types.ModuleType))
        or type(value) in (type(None), bool, int, float, str, bytes)
        or (
            _ORIGINAL_NUMPY_UFUNC_TYPE is not None
            and type(value) is _ORIGINAL_NUMPY_UFUNC_TYPE
        )
        or (
            _ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHER_TYPE is not None
            and type(value)
            is _ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHER_TYPE
        )
        or (
            isinstance(
                value,
                (
                    types.BuiltinFunctionType,
                    types.BuiltinMethodType,
                ),
            )
            and (
                bound_self is None
                or isinstance(bound_self, (type, types.ModuleType))
            )
        )
    )


def _late_class_runtime_snapshot(candidate: type) -> dict[str, Any]:
    """Freeze mutable metadata and closure cells of a validated late class."""
    members: dict[str, Any] = {}
    for name, function in candidate.__dict__.items():
        if not isinstance(function, types.FunctionType):
            continue
        closure = function.__closure__ or ()
        module_attributes: dict[tuple[str, ...], Any] = {}
        for code in _nested_code_objects(function.__code__):
            instructions = list(dis.get_instructions(code))
            for index, instruction in enumerate(instructions):
                if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
                    continue
                root_name = str(instruction.argval)
                value = function.__globals__.get(root_name)
                if not isinstance(value, types.ModuleType):
                    continue
                path = [root_name]
                cursor = index + 1
                while cursor < len(instructions) and instructions[
                    cursor
                ].opname in ("LOAD_ATTR", "LOAD_METHOD"):
                    attribute = str(instructions[cursor].argval)
                    try:
                        value = getattr(value, attribute)
                    except Exception:
                        break
                    path.append(attribute)
                    if _is_stable_module_attribute_dependency(value):
                        module_attributes[tuple(path)] = value
                    cursor += 1
        members[name] = {
            "function": function,
            "code": function.__code__,
            "globals": function.__globals__,
            "name": function.__name__,
            "module": function.__module__,
            "qualname": function.__qualname__,
            "doc": function.__doc__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "annotations_object": function.__annotations__,
            "annotations": dict(function.__annotations__),
            "attributes": dict(function.__dict__),
            "closure_cells": closure,
            "closure_contents": tuple(cell.cell_contents for cell in closure),
            "module_attributes": module_attributes,
        }
    return {
        "name": candidate.__name__,
        "module": candidate.__module__,
        "qualname": candidate.__qualname__,
        "doc": candidate.__doc__,
        "bases": candidate.__bases__,
        "mro": candidate.__mro__,
        "class_keys": frozenset(candidate.__dict__),
        "members": members,
    }


def _late_class_runtime_is_unchanged(
    candidate: type, snapshot: dict[str, Any]
) -> bool:
    if (
        candidate.__name__ != snapshot["name"]
        or candidate.__module__ != snapshot["module"]
        or candidate.__qualname__ != snapshot["qualname"]
        or candidate.__doc__ != snapshot["doc"]
        or candidate.__bases__ != snapshot["bases"]
        or candidate.__mro__ != snapshot["mro"]
        or frozenset(candidate.__dict__) != snapshot["class_keys"]
    ):
        return False
    for name, expected in snapshot["members"].items():
        function = candidate.__dict__.get(name)
        if (
            function is not expected["function"]
            or function.__code__ is not expected["code"]
            or function.__globals__ is not expected["globals"]
            or function.__name__ != expected["name"]
            or function.__module__ != expected["module"]
            or function.__qualname__ != expected["qualname"]
            or function.__doc__ != expected["doc"]
            or function.__defaults__ is not expected["defaults"]
            or function.__kwdefaults__ is not expected["kwdefaults"]
            or function.__annotations__ is not expected["annotations_object"]
            or function.__dict__ != expected["attributes"]
        ):
            return False
        annotations = function.__annotations__
        if set(annotations) != set(expected["annotations"]) or any(
            annotations[key] is not value
            for key, value in expected["annotations"].items()
        ):
            return False
        closure = function.__closure__ or ()
        if len(closure) != len(expected["closure_cells"]):
            return False
        for cell, original_cell, original_value in zip(
            closure,
            expected["closure_cells"],
            expected["closure_contents"],
        ):
            if cell is not original_cell or cell.cell_contents is not original_value:
                return False
        for path, original_value in expected["module_attributes"].items():
            value = function.__globals__.get(path[0])
            try:
                for attribute in path[1:]:
                    value = getattr(value, attribute)
            except Exception:
                return False
            if value is not original_value:
                return False
    return True


def _trusted_meep_class_snapshot() -> tuple[
    frozenset[type], dict[type, dict[str, Any]]
]:
    """Freeze package-owned classes and their original members.

    A class remains package-owned only by identity.  Individual members are
    trusted only while the exact descriptor captured during import is still
    installed, so monkey-patched Meep classes cannot regain shallow callable
    canonicalization merely by retaining their original class identity.
    """
    package_root = Path(__file__).resolve().parent
    trusted: set[type] = set()
    trusted_members: dict[type, dict[str, Any]] = {}
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "meep" and not module_name.startswith("meep."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        try:
            resolved_module = Path(module_file).resolve(strict=True)
        except OSError:
            continue
        if not _path_is_within(resolved_module, package_root):
            continue
        for candidate in tuple(getattr(module, "__dict__", {}).values()):
            if not isinstance(candidate, type):
                continue
            if candidate.__module__ != module_name:
                continue
            members = _trusted_meep_class_members(candidate, package_root)
            if members is not None:
                trusted.add(candidate)
                trusted_members[candidate] = members
    return frozenset(trusted), trusted_members


_TRUSTED_MEEP_CLASSES, _TRUSTED_MEEP_CLASS_MEMBERS = (
    _trusted_meep_class_snapshot()
)
_LATE_TRUSTED_MEEP_CLASS_GLOBALS: dict[type, dict[str, Any]] = {}
_LATE_TRUSTED_MEEP_CLASS_RUNTIME: dict[type, dict[str, Any]] = {}


def _trusted_annotation_description(
    value: Any, active: set[int] | None = None
) -> Any:
    """Create a stable token for an import-time trusted annotation value."""
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return {"literal": value}
    if value_type is float:
        return {"float64": struct.pack("!d", value).hex()}
    if value_type is bytes:
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    exported_names = _ORIGINAL_TYPING_EXPORTED_OBJECT_NAMES.get(id(value))
    if exported_names is not None:
        return {"typing_exports": list(exported_names)}
    if isinstance(value, type):
        try:
            module = type.__getattribute__(value, "__module__")
            qualname = type.__getattribute__(value, "__qualname__")
        except Exception:
            module = value_type.__module__
            qualname = value_type.__qualname__
        return {"type": f"{module}.{qualname}"}
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        return {
            "annotation_cycle": (
                f"{value_type.__module__}.{value_type.__qualname__}"
            )
        }
    active.add(identity)
    try:
        if value_type is _ORIGINAL_TYPING_TYPEVAR_TYPE:
            return {
                "typevar": {
                    "name": value.__name__,
                    "bound": _trusted_annotation_description(
                        value.__bound__, active
                    ),
                    "constraints": [
                        _trusted_annotation_description(item, active)
                        for item in value.__constraints__
                    ],
                    "covariant": bool(value.__covariant__),
                    "contravariant": bool(value.__contravariant__),
                }
            }
        if value_type is _ORIGINAL_TYPING_FORWARD_REF_TYPE:
            return {"forward_ref": value.__forward_arg__}
        try:
            origin = typing.get_origin(value)
            arguments = typing.get_args(value)
        except Exception:
            origin = None
            arguments = ()
        if origin is not None:
            return {
                "typing_alias": {
                    "runtime_type": (
                        f"{value_type.__module__}.{value_type.__qualname__}"
                    ),
                    "origin": _trusted_annotation_description(origin, active),
                    "arguments": [
                        _trusted_annotation_description(item, active)
                        for item in arguments
                    ],
                }
            }
        return {
            "opaque_trusted_annotation": (
                f"{value_type.__module__}.{value_type.__qualname__}"
            )
        }
    finally:
        active.remove(identity)


def _trusted_annotation_integrity_snapshot(
    value: Any, active: set[int] | None = None
) -> Any:
    """Capture a process-local, comparison-safe graph of trusted annotations."""
    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return ("value", value_type.__name__, value)
    if value_type is float:
        return ("float64", struct.pack("!d", value).hex())
    if value_type is complex:
        return (
            "complex128",
            struct.pack("!d", value.real).hex(),
            struct.pack("!d", value.imag).hex(),
        )
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        if value_type in (tuple, list):
            return (
                "sequence",
                value_type.__name__,
                identity,
                tuple(
                    _trusted_annotation_integrity_snapshot(item, active)
                    for item in value
                ),
            )
        if value_type is dict:
            return (
                "mapping",
                identity,
                tuple(
                    (
                        _trusted_annotation_integrity_snapshot(key, active),
                        _trusted_annotation_integrity_snapshot(item, active),
                    )
                    for key, item in dict.items(value)
                ),
            )
        if isinstance(value, type):
            return ("type_identity", identity)
        if (
            value_type.__module__ == "typing"
            or value_type in (types.GenericAlias, types.UnionType)
        ):
            try:
                state = object.__getattribute__(value, "__dict__")
            except (AttributeError, TypeError):
                state = None
            except Exception as error:
                raise BackendRecommendationError(
                    "trusted typing annotation state cannot be inspected"
                ) from error
            if state is not None and type(state) is not dict:
                raise BackendRecommendationError(
                    "trusted typing annotation state is malformed"
                )
            slot_state: list[Any] = []
            try:
                typing_mro = type.__getattribute__(value_type, "__mro__")
            except Exception as error:
                raise BackendRecommendationError(
                    "trusted typing annotation type cannot be inspected"
                ) from error
            for owner in typing_mro:
                owner_namespace = type.__getattribute__(owner, "__dict__")
                owner_name = type.__getattribute__(owner, "__name__")
                owner_module = type.__getattribute__(owner, "__module__")
                owner_qualname = type.__getattribute__(owner, "__qualname__")
                raw_slots = owner_namespace.get("__slots__", ())
                if type(raw_slots) is str:
                    raw_slots = (raw_slots,)
                elif type(raw_slots) not in (tuple, list):
                    raise BackendRecommendationError(
                        "trusted typing annotation slots are malformed"
                    )
                for raw_name in raw_slots:
                    if type(raw_name) is not str or raw_name in (
                        "__dict__",
                        "__weakref__",
                    ):
                        continue
                    slot_name = raw_name
                    if raw_name.startswith("__") and not raw_name.endswith("__"):
                        slot_name = f"_{owner_name.lstrip('_')}{raw_name}"
                    descriptor = owner_namespace.get(slot_name)
                    if descriptor is None:
                        continue
                    try:
                        slot_value = descriptor.__get__(value, value_type)
                    except AttributeError:
                        slot_token: Any = ("unset",)
                    except Exception as error:
                        raise BackendRecommendationError(
                            "trusted typing annotation slots cannot be inspected"
                        ) from error
                    else:
                        slot_token = _trusted_annotation_integrity_snapshot(
                            slot_value, active
                        )
                    slot_state.append(
                        (
                            f"{owner_module}.{owner_qualname}.{slot_name}",
                            slot_token,
                        )
                    )
            return (
                "typing_state",
                f"{value_type.__module__}.{value_type.__qualname__}",
                identity,
                (
                    None
                    if state is None
                    else _trusted_annotation_integrity_snapshot(state, active)
                ),
                tuple(slot_state),
            )
        return (
            "identity",
            f"{value_type.__module__}.{value_type.__qualname__}",
            identity,
        )
    finally:
        active.remove(identity)


def _trusted_member_functions(member: Any) -> tuple[types.FunctionType, ...]:
    if type(member) in (staticmethod, classmethod):
        candidates = (member.__func__,)
    elif type(member) is property:
        candidates = (member.fget, member.fset, member.fdel)
    elif type(member) is types.FunctionType:
        candidates = (member,)
    else:
        candidates = ()
    return tuple(
        candidate
        for candidate in candidates
        if type(candidate) is types.FunctionType
    )


def _trusted_function_annotation_snapshots(
    member_sets: Any,
) -> dict[types.FunctionType, dict[str, Any]]:
    snapshots: dict[types.FunctionType, dict[str, Any]] = {}
    for members in member_sets:
        for member in members.values():
            for function in _trusted_member_functions(member):
                annotations = getattr(function, "__annotations__", None)
                if type(annotations) is not dict or any(
                    type(key) is not str for key in annotations
                ):
                    continue
                items = tuple(dict.items(annotations))
                snapshots[function] = {
                    "object": annotations,
                    "items": items,
                    "integrity": tuple(
                        _trusted_annotation_integrity_snapshot(value)
                        for _, value in items
                    ),
                    "token": [
                        {
                            "name": key,
                            "value": _trusted_annotation_description(value),
                        }
                        for key, value in items
                    ],
                }
    return snapshots


_TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS = (
    _trusted_function_annotation_snapshots(
        (
            *_TRUSTED_MEEP_CLASS_MEMBERS.values(),
            *_TRUSTED_STDLIB_HEAP_CLASS_MEMBERS.values(),
        )
    )
)


def _trusted_function_annotation_token(
    function: types.FunctionType, annotations: Any
) -> Any | None:
    snapshot = _TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS.get(function)
    if snapshot is None:
        return None
    items = snapshot["items"]
    try:
        current_token = [
            {
                "name": key,
                "value": _trusted_annotation_description(value),
            }
            for key, value in dict.items(annotations)
        ] if type(annotations) is dict else None
        current_integrity = (
            tuple(
                _trusted_annotation_integrity_snapshot(value)
                for _, value in dict.items(annotations)
            )
            if type(annotations) is dict
            else None
        )
    except Exception as error:
        raise BackendRecommendationError(
            "trusted callable annotations cannot be verified exactly"
        ) from error
    if (
        annotations is not snapshot["object"]
        or type(annotations) is not dict
        or len(annotations) != len(items)
        or any(
            current_key != original_key or current_value is not original_value
            for (current_key, current_value), (original_key, original_value)
            in zip(dict.items(annotations), items)
        )
        or current_token != snapshot["token"]
        or current_integrity != snapshot["integrity"]
    ):
        raise BackendRecommendationError(
            "trusted callable annotations differ from their import-time state"
        )
    return snapshot["token"]


def _register_trusted_meep_class(value_type: type) -> None:
    """Register an exact package class loaded after backend-policy startup.

    Optional adjoint modules are imported lazily to keep ordinary Meep startup
    light.  Their classes therefore cannot be present in the import-time
    snapshot above.  A lazy module calls this function while its own class is
    still installed under the defining module and before it is returned to a
    consumer.  The same definition-path checks used by the startup snapshot
    are repeated here; arbitrary classes that merely claim a ``meep.*`` module
    name are not accepted.
    """
    global _TRUSTED_MEEP_CLASSES, _TRUSTED_MEEP_CLASS_MEMBERS
    global _LATE_TRUSTED_MEEP_CLASS_GLOBALS
    global _LATE_TRUSTED_MEEP_CLASS_RUNTIME
    global _TRUSTED_MEEP_DEPENDENCY_GRAPH
    global _TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS
    if not isinstance(value_type, type):
        raise BackendRecommendationError(
            "lazy Meep registration requires an exact class"
        )
    module_name = value_type.__module__
    inventory = _LATE_TRUSTED_MEEP_CLASS_INVENTORY.get(
        (module_name, value_type.__qualname__)
    )
    if inventory is None:
        raise BackendRecommendationError(
            "lazy Meep class is not in the package registration inventory"
        )
    if value_type in _TRUSTED_MEEP_CLASSES:
        return
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    module_origin = getattr(module_spec, "origin", None)
    if (
        module is None
        or not isinstance(module_file, str)
        or not isinstance(module_origin, str)
        or getattr(module_spec, "name", None) != module_name
        or getattr(module, "__package__", None)
        != module_name.rpartition(".")[0]
    ):
        raise BackendRecommendationError(
            "lazy Meep class has no package-owned defining module"
        )
    package_root = Path(__file__).resolve().parent
    try:
        resolved_module = Path(module_file).resolve(strict=True)
        resolved_origin = Path(module_origin).resolve(strict=True)
    except OSError as error:
        raise BackendRecommendationError(
            "lazy Meep class defining module cannot be resolved"
        ) from error
    expected_module = package_root.joinpath(*inventory["relative_path"])
    try:
        resolved_expected = expected_module.resolve(strict=True)
    except OSError as error:
        raise BackendRecommendationError(
            "lazy Meep class expected module cannot be resolved"
        ) from error
    if (
        resolved_module != resolved_origin
        or resolved_module != resolved_expected
        or not _path_is_within(resolved_module, package_root)
    ):
        raise BackendRecommendationError(
            "lazy Meep class defining module is outside the package"
        )
    registration_frame = inspect.currentframe()
    caller = None if registration_frame is None else registration_frame.f_back
    if (
        caller is None
        or caller.f_globals is not getattr(module, "__dict__", None)
        or caller.f_globals.get("__name__") != module_name
    ):
        raise BackendRecommendationError(
            "lazy Meep class registration did not originate in its module"
        )
    try:
        caller_path = Path(caller.f_code.co_filename).resolve(strict=True)
    except OSError as error:
        raise BackendRecommendationError(
            "lazy Meep class registration caller cannot be resolved"
        ) from error
    finally:
        del registration_frame
    if caller_path != resolved_module:
        raise BackendRecommendationError(
            "lazy Meep class registration caller differs from its module"
        )
    components = value_type.__qualname__.split(".")
    if not components or "<locals>" in components:
        raise BackendRecommendationError(
            "lazy Meep class has no stable package qualification"
        )
    sentinel = object()
    owner: Any = module
    for component in components:
        namespace = getattr(owner, "__dict__", None)
        owner = (
            namespace.get(component, sentinel)
            if isinstance(namespace, dict)
            else sentinel
        )
        if owner is sentinel:
            break
    if owner is not value_type:
        raise BackendRecommendationError(
            "lazy Meep class identity differs from its defining module"
        )
    if not _late_class_definition_matches_source(
        value_type,
        resolved_expected,
        inventory,
        module,
    ):
        raise BackendRecommendationError(
            "lazy Meep class definition differs from package source"
        )
    parent: types.ModuleType | None = None
    for index in range(1, len(module_name.split("."))):
        parent_name = ".".join(module_name.split(".")[:index])
        parent_module = sys.modules.get(parent_name)
        if not isinstance(parent_module, types.ModuleType):
            raise BackendRecommendationError(
                "lazy Meep class parent module is not loaded"
            )
        parent_spec = getattr(parent_module, "__spec__", None)
        if getattr(parent_spec, "name", None) != parent_name:
            raise BackendRecommendationError(
                "lazy Meep class parent module identity is malformed"
            )
        if parent is not None:
            child_name = parent_name.rpartition(".")[2]
            if parent.__dict__.get(child_name) is not parent_module:
                raise BackendRecommendationError(
                    "lazy Meep class parent module chain is inconsistent"
                )
        parent = parent_module
    members = _trusted_meep_class_members(value_type, package_root)
    if members is None:
        raise BackendRecommendationError(
            "lazy Meep class definition is outside the package"
        )
    with _LOCK:
        if value_type in _TRUSTED_MEEP_CLASSES:
            return
        _TRUSTED_MEEP_CLASSES = frozenset(
            (*_TRUSTED_MEEP_CLASSES, value_type)
        )
        _TRUSTED_MEEP_CLASS_MEMBERS = {
            **_TRUSTED_MEEP_CLASS_MEMBERS,
            value_type: members,
        }
        _LATE_TRUSTED_MEEP_CLASS_GLOBALS = {
            **_LATE_TRUSTED_MEEP_CLASS_GLOBALS,
            value_type: _late_class_global_snapshot(value_type, module),
        }
        _LATE_TRUSTED_MEEP_CLASS_RUNTIME = {
            **_LATE_TRUSTED_MEEP_CLASS_RUNTIME,
            value_type: _late_class_runtime_snapshot(value_type),
        }
        _TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS = {
            **_TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS,
            **_trusted_function_annotation_snapshots((members,)),
        }
        if "_TRUSTED_MEEP_DEPENDENCY_GRAPH" in globals():
            _TRUSTED_MEEP_DEPENDENCY_GRAPH = (
                _extend_trusted_dependency_graph(
                    _TRUSTED_MEEP_DEPENDENCY_GRAPH,
                    (value_type,),
                    traversal_filter=_is_package_owned_meep_dependency,
                )
            )


def _is_trusted_meep_class(value: type) -> bool:
    if value not in _TRUSTED_MEEP_CLASSES:
        return False
    globals_snapshot = _LATE_TRUSTED_MEEP_CLASS_GLOBALS.get(value)
    if globals_snapshot is None:
        return True
    runtime_snapshot = _LATE_TRUSTED_MEEP_CLASS_RUNTIME.get(value)
    if runtime_snapshot is None or not _late_class_runtime_is_unchanged(
        value, runtime_snapshot
    ):
        return False
    module = sys.modules.get(value.__module__)
    namespace = getattr(module, "__dict__", None)
    return isinstance(namespace, dict) and all(
        namespace.get(name) is dependency
        for name, dependency in globals_snapshot.items()
    )


def _is_original_trusted_meep_member(
    owner: type, name: str, value: Any
) -> bool:
    if not _is_trusted_meep_class(owner):
        return False
    sentinel = object()
    original = _TRUSTED_MEEP_CLASS_MEMBERS.get(owner, {}).get(name, sentinel)
    return original is value


def _is_original_trusted_meep_function(value: Any) -> bool:
    """Return whether a function is an original member of a frozen class."""
    if not isinstance(value, types.FunctionType):
        return False
    for owner, members in _TRUSTED_MEEP_CLASS_MEMBERS.items():
        if not _is_trusted_meep_class(owner):
            continue
        for member in members.values():
            if isinstance(member, (staticmethod, classmethod)):
                candidates = (member.__func__,)
            elif isinstance(member, property):
                candidates = (member.fget, member.fset, member.fdel)
            elif isinstance(member, types.FunctionType):
                candidates = (member,)
            else:
                continue
            if any(candidate is value for candidate in candidates):
                return True
    return False


def _is_original_trusted_stdlib_function(
    value: Any, seen: set[int] | None = None
) -> bool:
    for owner, members in _TRUSTED_STDLIB_HEAP_CLASS_MEMBERS.items():
        for member in members.values():
            if type(member) in (staticmethod, classmethod):
                candidates = (member.__func__,)
            elif type(member) is property:
                candidates = (member.fget, member.fset, member.fdel)
            else:
                candidates = (member,)
            if any(candidate is value for candidate in candidates):
                return (
                    _trusted_stdlib_heap_class_is_unchanged(owner, seen)
                    and _trusted_stdlib_function_is_unchanged(value, seen)
                )
    return False


def _is_exact_trusted_meep_type(value: Any, module: str, qualname: str) -> bool:
    value_type = type(value)
    return (
        value_type.__module__ == module
        and value_type.__qualname__ == qualname
        and _is_trusted_meep_class(value_type)
    )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_bounded_text(value: Any, maximum_bytes: int) -> str:
    """Render diagnostics without invoking an unbounded or escaping __str__."""
    try:
        rendered = str(value)
    except BaseException:
        return "<unprintable>"
    rendered = "".join(
        character if character.isprintable() else "?" for character in rendered
    )
    try:
        payload = rendered.encode("utf-8", errors="backslashreplace")
    except BaseException:
        return "<unencodable>"
    if len(payload) <= maximum_bytes:
        return rendered
    digest = hashlib.sha256(payload).hexdigest()
    prefix_limit = max(0, maximum_bytes - len(digest) - 9)
    prefix = payload[:prefix_limit].decode("utf-8", errors="ignore")
    return f"{prefix}<sha256:{digest}>"


def _preflight_failure_record(error: BaseException) -> dict[str, Any]:
    record = {
        "ok": False,
        "error_type": "local-preflight-exception",
        "error": _safe_bounded_text(error, 384),
    }
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) >= 4096:
        return {
            "ok": False,
            "error_type": "local-preflight-exception",
            "error": "bounded diagnostic overflow",
        }
    return record


def _file_input_manifest(field: str, value: Any) -> Any:
    if value is None:
        return None
    if type(value) is not str:
        raise BackendRecommendationError(
            f"simulation file input {field!r} must be an exact string"
        )
    if not value:
        return value
    raw_path = value
    dataset = ""
    if field == "epsilon_input_file" and ":" in value:
        raw_path, dataset = value.rsplit(":", 1)
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BackendRecommendationError(
            f"simulation file input does not exist: {raw_path}"
        ) from error
    if not resolved.is_file():
        raise BackendRecommendationError(
            f"simulation file input is not a regular file: {raw_path}"
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = resolved.stat()
    except OSError as error:
        raise BackendRecommendationError(
            f"simulation file input cannot be hashed exactly: {raw_path}"
        ) from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    identity_current = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if identity_before != identity_after or identity_after != identity_current:
        raise BackendRecommendationError(
            f"simulation file input changed while it was hashed: {raw_path}"
        )
    return {
        "file_input": field,
        "input_spec": value,
        "path": raw_path,
        "resolved_path": str(resolved),
        "dataset": dataset,
        "exists": True,
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _file_input_is_materialized(simulation: Any, field: str) -> bool:
    if field in ("epsilon_input_file", "load_structure_file"):
        return getattr(simulation, "structure", None) is not None
    if field == "load_fields_file":
        return (
            getattr(simulation, "fields", None) is not None
            and getattr(simulation, "_is_initialized", False) is True
        )
    return False


def _sealed_file_input_manifest(
    simulation: Any, field: str, value: Any
) -> Any:
    """Bind file bytes before init and retain that seal after materialization."""
    if value is None:
        configured = False
    elif type(value) is not str:
        raise BackendRecommendationError(
            f"simulation file input {field!r} must be an exact string"
        )
    else:
        configured = bool(value)
    seals = _FILE_INPUT_SEALS_BY_SIMULATION.get(simulation)
    sealed = seals.get(field) if seals else None
    if _file_input_is_materialized(simulation, field):
        if sealed is not None:
            if sealed.get("configured") is not True:
                return value
            owner = (
                getattr(simulation, "fields", None)
                if field == "load_fields_file"
                else getattr(simulation, "structure", None)
            )
            sealed_owner = sealed.get("_materialized_owner")
            if sealed_owner is None:
                raise BackendRecommendationError(
                    f"materialized simulation file input {field!r} has no "
                    "owner bound at native materialization"
                )
            if sealed_owner is not owner:
                raise BackendRecommendationError(
                    f"materialized simulation file input {field!r} changed "
                    "native owner without resetting its gpmeep seal"
                )
            return {
                "materialized_file_input": {
                    key: item
                    for key, item in sealed.items()
                    if not key.startswith("_")
                }
            }
        if configured:
            raise BackendRecommendationError(
                f"materialized simulation file input {field!r} has no "
                "pre-initialization gpmeep seal"
            )
        return value
    if seals is None:
        seals = {}
        _FILE_INPUT_SEALS_BY_SIMULATION[simulation] = seals
    if not configured:
        seals[field] = {"configured": False}
        return value
    manifest = _file_input_manifest(field, value)
    seals[field] = {"configured": True, "manifest": manifest}
    return manifest


def bind_simulation_file_input_owners(
    simulation: Any, fields: tuple[str, ...]
) -> None:
    """Bind configured file seals to owners immediately after native load."""
    with _LOCK:
        if type(fields) is not tuple or any(
            type(field) is not str or field not in _SIMULATION_FILE_INPUT_FIELDS
            for field in fields
        ):
            raise BackendRecommendationError(
                "file-input owner binding received invalid fields"
            )
        seals = _FILE_INPUT_SEALS_BY_SIMULATION.get(simulation)
        if not _is_exact_trusted_meep_type(
            simulation, "meep.simulation", "Simulation"
        ):
            if not seals or not any(
                sealed.get("configured") is True
                for sealed in seals.values()
            ):
                return
            raise BackendRecommendationError(
                "file-input owner binding requires an exact Meep Simulation"
            )
        for field in fields:
            sealed = seals.get(field) if seals else None
            if sealed is None or sealed.get("configured") is not True:
                continue
            if not _file_input_is_materialized(simulation, field):
                raise BackendRecommendationError(
                    f"simulation file input {field!r} was not materialized"
                )
            owner = (
                getattr(simulation, "fields", None)
                if field == "load_fields_file"
                else getattr(simulation, "structure", None)
            )
            if owner is None:
                raise BackendRecommendationError(
                    f"simulation file input {field!r} has no native owner"
                )
            current_value = getattr(simulation, field, None)
            current_manifest = _file_input_manifest(field, current_value)
            if current_manifest != sealed.get("manifest"):
                raise BackendRecommendationError(
                    f"simulation file input {field!r} changed after its "
                    "pre-initialization seal and before owner binding"
                )
            sealed_owner = sealed.get("_materialized_owner")
            if sealed_owner is None:
                sealed["_materialized_owner"] = owner
            elif sealed_owner is not owner:
                raise BackendRecommendationError(
                    f"simulation file input {field!r} changed native owner "
                    "during materialization"
                )


def clear_simulation_file_input_seals(simulation: Any) -> None:
    """Forget pre-initialization seals when Simulation.reset_meep is called."""
    with _LOCK:
        _FILE_INPUT_SEALS_BY_SIMULATION.pop(simulation, None)


def seal_simulation_file_inputs(mp: Any, simulation: Any) -> None:
    """Collectively seal exact file inputs before native initialization."""
    with _LOCK:
        try:
            ranks = int(mp.count_processors())
            local_rank = int(mp.my_rank())
            if ranks <= 0 or not 0 <= local_rank < ranks:
                raise BackendRecommendationError(
                    "file-input sealing received an invalid MPI rank inventory"
                )
            exact_simulation = _is_exact_trusted_meep_type(
                simulation, "meep.simulation", "Simulation"
            )
            if not exact_simulation:
                # Upstream utilities legitimately derive lightweight test and
                # instrumentation simulations.  They need no file seal when
                # all three native file inputs are unconfigured.  Read the
                # real instance dictionary directly and reject custom lookup
                # hooks so a subclass cannot hide a configured native input.
                try:
                    attributes = object.__getattribute__(
                        simulation, "__dict__"
                    )
                    getattribute = inspect.getattr_static(
                        type(simulation), "__getattribute__"
                    )
                    getattr_fallback = inspect.getattr_static(
                        type(simulation), "__getattr__", None
                    )
                except Exception as error:
                    raise BackendRecommendationError(
                        "file-input sealing requires an exact Meep Simulation"
                    ) from error
                if (
                    type(attributes) is not dict
                    or getattribute is not object.__getattribute__
                    or getattr_fallback is not None
                ):
                    raise BackendRecommendationError(
                        "file-input sealing requires an exact Meep Simulation"
                    )
                trusted_base = None
                for base in type(simulation).__mro__:
                    if (
                        type.__getattribute__(base, "__module__")
                        == "meep.simulation"
                        and type.__getattribute__(base, "__qualname__")
                        == "Simulation"
                        and _is_trusted_meep_class(base)
                    ):
                        trusted_base = base
                        break
                    namespace = type.__getattribute__(base, "__dict__")
                    if any(
                        name in namespace
                        for name in (
                            "__dict__",
                            "__getattribute__",
                            "__getattr__",
                            *_SIMULATION_FILE_INPUT_FIELDS,
                        )
                    ):
                        raise BackendRecommendationError(
                            "file-input sealing requires an exact Meep "
                            "Simulation"
                        )
                if trusted_base is None:
                    raise BackendRecommendationError(
                        "file-input sealing requires an exact Meep Simulation"
                    )
                for field in _SIMULATION_FILE_INPUT_FIELDS:
                    value = dict.get(attributes, field)
                    if value is not None and (
                        type(value) is not str or bool(value)
                    ):
                        raise BackendRecommendationError(
                            "configured file inputs require an exact Meep "
                            "Simulation"
                        )
                common = {"unconfigured_subclass_file_inputs": True}
            else:
                single_parallel_file = getattr(
                    simulation, "load_single_parallel_file", True
                )
                if type(single_parallel_file) is not bool:
                    raise BackendRecommendationError(
                        "load_single_parallel_file must be an exact boolean"
                    )
                common = {
                    "load_single_parallel_file": single_parallel_file,
                }
                for field in sorted(_SIMULATION_FILE_INPUT_FIELDS):
                    value = getattr(simulation, field, None)
                    bound = _sealed_file_input_manifest(
                        simulation, field, value
                    )
                    if (
                        field == "epsilon_input_file"
                        or single_parallel_file
                    ):
                        common[field] = bound
                    else:
                        common[field] = {
                            "rank_local": True,
                            "configured": bool(bound),
                        }
            local = {
                "ok": True,
                "active_size": ranks,
                "rank": local_rank,
                "common_digest": _sha256_json(common),
            }
        except Exception as error:
            local = _preflight_failure_record(error)
        records = _collective_records(mp, local, verify_count=False)
        failures = [
            (rank, record)
            for rank, record in enumerate(records)
            if record.get("ok") is not True
        ]
        if failures:
            detail = "; ".join(
                f"rank {rank}: {record.get('error', 'file seal failure')}"
                for rank, record in failures
            )
            raise BackendRecommendationError(
                f"distributed gpmeep file-input sealing failed: {detail}"
            )
        declared_sizes = {record.get("active_size") for record in records}
        common_digests = {record.get("common_digest") for record in records}
        if (
            len(declared_sizes) != 1
            or next(iter(declared_sizes)) != len(records)
            or len(common_digests) != 1
        ):
            raise BackendRecommendationError(
                "distributed gpmeep file-input seals differ across MPI ranks"
            )


def _original_trusted_descriptor(value_type: type, name: str) -> Any | None:
    missing = object()
    try:
        installed = inspect.getattr_static(value_type, name, missing)
    except Exception:
        return None
    for owner in value_type.__mro__:
        original = _TRUSTED_MEEP_CLASS_MEMBERS.get(owner, {}).get(name, missing)
        if original is not missing:
            return original if installed is original else None
    return None


def _validated_native_chunk_layout_manifest(
    structure: Any, *, require_copy_accessors: bool
) -> dict[str, Any]:
    """Return an authenticated, pointer-free native chunk layout."""
    if not _is_exact_trusted_meep_type(structure, "meep", "structure"):
        raise BackendRecommendationError(
            "Simulation chunk-layout source has no exact native structure"
        )
    required_methods = ["gpmeep_chunk_layout_manifest"]
    if require_copy_accessors:
        required_methods.extend(("get_chunk_volumes", "get_chunk_owners"))
    structure_attributes = getattr(structure, "__dict__", None)
    if not isinstance(structure_attributes, dict) or any(
        name in structure_attributes for name in required_methods
    ):
        raise BackendRecommendationError(
            "Simulation chunk-layout native accessors are shadowed"
        )
    original_methods = {
        name: _original_trusted_descriptor(type(structure), name)
        for name in required_methods
    }
    if any(method is None for method in original_methods.values()):
        raise BackendRecommendationError(
            "Simulation chunk-layout native accessors differ from import-time code"
        )
    try:
        manifest_descriptor = original_methods["gpmeep_chunk_layout_manifest"]
        manifest_method = manifest_descriptor.__get__(structure, type(structure))
        manifest = json.loads(manifest_method())
    except Exception as error:
        raise BackendRecommendationError(
            "Simulation chunk layout cannot be canonicalized exactly"
        ) from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "num_chunks", "chunks"}
        or manifest.get("schema") != "gpmeep-chunk-layout-v1"
        or type(manifest.get("num_chunks")) is not int
        or manifest["num_chunks"] <= 0
        or not isinstance(manifest.get("chunks"), list)
        or len(manifest["chunks"]) != manifest["num_chunks"]
    ):
        raise BackendRecommendationError(
            "Simulation chunk-layout manifest is malformed"
        )
    for index, chunk in enumerate(manifest["chunks"]):
        if (
            not isinstance(chunk, dict)
            or set(chunk)
            != {
                "index",
                "owner",
                "dimension",
                "a",
                "inva",
                "directions",
            }
            or type(chunk.get("index")) is not int
            or chunk["index"] != index
            or type(chunk.get("owner")) is not int
            or chunk["owner"] < 0
            or type(chunk.get("dimension")) is not int
            or type(chunk.get("a")) is not str
            or type(chunk.get("inva")) is not str
            or not isinstance(chunk.get("directions"), list)
            or len(chunk["directions"]) != 3
        ):
            raise BackendRecommendationError(
                "Simulation chunk-layout manifest is malformed"
            )
        try:
            scale = float.fromhex(chunk["a"])
            inverse_scale = float.fromhex(chunk["inva"])
        except ValueError as error:
            raise BackendRecommendationError(
                "Simulation chunk-layout manifest is malformed"
            ) from error
        if (
            not math.isfinite(scale)
            or scale <= 0
            or not math.isfinite(inverse_scale)
            or inverse_scale <= 0
        ):
            raise BackendRecommendationError(
                "Simulation chunk-layout manifest is malformed"
            )
        directions: set[int] = set()
        for slot, direction in enumerate(chunk["directions"]):
            if (
                not isinstance(direction, dict)
                or set(direction)
                != {"slot", "direction", "little", "big", "num"}
                or type(direction.get("slot")) is not int
                or direction["slot"] != slot
                or type(direction.get("direction")) is not int
                or direction["direction"] in directions
                or type(direction.get("little")) is not int
                or type(direction.get("big")) is not int
                or type(direction.get("num")) is not int
                or direction["num"] <= 0
                or direction["big"] < direction["little"]
            ):
                raise BackendRecommendationError(
                    "Simulation chunk-layout manifest is malformed"
                )
            directions.add(direction["direction"])
    return manifest


def _validated_chunk_layout_manifest(value: Any) -> dict[str, Any]:
    """Return the exact native layout a source Simulation will copy."""
    structure = getattr(value, "structure", None)
    if structure is None:
        raise BackendRecommendationError(
            "Simulation chunk-layout source must be initialized"
        )
    return _validated_native_chunk_layout_manifest(
        structure, require_copy_accessors=True
    )


def _chunk_layout_input_manifest(field: str, value: Any) -> Any:
    if type(value) is str:
        return _file_input_manifest(field, value)
    if isinstance(value, str):
        raise BackendRecommendationError(
            f"Simulation chunk-layout input {field!r} must be an exact string"
        )
    if _is_exact_trusted_meep_type(
        value, "meep.simulation", "Simulation"
    ):
        return {
            "simulation_chunk_layout": _validated_chunk_layout_manifest(value)
        }
    return value


def _native_proxy_python_state(owner: Any, simulation: Any) -> Any:
    attributes = getattr(owner, "__dict__", None)
    if not isinstance(attributes, dict):
        raise BackendRecommendationError(
            "initialized Meep native proxy has no exact Python state"
        )
    python_state = {
        key: item for key, item in attributes.items() if key != "this"
    }
    traversal = _CanonicalTraversal()
    traversal.native_simulation_roots.add(id(simulation))
    traversal.object_references[id(owner)] = 0
    traversal.object_values[id(owner)] = owner
    return {
        "owner_reference": 0,
        "attributes": _canonical(python_state, traversal),
    }


def _native_state_manifest(
    simulation: Any,
    owner: Any,
    *,
    module: str,
    qualname: str,
) -> Any:
    """Call an import-time authenticated SWIG state-manifest descriptor."""
    if not _is_exact_trusted_meep_type(
        simulation, "meep.simulation", "Simulation"
    ):
        return owner.gpmeep_state_manifest()
    if not _is_exact_trusted_meep_type(owner, module, qualname):
        raise BackendRecommendationError(
            f"initialized Meep {qualname} owner is not the exact native proxy"
        )
    attributes = getattr(owner, "__dict__", None)
    if (
        not isinstance(attributes, dict)
        or "this" not in attributes
        or "gpmeep_state_manifest" in attributes
    ):
        raise BackendRecommendationError(
            f"initialized Meep {qualname} state accessor is shadowed"
        )
    descriptor = _original_trusted_descriptor(
        type(owner), "gpmeep_state_manifest"
    )
    if descriptor is None:
        raise BackendRecommendationError(
            f"initialized Meep {qualname} state accessor differs from "
            "import-time code"
        )
    try:
        return descriptor.__get__(owner, type(owner))()
    except Exception as error:
        raise BackendRecommendationError(
            f"initialized Meep {qualname} state accessor failed"
        ) from error


def _native_fields_state(simulation: Any) -> dict[str, Any]:
    fields = getattr(simulation, "fields", None)
    if fields is None:
        return {
            "schema": "gpmeep-fields-state-v2",
            "uninitialized": True,
            "external_source_count": 0,
            "external_source_states": [],
        }
    try:
        native_state = json.loads(
            _native_state_manifest(
                simulation, fields, module="meep", qualname="fields"
            )
        )
    except Exception as error:
        raise BackendRecommendationError(
            "initialized Meep fields state cannot be canonicalized exactly"
        ) from error
    if (
        not isinstance(native_state, dict)
        or native_state.get("schema") != "gpmeep-fields-state-v2"
        or type(native_state.get("epoch")) is not int
        or native_state["epoch"] < 0
        or type(native_state.get("external_source_count")) is not int
        or native_state["external_source_count"] < 0
    ):
        raise BackendRecommendationError(
            "initialized Meep fields state manifest is malformed"
        )
    external_count = native_state["external_source_count"]
    source_records = native_state.get("sources")
    if (
        not isinstance(source_records, list)
        or any(
            not isinstance(record, dict)
            or record.get("cycle") is True
            or type(record.get("has_external_state")) is not bool
            for record in source_records
        )
        or sum(
            1 for record in source_records if record["has_external_state"]
        )
        != external_count
    ):
        raise BackendRecommendationError(
            "initialized Meep native source list is malformed"
        )
    if external_count:
        try:
            if _NATIVE_EXTERNAL_SOURCE_COLLECTOR is None:
                raise RuntimeError(
                    "trusted native source-state collector is unavailable"
                )
            external_states = _NATIVE_EXTERNAL_SOURCE_COLLECTOR(fields)
        except Exception as error:
            raise BackendRecommendationError(
                "initialized Python-backed native source state cannot be "
                "canonicalized exactly"
            ) from error
        if (
            not isinstance(external_states, list)
            or len(external_states) != external_count
            or not all(callable(value) for value in external_states)
        ):
            raise BackendRecommendationError(
                "initialized Python-backed native source inventory is malformed"
            )
        traversal = _CanonicalTraversal()
        native_state["external_source_states"] = _canonical(
            external_states, traversal
        )
    else:
        native_state["external_source_states"] = []
    native_state["python_state"] = _native_proxy_python_state(
        fields, simulation
    )
    return native_state


def _native_structure_state(simulation: Any) -> dict[str, Any]:
    structure = getattr(simulation, "structure", None)
    if structure is None:
        return {"schema": "gpmeep-structure-state-v1", "uninitialized": True}
    try:
        native_state = json.loads(
            _native_state_manifest(
                simulation,
                structure,
                module="meep",
                qualname="structure",
            )
        )
    except Exception as error:
        raise BackendRecommendationError(
            "initialized Meep structure state cannot be canonicalized exactly"
        ) from error
    if (
        not isinstance(native_state, dict)
        or native_state.get("schema") != "gpmeep-structure-state-v1"
        or type(native_state.get("epoch")) is not int
        or native_state["epoch"] < 0
    ):
        raise BackendRecommendationError(
            "initialized Meep structure state manifest is malformed"
        )
    native_state["python_state"] = _native_proxy_python_state(
        structure, simulation
    )
    return native_state


def _is_rank_local_indexed_source(value: Any) -> bool:
    return _is_exact_trusted_meep_type(
        value, "meep.source", "IndexedSource"
    )


def _shared_simulation_sources(value: Any) -> Any:
    """Remove only trusted rank-partitioned IndexedSource state.

    Adjoint construction creates one ``IndexedSource`` per rank-local source
    chunk, so its count, native sourcedata and amplitude array legitimately
    differ across MPI ranks.  The complete local objects are bound separately
    by ``_python_rank_local_state`` and then digested in ordered rank order.
    Every other source remains in the common workload fingerprint and must be
    identical on every rank.
    """

    if type(value) not in (list, tuple):
        return value
    return {
        "schema": "gpmeep-shared-simulation-sources-v1",
        "container": type(value).__name__,
        "uniform_sources": [
            source
            for source in value
            if not _is_rank_local_indexed_source(source)
        ],
    }


def _python_rank_local_state(simulation: Any) -> dict[str, Any]:
    sources = getattr(simulation, "sources", None)
    indexed_sources: list[dict[str, Any]] = []
    if type(sources) in (list, tuple):
        indexed_sources = [
            {"position": index, "source": source}
            for index, source in enumerate(sources)
            if _is_rank_local_indexed_source(source)
        ]
    traversal = _CanonicalTraversal(
        memoize_trusted_meep_dependency_graph=True
    )
    file_inputs: dict[str, Any] = {}
    if getattr(simulation, "load_single_parallel_file", True) is False:
        for field in sorted(_SIMULATION_RANK_LOCAL_LOAD_FIELDS):
            value = getattr(simulation, field, None)
            bound = _sealed_file_input_manifest(simulation, field, value)
            if bound not in (None, ""):
                file_inputs[field] = bound
    result = {
        "schema": "gpmeep-python-rank-local-state-v2",
        "indexed_sources": _canonical(indexed_sources, traversal),
        "file_inputs": file_inputs,
    }
    _finalize_trusted_meep_dependency_integrity(
        traversal, "Python rank-local state"
    )
    return result


def _native_simulation_state(simulation: Any) -> dict[str, Any]:
    return {
        "schema": "gpmeep-native-simulation-state-v2",
        "fields": _native_fields_state(simulation),
        "structure": _native_structure_state(simulation),
        "python_rank_local": _python_rank_local_state(simulation),
    }


def _simulation_workload_state(
    simulation: Any, *, include_native_state: bool = True
) -> dict[str, Any]:
    attributes = getattr(simulation, "__dict__", None)
    if not isinstance(attributes, dict):
        raise BackendRecommendationError(
            "Simulation workload state cannot be canonicalized exactly"
        )
    exact_meep_simulation = _is_exact_trusted_meep_type(
        simulation, "meep.simulation", "Simulation"
    )
    materialized_structure = (
        getattr(simulation, "structure", None)
        if exact_meep_simulation
        else None
    )
    materialized_chunk_layout = materialized_structure is not None
    state: dict[str, Any] = {}
    for key, value in sorted(attributes.items()):
        if key.startswith("_gpmeep_"):
            continue
        if exact_meep_simulation and key in _SIMULATION_NATIVE_RUNTIME_FIELDS:
            continue
        if exact_meep_simulation and key == "sources":
            state[key] = _shared_simulation_sources(value)
        else:
            if exact_meep_simulation and key in _SIMULATION_CHUNK_LAYOUT_FIELDS:
                state[key] = (
                    {"materialized_chunk_layout": key}
                    if materialized_chunk_layout
                    else _chunk_layout_input_manifest(key, value)
                )
            elif (
                exact_meep_simulation
                and key in _SIMULATION_RANK_LOCAL_LOAD_FIELDS
                and getattr(
                    simulation, "load_single_parallel_file", True
                )
                is False
            ):
                # Meep's ``single_parallel_file=False`` route assigns one
                # different rank shard to every process.  Keep only the
                # common configured/not-configured shape here; the complete
                # path, dataset, byte length and digest are bound in ordered
                # rank order by ``_python_rank_local_state``.
                bound = _sealed_file_input_manifest(simulation, key, value)
                if (
                    isinstance(bound, dict)
                    and "materialized_file_input" in bound
                ):
                    configured = bool(
                        bound["materialized_file_input"]["configured"]
                    )
                else:
                    configured = bool(bound)
                state[key] = {
                    "rank_local_file_input": key,
                    "configured": configured,
                }
            elif key in _SIMULATION_FILE_INPUT_FIELDS:
                state[key] = _sealed_file_input_manifest(
                    simulation, key, value
                )
            else:
                state[key] = value
    if materialized_chunk_layout:
        state["_gpmeep_materialized_chunk_layout"] = (
            _validated_native_chunk_layout_manifest(
                materialized_structure,
                require_copy_accessors=False,
            )
        )
    if exact_meep_simulation and include_native_state:
        state["_gpmeep_native_state"] = _native_simulation_state(simulation)
    return state


def _float_token(value: float) -> dict[str, str]:
    return {"float64_be": struct.pack(">d", value).hex()}


def _numpy_dtype_token(
    np: Any,
    dtype: Any,
    seen: set[int],
    active: set[int] | None = None,
) -> dict[str, Any]:
    """Return an exact semantic token for a NumPy dtype."""
    if (
        _ORIGINAL_NUMPY_DTYPE_TYPE is None
        or not isinstance(dtype, _ORIGINAL_NUMPY_DTYPE_TYPE)
        or type(dtype) not in _ORIGINAL_NUMPY_BUILTIN_DTYPE_TYPES
    ):
        raise BackendRecommendationError("NumPy dtype is malformed")
    if active is None:
        active = set()
    identity = id(dtype)
    if identity in active:
        raise BackendRecommendationError(
            "recursive NumPy dtype cannot be canonicalized exactly"
        )
    active.add(identity)
    try:
        fields = []
        for name in dtype.names or ():
            specification = dtype.fields[name]
            fields.append(
                {
                    "name": name,
                    "dtype": _numpy_dtype_token(
                        np, specification[0], seen, active
                    ),
                    "offset": int(specification[1]),
                    "title": (
                        _canonical(specification[2], seen)
                        if len(specification) > 2
                        else None
                    ),
                }
            )
        subdtype = None
        if dtype.subdtype is not None:
            subdtype = {
                "base": _numpy_dtype_token(
                    np, dtype.subdtype[0], seen, active
                ),
                "shape": [int(item) for item in dtype.subdtype[1]],
            }
        metadata = dtype.metadata
        return {
            "str": dtype.str,
            "name": dtype.name,
            "char": dtype.char,
            "kind": dtype.kind,
            "byteorder": dtype.byteorder,
            "itemsize": int(dtype.itemsize),
            "alignment": int(dtype.alignment),
            "hasobject": bool(dtype.hasobject),
            "isbuiltin": int(dtype.isbuiltin),
            "isnative": bool(dtype.isnative),
            "isalignedstruct": bool(dtype.isalignedstruct),
            "fields": fields,
            "subdtype": subdtype,
            "metadata": (
                None
                if metadata is None
                else _canonical(dict(metadata), seen)
            ),
        }
    finally:
        active.remove(identity)


def _numpy_array_token(np: Any, value: Any, seen: set[int]) -> dict[str, Any]:
    """Bind array values, layout, mutability, dtype, and view topology."""
    if value.dtype.hasobject:
        payload = {
            "values": _canonical(
                _ORIGINAL_NUMPY_NDARRAY_TOLIST(value), seen
            )
        }
    else:
        payload = {
            "sha256": hashlib.sha256(
                _ORIGINAL_NUMPY_NDARRAY_TOBYTES(value, order="C")
            ).hexdigest()
        }
    base = value.base
    if base is None:
        base_token = None
        byte_offset = 0
    else:
        if not isinstance(base, _ORIGINAL_NUMPY_NDARRAY_TYPE):
            raise BackendRecommendationError(
                "NumPy arrays backed by non-ndarray exporters cannot be "
                "canonicalized exactly"
            )
        try:
            data_pointer = int(value.__array_interface__["data"][0])
            base_pointer = int(base.__array_interface__["data"][0])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise BackendRecommendationError(
                "NumPy array view offset cannot be canonicalized exactly"
            ) from error
        byte_offset = data_pointer - base_pointer
        base_token = _canonical(base, seen)
    flags = value.flags
    return {
        "dtype": _numpy_dtype_token(np, value.dtype, seen),
        "shape": [int(item) for item in value.shape],
        "strides": (
            None
            if value.strides is None
            else [int(item) for item in value.strides]
        ),
        "flags": {
            "c_contiguous": bool(flags.c_contiguous),
            "f_contiguous": bool(flags.f_contiguous),
            "owndata": bool(flags.owndata),
            "writeable": bool(flags.writeable),
            "aligned": bool(flags.aligned),
            "writebackifcopy": bool(flags.writebackifcopy),
        },
        "byte_offset": byte_offset,
        "base": base_token,
        **payload,
    }


def _numpy_pcg64_state(value: Any, seen: set[int]) -> dict[str, Any]:
    """Read and validate complete public PCG64 state without object hooks."""

    try:
        state = _ORIGINAL_NUMPY_PCG64_STATE_DESCRIPTOR.__get__(
            value, _ORIGINAL_NUMPY_PCG64_TYPE
        )
        repeated_state = _ORIGINAL_NUMPY_PCG64_STATE_DESCRIPTOR.__get__(
            value, _ORIGINAL_NUMPY_PCG64_TYPE
        )
    except Exception as error:
        raise BackendRecommendationError(
            "NumPy PCG64 state cannot be canonicalized exactly"
        ) from error
    if (
        state != repeated_state
        or type(state) is not dict
        or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"}
        or state.get("bit_generator") != "PCG64"
        or type(state.get("state")) is not dict
        or set(state["state"]) != {"state", "inc"}
        or any(
            type(state["state"].get(name)) is not int
            or not 0 <= state["state"][name] < 2**128
            for name in ("state", "inc")
        )
        or state["state"]["inc"] % 2 != 1
        or type(state.get("has_uint32")) is not int
        or state["has_uint32"] not in {0, 1}
        or type(state.get("uinteger")) is not int
        or not 0 <= state["uinteger"] < 2**32
    ):
        raise BackendRecommendationError("NumPy PCG64 public state is malformed")
    canonical_state = {
        "bit_generator": state["bit_generator"],
        "state": {
            "state": state["state"]["state"],
            "inc": state["state"]["inc"],
        },
        "has_uint32": state["has_uint32"],
        "uinteger": state["uinteger"],
    }
    return _canonical(canonical_state, seen)


def _numpy_seed_sequence_state(value: Any, seen: set[int]) -> dict[str, Any]:
    """Read the exact public state that controls SeedSequence spawning."""

    try:
        state = _ORIGINAL_NUMPY_SEED_SEQUENCE_STATE_DESCRIPTOR.__get__(
            value, _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
        )
        repeated_state = (
            _ORIGINAL_NUMPY_SEED_SEQUENCE_STATE_DESCRIPTOR.__get__(
                value, _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
            )
        )
    except Exception as error:
        raise BackendRecommendationError(
            "NumPy SeedSequence state cannot be canonicalized exactly"
        ) from error

    def projection(item: Any, traversal: set[int]) -> dict[str, Any]:
        if (
            type(item) is not dict
            or set(item)
            != {"entropy", "spawn_key", "pool_size", "n_children_spawned"}
            or type(item.get("spawn_key")) is not tuple
            or any(
                type(component) is not int or component < 0
                for component in item["spawn_key"]
            )
            or type(item.get("pool_size")) is not int
            or item["pool_size"] < 4
            or type(item.get("n_children_spawned")) is not int
            or item["n_children_spawned"] < 0
        ):
            raise BackendRecommendationError(
                "NumPy SeedSequence public state is malformed"
            )
        return {
            "entropy": _canonical(item["entropy"], traversal),
            "spawn_key": list(item["spawn_key"]),
            "pool_size": item["pool_size"],
            "n_children_spawned": item["n_children_spawned"],
        }

    first = projection(state, _CanonicalTraversal())
    second = projection(repeated_state, _CanonicalTraversal())
    if first != second:
        raise BackendRecommendationError(
            "NumPy SeedSequence public state changed while inspected"
        )
    return projection(state, seen)


def _numpy_seed_sequence_pool(value: Any, seen: set[int]) -> Any:
    """Bind SeedSequence's mutable mixing pool with stable repeated reads."""

    try:
        import numpy as np

        pool = _ORIGINAL_NUMPY_SEED_SEQUENCE_POOL_DESCRIPTOR.__get__(
            value, _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
        )
        repeated_pool = _ORIGINAL_NUMPY_SEED_SEQUENCE_POOL_DESCRIPTOR.__get__(
            value, _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
        )
        state = _ORIGINAL_NUMPY_SEED_SEQUENCE_STATE_DESCRIPTOR.__get__(
            value, _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
        )
    except Exception as error:
        raise BackendRecommendationError(
            "NumPy SeedSequence pool cannot be canonicalized exactly"
        ) from error
    if (
        pool is not repeated_pool
        or type(pool) is not _ORIGINAL_NUMPY_NDARRAY_TYPE
        or type(state) is not dict
        or type(state.get("pool_size")) is not int
        or pool.ndim != 1
        or pool.shape != (state["pool_size"],)
        or pool.dtype.str != "<u4"
        or not pool.flags.c_contiguous
        or not pool.flags.owndata
    ):
        raise BackendRecommendationError("NumPy SeedSequence pool is malformed")

    first = _numpy_array_token(np, pool, _CanonicalTraversal())
    second = _numpy_array_token(np, repeated_pool, _CanonicalTraversal())
    if first != second:
        raise BackendRecommendationError(
            "NumPy SeedSequence pool changed while inspected"
        )
    token = _canonical(pool, seen)
    third = _numpy_array_token(np, pool, _CanonicalTraversal())
    if first != third:
        raise BackendRecommendationError(
            "NumPy SeedSequence pool changed while inspected"
        )
    return token


def _numpy_pcg64_seed_sequence(value: Any, seen: set[int]) -> Any:
    try:
        seed_sequence = (
            _ORIGINAL_NUMPY_BIT_GENERATOR_SEED_SEQUENCE_DESCRIPTOR.__get__(
                value, _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE
            )
        )
        repeated_seed_sequence = (
            _ORIGINAL_NUMPY_BIT_GENERATOR_SEED_SEQUENCE_DESCRIPTOR.__get__(
                value, _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE
            )
        )
    except Exception as error:
        raise BackendRecommendationError(
            "NumPy PCG64 SeedSequence cannot be canonicalized exactly"
        ) from error
    if (
        seed_sequence is not repeated_seed_sequence
        or type(seed_sequence) is not _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
    ):
        raise BackendRecommendationError(
            "NumPy PCG64 requires an exact stable SeedSequence"
        )
    return _canonical(seed_sequence, seen)


def _numpy_generate_state_surface_is_original() -> bool:
    function = _ORIGINAL_NUMPY_SEED_SEQUENCE_GENERATE_STATE
    integrity = _ORIGINAL_NUMPY_SEED_SEQUENCE_GENERATE_STATE_INTEGRITY
    if not isinstance(function, types.FunctionType) or not isinstance(
        integrity, dict
    ):
        return False
    closure = function.__closure__
    if (
        function.__code__ is not integrity.get("code")
        or function.__defaults__ is not integrity.get("defaults")
        or function.__kwdefaults__ is not integrity.get("kwdefaults")
        or closure is not integrity.get("closure")
        or type(closure) is not tuple
        or len(closure) != len(integrity.get("closure_cells", ()))
    ):
        return False
    try:
        for cell, expected_cell, expected_value in zip(
            closure,
            integrity["closure_cells"],
            integrity["closure_values"],
            strict=True,
        ):
            if cell is not expected_cell or cell.cell_contents is not expected_value:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    globals_namespace = function.__globals__
    if not isinstance(globals_namespace, dict):
        return False
    for name, expected in integrity.get("globals", ()):
        if globals_namespace.get(name) is not expected:
            return False
    owner = integrity.get("errstate_owner")
    if type(owner) is not integrity.get("errstate_owner_type"):
        return False
    try:
        if any(
            getattr(owner, name) is not expected
            for name, expected in integrity.get("errstate_owner_state", ())
        ):
            return False
    except AttributeError:
        return False
    target = integrity.get("call_target")
    target_namespace = getattr(target, "__dict__", None)
    expected_namespace = integrity.get("call_target_namespace")
    if (
        type(target) is not integrity.get("call_target_type")
        or getattr(target, "__code__", None)
        is not integrity.get("call_target_code")
        or getattr(target, "__defaults__", None)
        is not integrity.get("call_target_defaults")
        or getattr(target, "__kwdefaults__", None)
        is not integrity.get("call_target_kwdefaults")
        or getattr(target, "__closure__", None)
        is not integrity.get("call_target_closure")
        or not isinstance(target_namespace, dict)
        or set(target_namespace) != {name for name, _ in expected_namespace}
        or any(
            target_namespace.get(name) is not expected
            for name, expected in expected_namespace
        )
    ):
        return False
    return True


def _numpy_rng_token(np: Any, value: Any, seen: set[int]) -> dict[str, Any]:
    """Bind an exact Generator/PCG64 class surface and its complete state."""

    numpy_namespace = getattr(np, "__dict__", None)
    random_namespace = getattr(_ORIGINAL_NUMPY_RANDOM_MODULE, "__dict__", None)
    generator_namespace = type.__getattribute__(
        _ORIGINAL_NUMPY_GENERATOR_TYPE, "__dict__"
    )
    bit_generator_namespace = type.__getattribute__(
        _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE, "__dict__"
    )
    pcg64_namespace = type.__getattribute__(_ORIGINAL_NUMPY_PCG64_TYPE, "__dict__")
    seed_sequence_namespace = type.__getattribute__(
        _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE, "__dict__"
    )
    if (
        not isinstance(numpy_namespace, dict)
        or numpy_namespace.get("random") is not _ORIGINAL_NUMPY_RANDOM_MODULE
        or not isinstance(random_namespace, dict)
        or random_namespace.get("Generator") is not _ORIGINAL_NUMPY_GENERATOR_TYPE
        or random_namespace.get("BitGenerator")
        is not _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE
        or random_namespace.get("PCG64") is not _ORIGINAL_NUMPY_PCG64_TYPE
        or random_namespace.get("SeedSequence")
        is not _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE
        or generator_namespace.get("bit_generator")
        is not _ORIGINAL_NUMPY_GENERATOR_BIT_GENERATOR_DESCRIPTOR
        or generator_namespace.get("standard_normal")
        is not _ORIGINAL_NUMPY_GENERATOR_STANDARD_NORMAL
        or bit_generator_namespace.get("seed_seq")
        is not _ORIGINAL_NUMPY_BIT_GENERATOR_SEED_SEQUENCE_DESCRIPTOR
        or bit_generator_namespace.get("spawn")
        is not _ORIGINAL_NUMPY_BIT_GENERATOR_SPAWN
        or pcg64_namespace.get("state")
        is not _ORIGINAL_NUMPY_PCG64_STATE_DESCRIPTOR
        or seed_sequence_namespace.get("state")
        is not _ORIGINAL_NUMPY_SEED_SEQUENCE_STATE_DESCRIPTOR
        or seed_sequence_namespace.get("pool")
        is not _ORIGINAL_NUMPY_SEED_SEQUENCE_POOL_DESCRIPTOR
        or seed_sequence_namespace.get("spawn")
        is not _ORIGINAL_NUMPY_SEED_SEQUENCE_SPAWN
        or seed_sequence_namespace.get("generate_state")
        is not _ORIGINAL_NUMPY_SEED_SEQUENCE_GENERATE_STATE
        or not _numpy_generate_state_surface_is_original()
        or type(np.__version__) is not str
        or np.__version__ != _ORIGINAL_NUMPY_RNG_VERSION
    ):
        raise BackendRecommendationError(
            "NumPy RNG surface differs from its import-time dependencies"
        )

    if type(value) is _ORIGINAL_NUMPY_GENERATOR_TYPE:
        try:
            bit_generator = (
                _ORIGINAL_NUMPY_GENERATOR_BIT_GENERATOR_DESCRIPTOR.__get__(
                    value, _ORIGINAL_NUMPY_GENERATOR_TYPE
                )
            )
        except Exception as error:
            raise BackendRecommendationError(
                "NumPy Generator bit generator cannot be canonicalized exactly"
            ) from error
        if type(bit_generator) is not _ORIGINAL_NUMPY_PCG64_TYPE:
            raise BackendRecommendationError(
                "only exact NumPy PCG64 generators can be canonicalized"
            )
        return {
            "numpy_generator": {
                "numpy_version": _ORIGINAL_NUMPY_RNG_VERSION,
                "class": "numpy.random.Generator",
                "bit_generator_class": "numpy.random.PCG64",
                "bit_generator": _canonical(bit_generator, seen),
            }
        }
    if type(value) is _ORIGINAL_NUMPY_PCG64_TYPE:
        return {
            "numpy_bit_generator": {
                "numpy_version": _ORIGINAL_NUMPY_RNG_VERSION,
                "class": "numpy.random.PCG64",
                "state": _numpy_pcg64_state(value, seen),
                "seed_sequence": _numpy_pcg64_seed_sequence(value, seen),
            }
        }
    if type(value) is _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE:
        return {
            "numpy_seed_sequence": {
                "numpy_version": _ORIGINAL_NUMPY_RNG_VERSION,
                "class": "numpy.random.SeedSequence",
                "state": _numpy_seed_sequence_state(value, seen),
                "pool": _numpy_seed_sequence_pool(value, seen),
            }
        }
    raise BackendRecommendationError(
        "unsupported NumPy Generator, BitGenerator, or SeedSequence type "
        "cannot be canonicalized"
    )


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(
        _canonical(value, _CanonicalTraversal()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _code_token(code: types.CodeType, seen: set[int]) -> dict[str, Any]:
    line_table = getattr(
        code, "co_linetable", getattr(code, "co_lnotab", b"")
    )
    exception_table = getattr(code, "co_exceptiontable", b"")
    return {
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "filename": code.co_filename,
        "name": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "firstlineno": code.co_firstlineno,
        "linetable_sha256": hashlib.sha256(line_table).hexdigest(),
        "exceptiontable_sha256": hashlib.sha256(
            exception_table
        ).hexdigest(),
        "constants": [
            _code_token(item, seen)
            if isinstance(item, types.CodeType)
            else _canonical(item, seen)
            for item in code.co_consts
        ],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _nested_code_objects(code: types.CodeType):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _nested_code_objects(constant)


def _immutable_disassembly_constants(value: Any) -> bool:
    value_type = type(value)
    if value_type in (type(None), bool, int, float, complex, str, bytes, type(Ellipsis)):
        return True
    if value_type in (tuple, frozenset):
        return all(_immutable_disassembly_constants(item) for item in value)
    if value_type is types.CodeType:
        return _immutable_disassembly_constants(value.co_consts)
    return False


def _code_instructions(code: types.CodeType, seen: set[int] | None = None):
    """Memoize static disassembly, never the values canonicalized from code."""
    if not isinstance(seen, _CanonicalTraversal):
        return dis.get_instructions(code)
    cached = seen.bytecode_instructions.get(id(code))
    if cached is not None and cached[0] is code:
        return cached[1]
    if not _immutable_disassembly_constants(code):
        # dis also computes argrepr.  Preserve every call (and its errors)
        # when custom/mutable constants could have observable repr behavior.
        return dis.get_instructions(code)
    instructions = tuple(dis.get_instructions(code))
    seen.bytecode_instructions[id(code)] = (code, instructions)
    return instructions


def _referenced_global_names(
    code: types.CodeType, seen: set[int] | None = None
) -> set[str]:
    return {
        str(instruction.argval)
        for nested in _nested_code_objects(code)
        for instruction in _code_instructions(nested, seen)
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    }


def _trusted_function_runtime_snapshot(
    function: types.FunctionType,
) -> dict[str, Any]:
    closure = function.__closure__ or ()
    return {
        "object": function,
        "code": function.__code__,
        "globals": function.__globals__,
        "builtins": function.__builtins__,
        "name": function.__name__,
        "module": function.__module__,
        "qualname": function.__qualname__,
        "doc": function.__doc__,
        "defaults_object": function.__defaults__,
        "defaults_integrity": _trusted_annotation_integrity_snapshot(
            function.__defaults__
        ),
        "kwdefaults_object": function.__kwdefaults__,
        "kwdefaults_integrity": _trusted_annotation_integrity_snapshot(
            function.__kwdefaults__
        ),
        "annotations_object": function.__annotations__,
        "annotations_integrity": _trusted_annotation_integrity_snapshot(
            function.__annotations__
        ),
        "attributes_object": function.__dict__,
        "attributes_integrity": _trusted_annotation_integrity_snapshot(
            function.__dict__
        ),
        "closure_cells": closure,
        "closure_integrity": tuple(
            _trusted_annotation_integrity_snapshot(cell.cell_contents)
            for cell in closure
        ),
    }


def _trusted_function_runtime_is_unchanged(
    function: types.FunctionType, snapshot: dict[str, Any]
) -> bool:
    try:
        closure = function.__closure__ or ()
        return (
            function is snapshot["object"]
            and function.__code__ is snapshot["code"]
            and function.__globals__ is snapshot["globals"]
            and function.__builtins__ is snapshot["builtins"]
            and function.__name__ == snapshot["name"]
            and function.__module__ == snapshot["module"]
            and function.__qualname__ == snapshot["qualname"]
            and function.__doc__ == snapshot["doc"]
            and function.__defaults__ is snapshot["defaults_object"]
            and _trusted_annotation_integrity_snapshot(function.__defaults__)
            == snapshot["defaults_integrity"]
            and function.__kwdefaults__ is snapshot["kwdefaults_object"]
            and _trusted_annotation_integrity_snapshot(function.__kwdefaults__)
            == snapshot["kwdefaults_integrity"]
            and function.__annotations__ is snapshot["annotations_object"]
            and _trusted_annotation_integrity_snapshot(function.__annotations__)
            == snapshot["annotations_integrity"]
            and function.__dict__ is snapshot["attributes_object"]
            and _trusted_annotation_integrity_snapshot(function.__dict__)
            == snapshot["attributes_integrity"]
            and len(closure) == len(snapshot["closure_cells"])
            and all(
                cell is original
                for cell, original in zip(
                    closure, snapshot["closure_cells"]
                )
            )
            and tuple(
                _trusted_annotation_integrity_snapshot(cell.cell_contents)
                for cell in closure
            )
            == snapshot["closure_integrity"]
        )
    except Exception:
        return False


def _trusted_dependency_binding_integrity(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return _trusted_annotation_integrity_snapshot(value)
    return (
        "identity",
        f"{type(value).__module__}.{type(value).__qualname__}",
        id(value),
    )


def _expected_class_slotnames(candidate: type) -> list[str]:
    """Recompute CPython copyreg's derived ``__slotnames__`` cache."""

    names: list[str] = []
    for owner in type.__getattribute__(candidate, "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        if "__slots__" not in namespace:
            continue
        slots = namespace["__slots__"]
        if type(slots) is str:
            slots = (slots,)
        if type(slots) not in (tuple, list):
            raise BackendRecommendationError(
                "class slots cannot be authenticated exactly"
            )
        for name in slots:
            if type(name) is not str:
                raise BackendRecommendationError(
                    "class slot name is not an exact string"
                )
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                stripped = type.__getattribute__(owner, "__name__").lstrip("_")
                names.append(f"_{stripped}{name}" if stripped else name)
            else:
                names.append(name)
    return names


def _class_members_match_dependency_snapshot(
    candidate: type,
    namespace: types.MappingProxyType,
    members: dict[str, Any],
) -> bool:
    """Allow only copyreg's authentic, lazily-added slot-name cache."""

    current_names = set(namespace)
    original_names = set(members)
    if "__slotnames__" in original_names:
        cache = namespace.get("__slotnames__")
        return (
            current_names == original_names
            and cache is members["__slotnames__"]
            and type(cache) is list
            and cache == _expected_class_slotnames(candidate)
            and all(
                namespace.get(name) is value
                for name, value in members.items()
                if name != "__slotnames__"
            )
        )

    added = current_names - original_names
    if current_names - {"__slotnames__"} != original_names:
        return False
    if added:
        cache = namespace.get("__slotnames__")
        if (
            added != {"__slotnames__"}
            or type(cache) is not list
            or cache != _expected_class_slotnames(candidate)
        ):
            return False
    return all(namespace.get(name) is value for name, value in members.items())


def _trusted_dependency_graph_snapshot(
    roots: tuple[Any, ...],
    *,
    traversal_filter: Any | None = None,
) -> dict[str, Any]:
    functions: dict[types.FunctionType, dict[str, Any]] = {}
    classes: dict[type, dict[str, Any]] = {}
    bindings: list[tuple[Any, ...]] = []
    maximum_nodes = 4096

    def check_budget() -> None:
        if len(functions) + len(classes) > maximum_nodes:
            raise BackendRecommendationError(
                "trusted standard-library dependency graph is too large"
            )

    def visit_value(value: Any, active: set[int]) -> None:
        value_type = type(value)
        if value_type is types.FunctionType:
            if traversal_filter is None or traversal_filter(value):
                visit_function(value, active)
        elif isinstance(value, type):
            flags = int(type.__getattribute__(value, "__flags__"))
            if (
                flags & _PY_TPFLAGS_HEAPTYPE
                and (traversal_filter is None or traversal_filter(value))
            ):
                visit_class(value, active)
        elif value_type in (tuple, list):
            identity = id(value)
            if identity in active:
                return
            active.add(identity)
            try:
                for item in value:
                    visit_value(item, active)
            finally:
                active.remove(identity)
        elif value_type is dict:
            identity = id(value)
            if identity in active:
                return
            active.add(identity)
            try:
                for key, item in dict.items(value):
                    visit_value(key, active)
                    visit_value(item, active)
            finally:
                active.remove(identity)

    def visit_function(
        function: types.FunctionType, active: set[int]
    ) -> None:
        if function in functions:
            return
        functions[function] = _trusted_function_runtime_snapshot(function)
        check_budget()
        namespace = function.__globals__
        builtin_namespace = function.__builtins__
        for name in sorted(_referenced_global_names(function.__code__)):
            if name in namespace:
                source = "global"
                dependency = namespace[name]
            elif type(builtin_namespace) is dict and name in builtin_namespace:
                source = "builtin-dict"
                dependency = builtin_namespace[name]
            elif builtin_namespace is builtins and hasattr(builtins, name):
                source = "builtin-module"
                dependency = getattr(builtins, name)
            else:
                continue
            bindings.append(
                (
                    "name",
                    function,
                    source,
                    name,
                    dependency,
                    _trusted_dependency_binding_integrity(dependency),
                )
            )
            visit_value(dependency, active)

        for nested in _nested_code_objects(function.__code__):
            instructions = list(dis.get_instructions(nested))
            for index, instruction in enumerate(instructions):
                if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
                    continue
                root_name = str(instruction.argval)
                root = namespace.get(root_name)
                if not isinstance(root, types.ModuleType):
                    continue
                value = root
                path: list[str] = []
                cursor = index + 1
                while cursor < len(instructions) and instructions[
                    cursor
                ].opname in ("LOAD_ATTR", "LOAD_METHOD"):
                    attribute = str(instructions[cursor].argval)
                    try:
                        value = getattr(value, attribute)
                    except Exception:
                        break
                    path.append(attribute)
                    if isinstance(value, types.MethodType):
                        bindings.append(
                            (
                                "module-bound-method",
                                function,
                                root_name,
                                tuple(path),
                                value.__func__,
                                value.__self__,
                            )
                        )
                        visit_value(value.__func__, active)
                        cursor += 1
                        continue
                    stable_module_dependency = (
                        _is_stable_module_attribute_dependency(value)
                        or type(value) in (
                            tuple,
                            list,
                            dict,
                            set,
                            frozenset,
                            bytearray,
                        )
                    )
                    if stable_module_dependency:
                        bindings.append(
                            (
                                "module-path",
                                function,
                                root_name,
                                tuple(path),
                                value,
                                _trusted_dependency_binding_integrity(value),
                            )
                        )
                    visit_value(value, active)
                    cursor += 1

    def visit_class(candidate: type, active: set[int]) -> None:
        if candidate in classes:
            return
        namespace = type.__getattribute__(candidate, "__dict__")
        classes[candidate] = {
            "name": type.__getattribute__(candidate, "__name__"),
            "module": type.__getattribute__(candidate, "__module__"),
            "qualname": type.__getattribute__(candidate, "__qualname__"),
            "doc": type.__getattribute__(candidate, "__doc__"),
            "bases": type.__getattribute__(candidate, "__bases__"),
            "mro": type.__getattribute__(candidate, "__mro__"),
            "members": dict(namespace),
        }
        check_budget()
        for base in type.__getattribute__(candidate, "__bases__"):
            visit_value(base, active)
        for member in namespace.values():
            for function in _trusted_member_functions(member):
                if traversal_filter is None or traversal_filter(function):
                    visit_function(function, active)
            if not _trusted_member_functions(member):
                visit_value(member, active)

    for root in roots:
        visit_value(root, set())
    edges: dict[Any, set[Any]] = {}
    for binding in bindings:
        owner = binding[1]
        dependency = binding[4]
        if type(dependency) is types.FunctionType or isinstance(
            dependency, type
        ):
            edges.setdefault(owner, set()).add(dependency)
    for candidate, state in classes.items():
        dependencies = edges.setdefault(candidate, set())
        for base in state["bases"]:
            if int(type.__getattribute__(base, "__flags__")) & _PY_TPFLAGS_HEAPTYPE:
                dependencies.add(base)
        for member in state["members"].values():
            dependencies.update(_trusted_member_functions(member))
            if isinstance(member, type):
                dependencies.add(member)
    return {
        "functions": functions,
        "classes": classes,
        "bindings": tuple(bindings),
        "edges": {key: frozenset(value) for key, value in edges.items()},
    }


def _extend_trusted_dependency_graph(
    snapshot: dict[str, Any],
    roots: tuple[Any, ...],
    *,
    traversal_filter: Any | None = None,
) -> dict[str, Any]:
    """Add lazy package roots without re-trusting existing graph nodes.

    A package class may be registered after startup by an optional module.
    Rebuilding the complete graph at that point would silently replace the
    import-time snapshots of existing classes and functions with their current
    state. Besides weakening mutation detection, that made legitimate
    validation instrumentation part of the trusted package identity. Extend
    only from the new roots and retain every pre-existing node, binding, and
    edge exactly as it was originally authenticated.
    """

    addition = _trusted_dependency_graph_snapshot(
        roots, traversal_filter=traversal_filter
    )
    known_nodes = frozenset(
        (*snapshot["functions"], *snapshot["classes"])
    )
    functions = dict(snapshot["functions"])
    classes = dict(snapshot["classes"])
    for function, state in addition["functions"].items():
        functions.setdefault(function, state)
    for candidate, state in addition["classes"].items():
        classes.setdefault(candidate, state)
    bindings = tuple(snapshot["bindings"]) + tuple(
        binding
        for binding in addition["bindings"]
        if binding[1] not in known_nodes
    )
    edges = dict(snapshot["edges"])
    for owner, dependencies in addition["edges"].items():
        if owner not in known_nodes:
            edges[owner] = dependencies
    return {
        "functions": functions,
        "classes": classes,
        "bindings": bindings,
        "edges": edges,
    }


def _trusted_dependency_graph_is_unchanged(
    snapshot: dict[str, Any], root: Any | None = None
) -> bool:
    try:
        reachable: set[Any] | None = None
        if root is not None:
            reachable = set()
            pending = [root]
            while pending:
                node = pending.pop()
                if node in reachable:
                    continue
                reachable.add(node)
                pending.extend(snapshot["edges"].get(node, ()))
        for function, runtime in snapshot["functions"].items():
            if reachable is not None and function not in reachable:
                continue
            if not _trusted_function_runtime_is_unchanged(function, runtime):
                return False
        for candidate, state in snapshot["classes"].items():
            if reachable is not None and candidate not in reachable:
                continue
            namespace = type.__getattribute__(candidate, "__dict__")
            members = state["members"]
            if (
                type.__getattribute__(candidate, "__name__") != state["name"]
                or type.__getattribute__(candidate, "__module__")
                != state["module"]
                or type.__getattribute__(candidate, "__qualname__")
                != state["qualname"]
                or type.__getattribute__(candidate, "__doc__") != state["doc"]
                or type.__getattribute__(candidate, "__bases__")
                != state["bases"]
                or type.__getattribute__(candidate, "__mro__") != state["mro"]
                or not _class_members_match_dependency_snapshot(
                    candidate, namespace, members
                )
            ):
                return False
        for binding in snapshot["bindings"]:
            kind, function = binding[:2]
            if reachable is not None and function not in reachable:
                continue
            if kind == "name":
                _, _, source, name, expected, integrity = binding
                if source == "global":
                    current = function.__globals__.get(name)
                elif source == "builtin-dict":
                    builtins_dict = function.__builtins__
                    if type(builtins_dict) is not dict:
                        return False
                    current = builtins_dict.get(name)
                else:
                    if function.__builtins__ is not builtins:
                        return False
                    current = getattr(builtins, name, None)
            elif kind == "module-path":
                _, _, root_name, path, expected, integrity = binding
                current = function.__globals__.get(root_name)
                for attribute in path:
                    current = getattr(current, attribute)
            else:
                (
                    _,
                    _,
                    root_name,
                    path,
                    expected_function,
                    expected_self,
                ) = binding
                current = function.__globals__.get(root_name)
                for attribute in path:
                    current = getattr(current, attribute)
                if (
                    not isinstance(current, types.MethodType)
                    or current.__func__ is not expected_function
                    or current.__self__ is not expected_self
                ):
                    return False
                continue
            if (
                current is not expected
                or _trusted_dependency_binding_integrity(current) != integrity
            ):
                return False
        return True
    except Exception:
        return False


def _trusted_dependency_node_is_unchanged(
    snapshot: dict[str, Any], node: Any
) -> bool:
    """Validate one node whose runtime state is read by its token.

    A complete graph check at a traversal boundary establishes the transitive
    package identity.  The much smaller node check is intentionally repeated
    around each token construction so a function/class cannot be changed,
    observed by the token builder, and restored before the traversal-level
    final check.  For a class, its direct member functions are checked by
    ``_trusted_dependency_token_scope_is_unchanged`` because their code is
    serialized as part of the class token. Global/module dependency bindings
    are not part of a shallow trusted token; the complete boundary checks
    authenticate them without adding irrelevant per-token work.
    """
    try:
        if type(node) is types.FunctionType and node in snapshot["functions"]:
            if not _trusted_function_runtime_is_unchanged(
                node, snapshot["functions"][node]
            ):
                return False
        elif isinstance(node, type) and node in snapshot["classes"]:
            state = snapshot["classes"][node]
            namespace = type.__getattribute__(node, "__dict__")
            if (
                type.__getattribute__(node, "__name__") != state["name"]
                or type.__getattribute__(node, "__module__")
                != state["module"]
                or type.__getattribute__(node, "__qualname__")
                != state["qualname"]
                or type.__getattribute__(node, "__doc__") != state["doc"]
                or type.__getattribute__(node, "__bases__")
                != state["bases"]
                or type.__getattribute__(node, "__mro__") != state["mro"]
                or not _class_members_match_dependency_snapshot(
                    node, namespace, state["members"]
                )
            ):
                return False
        else:
            return False

        return True
    except Exception:
        return False


def _trusted_dependency_token_scope_is_unchanged(
    snapshot: dict[str, Any], root: Any
) -> bool:
    """Validate state that can directly affect one trusted token."""
    if not _trusted_dependency_node_is_unchanged(snapshot, root):
        return False
    if root in snapshot["classes"]:
        for dependency in snapshot["edges"].get(root, ()):
            if (
                dependency in snapshot["functions"]
                and not _trusted_dependency_node_is_unchanged(
                    snapshot, dependency
                )
            ):
                return False
    return True


def _trusted_dependency_value_token(
    value: Any,
    seen: _CanonicalTraversal,
    references: dict[int, int] | None = None,
) -> Any:
    """Return a cross-rank token for reachable mutable stdlib table state."""
    if references is None:
        references = {}
    value_type = type(value)
    if value_type is _ORIGINAL_RE_NAMED_INT_CONSTANT_TYPE:
        return {
            "re_named_int_constant": {
                "value": int.__index__(value),
                "name": object.__getattribute__(value, "name"),
                "class": _class_token(value_type, seen),
            }
        }
    if value_type not in (tuple, list, dict, set, frozenset, bytearray):
        return _canonical(value, seen)
    identity = id(value)
    existing = references.get(identity)
    if existing is not None:
        return {"trusted_dependency_reference": existing}
    reference = len(references)
    references[identity] = reference
    if value_type in (tuple, list):
        payload: Any = {
            value_type.__name__: [
                _trusted_dependency_value_token(item, seen, references)
                for item in value
            ]
        }
    elif value_type is dict:
        payload = {
            "dict": [
                {
                    "key": _trusted_dependency_value_token(
                        key, seen, references
                    ),
                    "value": _trusted_dependency_value_token(
                        item, seen, references
                    ),
                }
                for key, item in dict.items(value)
            ]
        }
    elif value_type in (set, frozenset):
        unordered_items = [
            _trusted_dependency_value_token(
                item, _CanonicalTraversal(), {}
            )
            for item in value
        ]
        unordered_items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        payload = {
            value_type.__name__: unordered_items
        }
    else:
        payload = {"bytearray": bytes(value).hex()}
    return {
        "trusted_dependency_object": reference,
        "value": payload,
    }


def _trusted_stdlib_mutable_dependency_state(
    snapshot: dict[str, Any], root: types.FunctionType
) -> list[dict[str, Any]]:
    """Bind reachable mutable lookup tables without trusting their identity."""
    reachable: set[Any] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        pending.extend(snapshot["edges"].get(node, ()))

    def capture() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for binding in snapshot["bindings"]:
            kind, function = binding[:2]
            if function not in reachable:
                continue
            if kind == "name":
                _, _, source, name, expected, _ = binding
                if source == "global":
                    current = function.__globals__.get(name)
                elif source == "builtin-dict":
                    builtin_namespace = function.__builtins__
                    if type(builtin_namespace) is not dict:
                        raise BackendRecommendationError(
                            "trusted stdlib builtins mapping changed"
                        )
                    current = builtin_namespace.get(name)
                else:
                    if function.__builtins__ is not builtins:
                        raise BackendRecommendationError(
                            "trusted stdlib builtins module changed"
                        )
                    current = getattr(builtins, name, None)
                label = f"{source}:{name}"
            elif kind == "module-path":
                _, _, root_name, path, expected, _ = binding
                current = function.__globals__.get(root_name)
                for attribute in path:
                    current = getattr(current, attribute)
                label = "module:" + ".".join((root_name, *path))
            else:
                # The graph verifier binds Python method wrappers by stable
                # ``__func__``/``__self__`` pairs. They are not mutable table
                # state and their ephemeral wrapper identity is irrelevant.
                continue
            if current is not expected:
                raise BackendRecommendationError(
                    "trusted standard-library dependency binding changed"
                )
            if type(current) not in (
                tuple,
                list,
                dict,
                set,
                frozenset,
                bytearray,
            ):
                continue
            owner_module = str(function.__module__)
            derived_cache = (
                owner_module == "re" and label == "global:_cache"
            ) or (
                owner_module == "linecache" and label == "global:cache"
            )
            if derived_cache:
                # These caches are derived exclusively by the authenticated
                # stdlib implementation from the semantic tables that are
                # bound below. Their population history is not an input to a
                # callback and may legitimately differ across MPI ranks.
                continue
            if label in {
                "module:sys.modules",
                "module:sys.path",
            }:
                raise BackendRecommendationError(
                    "ambient standard-library module/path state cannot be "
                    "canonicalized as callback semantics"
                )
            traversal = _CanonicalTraversal()
            digest = _sha256_json(
                _trusted_dependency_value_token(current, traversal)
            )
            records.append(
                {
                    "owner": (
                        f"{function.__module__}.{function.__qualname__}"
                    ),
                    "binding": label,
                    "sha256": digest,
                }
            )
        records.sort(
            key=lambda record: (
                record["owner"], record["binding"], record["sha256"]
            )
        )
        return records

    before = capture()
    after = capture()
    if before != after:
        raise BackendRecommendationError(
            "trusted standard-library mutable dependency changed during "
            "canonicalization"
        )
    return before


_TRUSTED_STDLIB_DEPENDENCY_GRAPH = _trusted_dependency_graph_snapshot(
    tuple(_TRUSTED_STDLIB_HEAP_CLASS_MEMBERS)
)


def _is_package_owned_meep_dependency(value: Any) -> bool:
    package_root = Path(__file__).resolve().parent
    if type(value) is types.FunctionType:
        try:
            code_path = Path(value.__code__.co_filename).resolve(strict=True)
        except OSError:
            return False
        return _path_is_within(code_path, package_root)
    if isinstance(value, type):
        module_name = type.__getattribute__(value, "__module__")
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if (
            module_name != "meep"
            and not module_name.startswith("meep.")
        ) or not isinstance(module_file, str):
            return False
        try:
            module_path = Path(module_file).resolve(strict=True)
        except OSError:
            return False
        return _path_is_within(module_path, package_root)
    return False


_TRUSTED_MEEP_DEPENDENCY_GRAPH = _trusted_dependency_graph_snapshot(
    tuple(_TRUSTED_MEEP_CLASSES),
    traversal_filter=_is_package_owned_meep_dependency,
)


def _trusted_meep_dependency_is_unchanged(
    root: Any, seen: Any | None = None
) -> bool:
    graph = _TRUSTED_MEEP_DEPENDENCY_GRAPH
    if root not in graph["functions"] and root not in graph["classes"]:
        # Test doubles and explicitly injected instrumentation classes are
        # still handled by the full generic class/function token. Only
        # package-owned roots use the import-time dependency shortcut.
        return True
    if (
        isinstance(seen, _CanonicalTraversal)
        and seen.memoize_trusted_meep_dependency_graph
    ):
        if seen.trusted_meep_dependency_graph_checked:
            if isinstance(root, type) and id(root) in seen.class_tokens:
                # The exact token is reused without reading package state.
                # The traversal-end complete graph check still rejects a
                # persistent mutation before the token can be consumed.
                return True
            return _trusted_dependency_token_scope_is_unchanged(
                graph, root
            )
        unchanged = _trusted_dependency_graph_is_unchanged(graph)
        if unchanged:
            seen.trusted_meep_dependency_graph_checked = True
        return unchanged and _trusted_dependency_token_scope_is_unchanged(
            graph, root
        )
    return _trusted_dependency_graph_is_unchanged(graph, root=root)


def _finalize_trusted_meep_dependency_integrity(
    traversal: _CanonicalTraversal, label: str
) -> None:
    """Recheck a memoized package graph after one complete token traversal."""
    if not isinstance(traversal, _CanonicalTraversal):
        raise BackendRecommendationError(
            "trusted Meep dependency traversal state is malformed"
        )
    if not traversal.memoize_trusted_meep_dependency_graph:
        return
    if (
        traversal.trusted_meep_dependency_graph_checked
        and not _trusted_dependency_graph_is_unchanged(
            _TRUSTED_MEEP_DEPENDENCY_GRAPH
        )
    ):
        raise BackendRecommendationError(
            f"trusted Meep package dependency changed during {label} canonicalization"
        )


_TRUSTED_STDLIB_HEAP_CLASS_RUNTIME = {
    candidate: _late_class_runtime_snapshot(candidate)
    for candidate in _TRUSTED_STDLIB_HEAP_CLASS_MEMBERS
}
_TRUSTED_STDLIB_HEAP_CLASS_GLOBALS = {
    candidate: _late_class_global_snapshot(
        candidate, sys.modules[candidate.__module__]
    )
    for candidate in _TRUSTED_STDLIB_HEAP_CLASS_MEMBERS
}


def _trusted_stdlib_heap_class_is_unchanged(
    candidate: type, seen: set[int] | None = None
) -> bool:
    runtime = _TRUSTED_STDLIB_HEAP_CLASS_RUNTIME.get(candidate)
    globals_snapshot = _TRUSTED_STDLIB_HEAP_CLASS_GLOBALS.get(candidate)
    if runtime is None or globals_snapshot is None:
        return False
    if not _late_class_runtime_is_unchanged(candidate, runtime):
        return False
    module = sys.modules.get(candidate.__module__)
    namespace = getattr(module, "__dict__", None)
    return isinstance(namespace, dict) and all(
        namespace.get(name) is dependency
        for name, dependency in globals_snapshot.items()
    )


def _trusted_stdlib_function_is_unchanged(
    function: types.FunctionType, seen: set[int] | None = None
) -> bool:
    if isinstance(seen, _CanonicalTraversal):
        cached = seen.trusted_stdlib_function_integrity.get(id(function))
        if cached is not None:
            return cached
    unchanged = _trusted_dependency_graph_is_unchanged(
        _TRUSTED_STDLIB_DEPENDENCY_GRAPH, root=function
    )
    if isinstance(seen, _CanonicalTraversal):
        seen.trusted_stdlib_function_integrity[id(function)] = unchanged
    return unchanged


def _reject_implicit_namespace_operations(
    code: types.CodeType, *, allow_format_value: bool = False,
    seen: set[int] | None = None,
) -> None:
    forbidden = {
        "IMPORT_NAME",
        "IMPORT_FROM",
        "IMPORT_STAR",
        "LOAD_BUILD_CLASS",
        "BUILD_SET",
        "SET_UPDATE",
        "SET_ADD",
    }
    if not allow_format_value:
        forbidden.add("FORMAT_VALUE")
    unsupported = {
        instruction.opname
        for nested in _nested_code_objects(code)
        for instruction in _code_instructions(nested, seen)
        if instruction.opname in forbidden
    }
    if unsupported:
        raise BackendRecommendationError(
            "implicit import or class-building operations cannot be "
            "canonicalized exactly: " + ", ".join(sorted(unsupported))
        )
    sensitive_attributes = {
        str(instruction.argval)
        for nested in _nested_code_objects(code)
        for instruction in _code_instructions(nested, seen)
        if instruction.opname in ("LOAD_ATTR", "LOAD_METHOD")
        and str(instruction.argval)
        in {"__hash__", "__repr__", "__str__", "__format__"}
    }
    if sensitive_attributes:
        raise BackendRecommendationError(
            "identity-sensitive special method access cannot be "
            "canonicalized exactly: "
            + ", ".join(sorted(sensitive_attributes))
        )


def _format_values_are_builtin_print_only(
    function: types.FunctionType, seen: set[int] | None = None
) -> bool:
    """Accept f-strings only when their value is consumed by ``print``.

    ``FORMAT_VALUE`` can otherwise turn process-local object identity into a
    branch or a mutation which the acknowledgement digest cannot represent.
    Plain diagnostic output is different: the formatted value is consumed by
    an exact ``builtins.print`` call whose return value is discarded.  This
    conservative bytecode walk admits that common Meep callback pattern while
    continuing to reject returned strings and strings passed through another
    callable first.
    """

    namespace = function.__globals__
    if type(namespace) is not dict:
        return False
    sentinel = object()
    resolved_print = namespace.get("print", sentinel)
    if resolved_print is sentinel:
        builtin_namespace = function.__builtins__
        if type(builtin_namespace) is dict:
            resolved_print = builtin_namespace.get("print", sentinel)
        elif builtin_namespace is builtins:
            resolved_print = builtins.print
    if resolved_print is not _ORIGINAL_BUILTINS_PRINT:
        return False

    ignored = {
        "CACHE",
        "COPY_FREE_VARS",
        "EXTENDED_ARG",
        "MAKE_CELL",
        "NOP",
        "RESUME",
    }
    statement_boundaries = {
        "POP_TOP",
        "RETURN_VALUE",
        "RETURN_CONST",
        "RAISE_VARARGS",
        "RERAISE",
        "YIELD_VALUE",
    }

    def pop_items(stack: list[bool], count: int) -> list[bool]:
        popped: list[bool] = []
        for _ in range(max(0, count)):
            popped.append(stack.pop() if stack else False)
        return popped

    def is_print_statement(
        code: types.CodeType,
        instructions: list[dis.Instruction],
        call_index: int,
    ) -> bool:
        after = call_index + 1
        while after < len(instructions) and instructions[after].opname in ignored:
            after += 1
        if after >= len(instructions) or instructions[after].opname != "POP_TOP":
            return False

        start = call_index - 1
        while start >= 0:
            opname = instructions[start].opname
            if (
                opname in statement_boundaries
                or opname.startswith(("STORE_", "DELETE_", "JUMP", "POP_JUMP"))
                or opname == "FOR_ITER"
            ):
                break
            start -= 1
        start += 1
        while start < call_index and instructions[start].opname in ignored:
            start += 1
        for instruction in instructions[start:call_index]:
            if instruction.opname != "KW_NAMES":
                continue
            constant_index = instruction.arg
            if (
                not isinstance(constant_index, int)
                or constant_index < 0
                or constant_index >= len(code.co_consts)
            ):
                return False
            keyword_names = code.co_consts[constant_index]
            if (
                type(keyword_names) is not tuple
                or any(type(name) is not str for name in keyword_names)
                or "file" in keyword_names
            ):
                return False
        return (
            start < call_index
            and instructions[start].opname == "LOAD_GLOBAL"
            and instructions[start].argval == "print"
        )

    saw_format = False
    for nested in _nested_code_objects(function.__code__):
        instructions = list(_code_instructions(nested, seen))
        format_indices = [
            index
            for index, instruction in enumerate(instructions)
            if instruction.opname == "FORMAT_VALUE"
        ]
        saw_format = saw_format or bool(format_indices)
        for format_index in format_indices:
            # The current FORMAT_VALUE has replaced its input with one tagged
            # string.  Entries below it are deliberately represented as an
            # unbounded untagged prefix by pop_items().
            stack = [True]
            pending_tainted_arguments = False
            accepted = False
            for index in range(format_index + 1, len(instructions)):
                instruction = instructions[index]
                opname = instruction.opname
                if opname in ignored or opname == "KW_NAMES":
                    continue
                if opname.startswith("LOAD_") or opname == "PUSH_NULL":
                    if opname == "LOAD_ATTR":
                        tagged = any(pop_items(stack, 1))
                        stack.append(tagged)
                    elif opname == "LOAD_METHOD":
                        tagged = any(pop_items(stack, 1))
                        stack.extend((tagged, tagged))
                    else:
                        stack.append(False)
                    continue
                if opname == "COPY":
                    depth = int(instruction.arg or 1)
                    stack.append(stack[-depth] if depth <= len(stack) else False)
                    continue
                if opname == "SWAP":
                    depth = int(instruction.arg or 1)
                    if depth <= len(stack):
                        stack[-1], stack[-depth] = stack[-depth], stack[-1]
                    continue
                if opname == "FORMAT_VALUE":
                    tagged = any(pop_items(stack, 1))
                    stack.append(tagged)
                    continue
                if opname == "BUILD_STRING":
                    tagged = any(pop_items(stack, int(instruction.arg or 0)))
                    stack.append(tagged)
                    continue
                if opname in {"BUILD_TUPLE", "BUILD_LIST", "BUILD_SET"}:
                    tagged = any(pop_items(stack, int(instruction.arg or 0)))
                    stack.append(tagged)
                    continue
                if opname == "BUILD_MAP":
                    tagged = any(pop_items(stack, 2 * int(instruction.arg or 0)))
                    stack.append(tagged)
                    continue
                if opname == "BUILD_CONST_KEY_MAP":
                    tagged = any(
                        pop_items(stack, int(instruction.arg or 0) + 1)
                    )
                    stack.append(tagged)
                    continue
                if opname in {
                    "BINARY_OP",
                    "BINARY_SUBSCR",
                    "COMPARE_OP",
                    "IS_OP",
                    "CONTAINS_OP",
                }:
                    tagged = any(pop_items(stack, 2))
                    stack.append(tagged)
                    continue
                if opname.startswith("UNARY_") or opname == "GET_ITER":
                    tagged = any(pop_items(stack, 1))
                    stack.append(tagged)
                    continue
                if opname == "PRECALL":
                    pending_tainted_arguments = any(
                        pop_items(stack, int(instruction.arg or 0))
                    )
                    if pending_tainted_arguments:
                        next_index = index + 1
                        while (
                            next_index < len(instructions)
                            and instructions[next_index].opname in ignored
                        ):
                            next_index += 1
                        accepted = (
                            next_index < len(instructions)
                            and instructions[next_index].opname == "CALL"
                            and is_print_statement(
                                nested, instructions, next_index
                            )
                        )
                        break
                    continue
                if opname == "CALL":
                    # Python 3.11 PRECALL has already removed arguments.  The
                    # remaining callable protocol pair is replaced by one
                    # untagged result.  On bytecode without PRECALL, consume
                    # the explicit arguments here as well.
                    explicit = 0 if "PRECALL" in dis.opmap else int(
                        instruction.arg or 0
                    )
                    tagged = pending_tainted_arguments or any(
                        pop_items(stack, explicit + 2)
                    )
                    if tagged:
                        accepted = is_print_statement(
                            nested, instructions, index
                        )
                        break
                    stack.append(False)
                    pending_tainted_arguments = False
                    continue
                if opname.startswith(("STORE_", "DELETE_")) or opname in {
                    "POP_TOP",
                    "RETURN_VALUE",
                    "RETURN_CONST",
                    "LIST_APPEND",
                    "SET_ADD",
                    "MAP_ADD",
                }:
                    if any(stack):
                        break
                    continue
                if (
                    opname.startswith(("JUMP", "POP_JUMP"))
                    or opname == "FOR_ITER"
                ):
                    break

                # Unknown transformations are acceptable only before the
                # tagged formatted value can reach them.  Once tagged, fail
                # closed instead of guessing their stack/data-flow semantics.
                if any(stack):
                    break
                try:
                    effect = dis.stack_effect(
                        instruction.opcode,
                        instruction.arg,
                        jump=False,
                    )
                except (TypeError, ValueError):
                    break
                if effect < 0:
                    pop_items(stack, -effect)
                elif effect > 0:
                    stack.extend(False for _ in range(effect))
            if not accepted:
                return False
    return saw_format


_DYNAMIC_NAMESPACE_BUILTINS = frozenset(
    {
        builtins.globals,
        builtins.locals,
        builtins.vars,
        builtins.eval,
        builtins.exec,
        builtins.compile,
        builtins.__import__,
    }
)
_UNSUPPORTED_CALLABLE_BUILTINS = _DYNAMIC_NAMESPACE_BUILTINS | frozenset(
    {
        builtins.id,
        builtins.hash,
        builtins.repr,
        builtins.str,
        builtins.ascii,
        builtins.format,
        builtins.set,
        builtins.frozenset,
    }
)
_INTRINSIC_UNSUPPORTED_CALLABLE_VALUES = frozenset(
    {builtins.id, builtins.hash}
)


def _referenced_namespace_dependencies(
    referenced_names: set[str],
    global_namespace: dict[str, Any],
    builtin_namespace: Any,
    seen: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(global_namespace) is not dict or type(builtin_namespace) is not dict:
        raise BackendRecommendationError(
            "function global and builtins namespaces must be exact dictionaries"
        )
    globals_token: dict[str, Any] = {}
    builtins_token: dict[str, Any] = {}
    for name in sorted(referenced_names):
        if name == "__builtins__":
            raise BackendRecommendationError(
                "direct builtins namespace access cannot be canonicalized exactly"
            )
        if name in global_namespace:
            dependency = global_namespace[name]
            target = globals_token
        elif name in builtin_namespace:
            dependency = builtin_namespace[name]
            target = builtins_token
        else:
            builtins_token[name] = {"missing": True}
            continue
        if any(
            dependency is dynamic
            for dynamic in _UNSUPPORTED_CALLABLE_BUILTINS
        ):
            raise BackendRecommendationError(
                "dynamic namespace, identity-sensitive, or nondeterministic "
                "builtin cannot be canonicalized exactly"
            )
        target[name] = _global_dependency_token(dependency, seen)
    return globals_token, builtins_token


def _module_attribute_dependencies(
    code: types.CodeType,
    global_bindings: dict[str, Any],
    deref_bindings: dict[str, Any],
    fast_bindings: dict[str, Any],
    seen: set[int],
) -> dict[str, Any]:
    """Bind statically resolvable module attributes used by a callable."""
    dependencies: dict[str, Any] = {}
    unresolved: set[str] = set()
    for nested in _nested_code_objects(code):
        instructions = list(_code_instructions(nested, seen))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in (
                "LOAD_GLOBAL",
                "LOAD_NAME",
                "LOAD_DEREF",
                "LOAD_FAST",
            ):
                continue
            name = str(instruction.argval)
            if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                value = global_bindings.get(name)
            elif instruction.opname == "LOAD_DEREF":
                value = deref_bindings.get(name)
            else:
                value = fast_bindings.get(name) if nested is code else None
            if not isinstance(value, types.ModuleType):
                continue
            path = name
            resolved_any = False
            cursor = index + 1
            while cursor < len(instructions) and instructions[cursor].opname in (
                "LOAD_ATTR",
                "LOAD_METHOD",
            ):
                attribute = str(instructions[cursor].argval)
                path = f"{path}.{attribute}"
                try:
                    value = getattr(value, attribute)
                except Exception as error:
                    raise BackendRecommendationError(
                        f"cannot resolve callable module dependency {path!r}: {error}"
                    ) from error
                dependencies[path] = _global_dependency_token(value, seen)
                resolved_any = True
                cursor += 1
            if not resolved_any:
                unresolved.add(name)
    if unresolved:
        names = ", ".join(sorted(unresolved))
        raise BackendRecommendationError(
            "callable dynamically uses module state that cannot be bound "
            f"exactly: {names}"
        )
    return dependencies


def _global_dependency_token(value: Any, seen: set[int]) -> Any:
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__}
    return _canonical(value, seen)


def _direct_binding_token(value: Any, seen: set[int]) -> Any:
    """Allow a direct module binding only when bytecode proves its attr path."""
    if isinstance(value, types.ModuleType):
        return {"module_binding": value.__name__}
    return _canonical(value, seen)


def _builtin_callable_token(value: Any, seen: set[int]) -> dict[str, Any]:
    callable_name = getattr(value, "__name__", "")
    bound_self = getattr(value, "__self__", None)
    if (
        bound_self is not None
        and not isinstance(bound_self, (types.ModuleType, type))
        and callable_name in {
        "__hash__",
        "__repr__",
        "__str__",
        "__format__",
        }
    ):
        raise BackendRecommendationError(
            "identity-sensitive builtin method cannot be canonicalized exactly"
        )
    owner = getattr(value, "__objclass__", None)
    token: dict[str, Any] = {
        "builtin_callable": f"{getattr(value, '__module__', '')}."
        f"{getattr(value, '__qualname__', getattr(value, '__name__', type(value).__qualname__))}",
        "descriptor_owner": (
            f"{owner.__module__}.{owner.__qualname__}"
            if isinstance(owner, type)
            else ""
        ),
    }
    if bound_self is not None:
        if isinstance(bound_self, types.ModuleType):
            token["bound_module"] = bound_self.__name__
        elif isinstance(bound_self, type):
            token["bound_class"] = _class_token(bound_self, seen)
        else:
            token["bound_self"] = _canonical(bound_self, seen)
    return token


def _slot_names(value_type: type) -> tuple[tuple[type, str], ...]:
    names: list[tuple[type, str]] = []
    try:
        mro = type.__getattribute__(value_type, "__mro__")
    except Exception as error:
        raise BackendRecommendationError(
            "object slots cannot be canonicalized exactly"
        ) from error
    if type(mro) is not tuple:
        raise BackendRecommendationError(
            "object slots cannot be canonicalized exactly"
        )
    for owner in mro:
        try:
            namespace = type.__getattribute__(owner, "__dict__")
            owner_name = type.__getattribute__(owner, "__name__")
        except Exception as error:
            raise BackendRecommendationError(
                "object slots cannot be canonicalized exactly"
            ) from error
        if (
            type(namespace) is not types.MappingProxyType
            or type(owner_name) is not str
        ):
            raise BackendRecommendationError(
                "object slots cannot be canonicalized exactly"
            )
        raw_slots = namespace.get("__slots__", ())
        if type(raw_slots) is str:
            raw_slots = (raw_slots,)
        elif type(raw_slots) is tuple:
            pass
        else:
            raise BackendRecommendationError(
                "object slots cannot be canonicalized exactly"
            )
        candidates = raw_slots
        for raw_name in candidates:
            if not isinstance(raw_name, str):
                raise BackendRecommendationError(
                    "object slots cannot be canonicalized exactly"
                )
            if raw_name in ("__dict__", "__weakref__"):
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                name = f"_{owner_name.lstrip('_')}{raw_name}"
            names.append((owner, name))
    return tuple(names)


def _trusted_ephemeral_fields(value: Any) -> set[str]:
    """Exclude only derived/runtime state owned by exact gpmeep classes."""
    value_type = type(value)
    module = value_type.__module__
    name = value_type.__qualname__
    excluded: set[str] = set()
    if _is_trusted_meep_class(value_type) and module.startswith("meep."):
        excluded.update({"swigobj", "swig_obj"})
    if _is_exact_trusted_meep_type(
        value, "meep.simulation", "Simulation"
    ):
        excluded.update(_EPHEMERAL_OBJECT_FIELDS)
        excluded.update(
            key
            for key in getattr(value, "__dict__", {})
            if key.startswith("_gpmeep_")
        )
    if _is_exact_trusted_meep_type(
        value, "meep.verbosity_mgr", "Verbosity"
    ):
        excluded.add("_cvars")
    return excluded


def _instance_state(value: Any, seen: set[int]) -> dict[str, Any] | None:
    value_type = type(value)
    extension_base = False
    if not _is_trusted_meep_class(value_type):
        for base in value_type.__mro__[1:]:
            if base is object:
                continue
            try:
                source_file = inspect.getsourcefile(base)
            except (OSError, TypeError):
                source_file = None
            if (
                not (
                    int(getattr(base, "__flags__", 0))
                    & _PY_TPFLAGS_HEAPTYPE
                )
                or source_file is None
            ):
                extension_base = True
                break
    if extension_base:
        raise BackendRecommendationError(
            "instances with hidden extension-base state cannot be "
            "canonicalized exactly"
        )
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except AttributeError:
        attributes = None
    except Exception as error:
        raise BackendRecommendationError(
            "instance dictionary cannot be canonicalized exactly"
        ) from error
    if attributes is not None and type(attributes) is not dict:
        raise BackendRecommendationError(
            "instance dictionary cannot be canonicalized exactly"
        )
    slots = _slot_names(type(value))
    if attributes is None and not slots:
        return None
    excluded = _trusted_ephemeral_fields(value)
    dictionary_state = None
    if attributes is not None:
        if _is_exact_trusted_meep_type(
            value, "meep.simulation", "Simulation"
        ):
            # The root Simulation's rank-local fields/structure manifest is
            # bound once, in ordered MPI-rank order, by guard_run.  Bound
            # methods retained by DFT wrappers can point back to that same
            # object; including its native manifest again here would turn a
            # legitimate rank-local epoch difference into a false semantic
            # Python-workload mismatch.  Other nested Simulation instances
            # remain fully bound because they are not the guarded root.
            include_native_state = not (
                isinstance(seen, _CanonicalTraversal)
                and id(value) in seen.native_simulation_roots
            )
            dictionary_state = _simulation_workload_state(
                value, include_native_state=include_native_state
            )
        else:
            dictionary_state = {
                key: item for key, item in attributes.items() if key not in excluded
            }
    slot_state: dict[str, Any] = {}
    for owner, name in slots:
        try:
            owner_namespace = type.__getattribute__(owner, "__dict__")
            owner_module = type.__getattribute__(owner, "__module__")
            owner_qualname = type.__getattribute__(owner, "__qualname__")
        except Exception as error:
            raise BackendRecommendationError(
                "object slots cannot be canonicalized exactly"
            ) from error
        if (
            type(owner_namespace) is not types.MappingProxyType
            or type(owner_module) is not str
            or type(owner_qualname) is not str
        ):
            raise BackendRecommendationError(
                "object slots cannot be canonicalized exactly"
            )
        descriptor = owner_namespace.get(name)
        if descriptor is None:
            continue
        slot_key = f"{owner_module}.{owner_qualname}.{name}"
        try:
            item = descriptor.__get__(value, type(value))
        except AttributeError:
            slot_state[slot_key] = {"slot": "unset"}
        except Exception as error:
            raise BackendRecommendationError(
                "object slots cannot be canonicalized exactly"
            ) from error
        else:
            slot_state[slot_key] = item
    try:
        canonical_dictionary = _canonical(dictionary_state, seen)
        canonical_slots = _canonical(slot_state, seen)
    except BackendRecommendationError as error:
        raise BackendRecommendationError(
            "object state cannot be canonicalized exactly for "
            f"{type(value).__module__}.{type(value).__qualname__}: {error}"
        ) from error
    return {
        "dict": canonical_dictionary,
        "slots": canonical_slots,
    }


def _definition_annotation_token(
    annotations: Any, seen: set[int]
) -> list[dict[str, Any]]:
    if type(annotations) is not dict:
        raise BackendRecommendationError(
            "callable annotations cannot be canonicalized exactly"
        )
    if any(type(key) is not str for key in annotations):
        raise BackendRecommendationError(
            "callable annotation keys must be text"
        )
    return [
        {"name": key, "value": _canonical(item, seen)}
        for key, item in dict.items(annotations)
    ]


def _shallow_function_token(
    value: Any, seen: set[int], *, transitive: bool
) -> Any:
    if type(value) in (staticmethod, classmethod):
        attributes = object.__getattribute__(value, "__dict__")
        if type(attributes) is not dict:
            raise BackendRecommendationError(
                "descriptor wrapper state cannot be canonicalized exactly"
            )
        return {
            type(value).__name__: {
                "function": _shallow_function_token(
                    value.__func__, seen, transitive=transitive
                ),
                "attributes": _canonical(attributes, seen),
            }
        }
    if isinstance(value, (staticmethod, classmethod)):
        raise BackendRecommendationError(
            "descriptor wrapper subclasses cannot be canonicalized exactly"
        )
    if type(value) is property:
        return {
            "property": {
                "get": _shallow_function_token(
                    value.fget, seen, transitive=transitive
                ),
                "set": _shallow_function_token(
                    value.fset, seen, transitive=transitive
                ),
                "delete": _shallow_function_token(
                    value.fdel, seen, transitive=transitive
                ),
                "doc": _canonical(value.__doc__, seen),
                "name": _canonical(getattr(value, "__name__", None), seen),
                "isabstractmethod": bool(value.__isabstractmethod__),
            }
        }
    if isinstance(value, property):
        raise BackendRecommendationError(
            "property subclasses cannot be canonicalized exactly"
        )
    if value is None:
        return None
    if transitive:
        return _canonical(value, seen)
    function = value.__func__ if isinstance(value, types.MethodType) else value
    code = getattr(function, "__code__", None)
    if code is None:
        if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            return _builtin_callable_token(value, seen)
        if isinstance(value, _IMMUTABLE_DESCRIPTOR_TYPES):
            return _builtin_callable_token(value, seen)
        return _canonical(value, seen)
    annotations = getattr(function, "__annotations__", {})
    trusted_annotations = (
        _trusted_function_annotation_token(function, annotations)
        if not transitive and type(function) is types.FunctionType
        else None
    )
    return {
        "function": f"{getattr(function, '__module__', '')}."
        f"{getattr(function, '__qualname__', type(function).__qualname__)}",
        "code": _code_token(code, seen),
        "defaults": _canonical(getattr(function, "__defaults__", None), seen),
        "kwdefaults": _canonical(getattr(function, "__kwdefaults__", None), seen),
        "annotations": (
            trusted_annotations
            if trusted_annotations is not None
            else _definition_annotation_token(annotations, seen)
        ),
        "attributes": _canonical(getattr(function, "__dict__", {}), seen),
    }


def _trusted_meep_snapshot_function_token(
    function: types.FunctionType, seen: set[int]
) -> dict[str, Any]:
    """Serialize a trusted function from its import-time runtime snapshot.

    Boundary and node checks authenticate the live object.  Reading the
    already-authenticated references here prevents a concurrent temporary
    ``__code__``/defaults replacement from contaminating a workload token and
    then disappearing before the post-token check.
    """
    runtime = _TRUSTED_MEEP_DEPENDENCY_GRAPH["functions"].get(function)
    if runtime is None or runtime.get("object") is not function:
        raise BackendRecommendationError(
            "trusted Meep function has no import-time dependency snapshot"
        )
    annotation_snapshot = _TRUSTED_FUNCTION_ANNOTATION_SNAPSHOTS.get(
        function
    )
    annotations = (
        annotation_snapshot["token"]
        if annotation_snapshot is not None
        else _definition_annotation_token(
            runtime["annotations_object"], seen
        )
    )
    return {
        "function": f"{runtime['module']}.{runtime['qualname']}",
        "code": _code_token(runtime["code"], seen),
        "defaults": _canonical(runtime["defaults_object"], seen),
        "kwdefaults": _canonical(runtime["kwdefaults_object"], seen),
        "annotations": annotations,
        "attributes": _canonical(runtime["attributes_object"], seen),
    }


def _trusted_meep_snapshot_member_token(
    value: Any, seen: set[int]
) -> Any:
    """Serialize an original trusted descriptor from captured function state."""
    functions = _trusted_member_functions(value)
    if not functions or any(
        function not in _TRUSTED_MEEP_DEPENDENCY_GRAPH["functions"]
        for function in functions
    ):
        # Synthetic test graphs and optional modules may deliberately expose
        # an original member that is outside the graph being checked.  It is
        # not eligible for the dependency shortcut and retains the complete
        # live token path.
        return _shallow_function_token(value, seen, transitive=False)
    if type(value) in (staticmethod, classmethod):
        attributes = object.__getattribute__(value, "__dict__")
        if type(attributes) is not dict:
            raise BackendRecommendationError(
                "trusted Meep descriptor wrapper state is malformed"
            )
        return {
            type(value).__name__: {
                "function": _trusted_meep_snapshot_function_token(
                    value.__func__, seen
                ),
                "attributes": _canonical(attributes, seen),
            }
        }
    if type(value) is property:
        def function_token(function: Any) -> Any:
            return (
                None
                if function is None
                else _trusted_meep_snapshot_function_token(function, seen)
            )

        return {
            "property": {
                "get": function_token(value.fget),
                "set": function_token(value.fset),
                "delete": function_token(value.fdel),
                "doc": _canonical(value.__doc__, seen),
                "name": _canonical(getattr(value, "__name__", None), seen),
                "isabstractmethod": bool(value.__isabstractmethod__),
            }
        }
    function = value.__func__ if isinstance(value, types.MethodType) else value
    if type(function) is not types.FunctionType:
        raise BackendRecommendationError(
            "trusted Meep class member has no exact Python function"
        )
    return _trusted_meep_snapshot_function_token(function, seen)


def _annotation_token(annotations: Any, seen: set[int]) -> Any:
    if type(annotations) is not dict:
        raise BackendRecommendationError(
            "callable annotations cannot be canonicalized exactly"
        )
    if any(type(key) is not str for key in annotations):
        raise BackendRecommendationError(
            "callable annotation keys must be text"
        )
    return _canonical(annotations, seen)


_NAMEDTUPLE_GENERATED_MEMBER_NAMES = frozenset(
    {
        "__new__",
        "_make",
        "__replace__",
        "_replace",
        "__repr__",
        "_asdict",
        "__getnewargs__",
    }
)


def _namedtuple_class_fields(value_type: type) -> tuple[str, ...] | None:
    """Recognize a direct, slotless namedtuple without dynamic lookup."""
    try:
        namespace = type.__getattribute__(value_type, "__dict__")
        bases = type.__getattribute__(value_type, "__bases__")
    except Exception:
        return None
    if type(namespace) is not types.MappingProxyType or bases != (tuple,):
        return None
    fields = namespace.get("_fields")
    defaults = namespace.get("_field_defaults")
    if (
        namespace.get("__slots__") != ()
        or "__dict__" in namespace
        or type(fields) is not tuple
        or not fields
        or any(type(field) is not str for field in fields)
        or len(set(fields)) != len(fields)
        or type(defaults) is not dict
        or tuple(defaults) != fields[len(fields) - len(defaults) :]
        or namespace.get("__match_args__", fields) != fields
        or type(namespace.get("__new__")) is not staticmethod
        or type(namespace.get("_make")) is not classmethod
        or type(namespace.get("_replace")) is not types.FunctionType
        or type(namespace.get("__repr__")) is not types.FunctionType
        or type(namespace.get("_asdict")) is not types.FunctionType
        or type(namespace.get("__getnewargs__")) is not types.FunctionType
    ):
        return None
    for field in fields:
        getter = namespace.get(field)
        if not (
            type(getter).__module__ == "_collections"
            and type(getter).__qualname__ == "_tuplegetter"
        ):
            return None
    replace_alias = namespace.get("__replace__")
    if replace_alias is not None and replace_alias is not namespace["_replace"]:
        return None
    return fields


def _namedtuple_generated_member_token(
    member_name: str,
    value: Any,
    fields: tuple[str, ...],
    owner_prefix: str,
    seen: set[int],
) -> Any:
    wrapper_name = None
    wrapper_attributes = None
    function = value
    if type(value) in (staticmethod, classmethod):
        wrapper_name = type(value).__name__
        wrapper_attributes = object.__getattribute__(value, "__dict__")
        if type(wrapper_attributes) is not dict:
            raise BackendRecommendationError(
                "namedtuple wrapper state cannot be canonicalized exactly"
            )
        function = value.__func__
    if type(function) is not types.FunctionType:
        raise BackendRecommendationError(
            "namedtuple generated member cannot be canonicalized exactly"
        )
    allow_format_value = False
    if member_name in _ORIGINAL_NAMEDTUPLE_FORMAT_CODES:
        closure = function.__closure__ or ()
        if (
            collections.namedtuple is not _ORIGINAL_COLLECTIONS_NAMEDTUPLE
            or function.__code__
            is not _ORIGINAL_NAMEDTUPLE_FORMAT_CODES[member_name]
            or function.__globals__
            is not _ORIGINAL_COLLECTIONS_NAMEDTUPLE_GLOBALS
            or function.__name__ != member_name
            or function.__qualname__ != f"{owner_prefix}.{member_name}"
            or function.__module__ != "collections"
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
            or function.__annotations__ != {}
            or function.__dict__ != {}
            or len(closure) != len(function.__code__.co_freevars)
        ):
            raise BackendRecommendationError(
                "namedtuple formatted member is not an authentic generated "
                "function"
            )
        closure_values = dict(
            zip(
                function.__code__.co_freevars,
                (cell.cell_contents for cell in closure),
            )
        )
        if member_name == "_make":
            authentic_closure = (
                set(closure_values) == {"_len", "num_fields", "tuple_new"}
                and closure_values["_len"] is len
                and type(closure_values["num_fields"]) is int
                and closure_values["num_fields"] == len(fields)
                and closure_values["tuple_new"] is tuple.__new__
            )
        else:
            authentic_closure = (
                set(closure_values) == {"_map", "field_names"}
                and closure_values["_map"] is map
                and closure_values["field_names"] is fields
            )
        if not authentic_closure:
            raise BackendRecommendationError(
                "namedtuple formatted member closure is not authentic"
            )
        allow_format_value = True
    token: Any = _callable_token(
        function,
        seen,
        allow_format_value=allow_format_value,
    )
    if wrapper_name is not None:
        token = {
            wrapper_name: {
                "function": token,
                "attributes": _canonical(wrapper_attributes, seen),
            }
        }
    return token


def _abc_registry_token(
    value: type, implementation: Any
) -> list[dict[str, Any]]:
    """Return the semantic virtual-subclass registry of an exact ABC.

    ABC positive and negative caches are derived performance state.  The
    registry itself changes ``issubclass``/``isinstance`` behavior, so it is
    resolved through the saved CPython accessor and tokenized.  Two snapshots
    around token construction make concurrent registration fail closed.
    """
    if (
        type(value) is not _ORIGINAL_ABC_META
        or type(implementation) is not _ORIGINAL_ABC_IMPL_TYPE
        or abc._get_dump is not _ORIGINAL_ABC_GET_DUMP
    ):
        raise BackendRecommendationError(
            "ABC registry implementation cannot be authenticated exactly"
        )

    def snapshot() -> tuple[
        tuple[tuple[int, int], ...], tuple[type, ...]
    ]:
        try:
            dump = _ORIGINAL_ABC_GET_DUMP(value)
        except Exception as error:
            raise BackendRecommendationError(
                "ABC registry state cannot be inspected exactly"
            ) from error
        if (
            type(dump) is not tuple
            or len(dump) != 4
            or any(type(item) is not set for item in dump[:3])
            or type(dump[3]) is not int
        ):
            raise BackendRecommendationError(
                "ABC registry state is malformed"
            )
        resolved: list[type] = []
        identities: list[tuple[int, int]] = []
        for reference in dump[0]:
            if type(reference) is not weakref.ReferenceType:
                raise BackendRecommendationError(
                    "ABC virtual-subclass registry is malformed"
                )
            registered = reference()
            if not isinstance(registered, type):
                raise BackendRecommendationError(
                    "ABC virtual-subclass registry contains a stale class"
                )
            resolved.append(registered)
            identities.append((id(reference), id(registered)))
        if len({id(item) for item in resolved}) != len(resolved):
            raise BackendRecommendationError(
                "ABC virtual-subclass registry contains duplicate classes"
            )
        return tuple(sorted(identities)), tuple(resolved)

    before_identity, registered_classes = snapshot()
    tokens = [
        {"virtual_subclass": _class_token(item, _CanonicalTraversal())}
        for item in registered_classes
    ]
    tokens.sort(
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    after_identity, _ = snapshot()
    if before_identity != after_identity:
        raise BackendRecommendationError(
            "ABC virtual-subclass registry changed during canonicalization"
        )
    return tokens


def _class_token(value: type, seen: set[int]) -> dict[str, Any]:
    try:
        module = type.__getattribute__(value, "__module__")
        qualname = type.__getattribute__(value, "__qualname__")
        flags_value = type.__getattribute__(value, "__flags__")
        namespace = type.__getattribute__(value, "__dict__")
        bases = type.__getattribute__(value, "__bases__")
    except Exception as error:
        raise BackendRecommendationError(
            "class internals cannot be canonicalized exactly"
        ) from error
    if (
        type(module) is not str
        or type(qualname) is not str
        or type(flags_value) is not int
        or type(namespace) is not types.MappingProxyType
        or type(bases) is not tuple
        or any(not isinstance(base, type) for base in bases)
    ):
        raise BackendRecommendationError(
            "class internals are malformed"
        )
    name = f"{module}.{qualname}"
    flags = flags_value
    if not flags & _PY_TPFLAGS_HEAPTYPE:
        return {"immutable_class": name}
    stdlib_members = _TRUSTED_STDLIB_HEAP_CLASS_MEMBERS.get(value)
    if stdlib_members is not None:
        if not _trusted_stdlib_heap_class_is_unchanged(value, seen):
            raise BackendRecommendationError(
                f"trusted standard-library class {name} was modified"
            )
        current_members = namespace
        if set(current_members) != set(stdlib_members) or any(
            current_members.get(key) is not member
            for key, member in stdlib_members.items()
        ):
            raise BackendRecommendationError(
                f"trusted standard-library class {name} was modified"
            )
        if value in _TRUSTED_STDLIB_CLASSES_REQUIRING_PROJECTION:
            raise BackendRecommendationError(
                f"trusted standard-library class {name} must be captured "
                "through a statically projected method"
            )
        if not _trusted_dependency_graph_is_unchanged(
            _TRUSTED_STDLIB_DEPENDENCY_GRAPH, root=value
        ):
            raise BackendRecommendationError(
                f"trusted standard-library class {name} dependency was "
                "modified"
            )
        token = {
            "trusted_stdlib_class": name,
            "bases": [_class_token(base, seen) for base in bases],
            "metaclass": _class_token(type(value), seen),
        }
        if not _trusted_dependency_graph_is_unchanged(
            _TRUSTED_STDLIB_DEPENDENCY_GRAPH, root=value
        ):
            raise BackendRecommendationError(
                f"trusted standard-library class {name} dependency changed "
                "during canonicalization"
            )
        return token
    trusted_meep_dependency = (
        _is_trusted_meep_class(value)
        and (
            value in _TRUSTED_MEEP_DEPENDENCY_GRAPH["classes"]
        )
    )
    if trusted_meep_dependency and not (
        _trusted_meep_dependency_is_unchanged(value, seen)
    ):
        raise BackendRecommendationError(
            f"trusted Meep class {name} package dependency was modified"
        )
    if isinstance(seen, _CanonicalTraversal):
        identity = id(value)
        if identity in seen.active_classes:
            return {"class_cycle": name}
        if identity in seen.class_tokens:
            return seen.class_tokens[identity]
        seen.active_classes.add(identity)
    else:
        identity = id(value)
        if identity in seen:
            return {"class_cycle": name}
        seen.add(identity)
    try:
        namedtuple_fields = _namedtuple_class_fields(value)
        namedtuple_owner_prefix = None
        if namedtuple_fields is not None:
            namedtuple_new = namespace["__new__"].__func__
            namedtuple_owner_prefix = namedtuple_new.__qualname__.rsplit(
                ".", 1
            )[0]
        attributes: list[dict[str, Any]] = []
        for key, item in namespace.items():
            if (
                key == "__abstractmethods__"
                and type(value) is _ORIGINAL_ABC_META
            ):
                if type(item) is not frozenset or any(
                    type(name) is not str for name in item
                ):
                    raise BackendRecommendationError(
                        "abstract method state cannot be canonicalized exactly"
                    )
                # Preserve the container's definition/reference topology: a
                # callback can explicitly compare this exact frozenset by
                # identity.  Only a newly defined ABC membership set has its
                # hash-seeded element order normalized.  Captured user sets
                # remain order-sensitive elsewhere because their iteration can
                # affect a callback's result.
                item_token = _canonical(item, seen)
                if "object_reference" not in item_token:
                    semantic_token = item_token.get("value", item_token)
                    set_token = semantic_token.get("set", {})
                    if (
                        set_token.get("type") != "frozenset"
                        or type(set_token.get("items")) is not list
                        or any(
                            type(name) is not str
                            for name in set_token["items"]
                        )
                    ):
                        raise BackendRecommendationError(
                            "abstract method state cannot be canonicalized "
                            "exactly"
                        )
                    set_token["items"].sort()
            elif key == "_abc_impl":
                item_token = {
                    "abc_virtual_registry": _abc_registry_token(value, item)
                }
            elif key.startswith("_abc_"):
                raise BackendRecommendationError(
                    "ABC runtime class state cannot be canonicalized exactly"
                )
            elif isinstance(
                item,
                (
                    types.MemberDescriptorType,
                    types.GetSetDescriptorType,
                ),
            ):
                descriptor_owner = getattr(item, "__objclass__", None)
                if not isinstance(descriptor_owner, type):
                    raise BackendRecommendationError(
                        "data descriptor owner cannot be canonicalized exactly"
                    )
                item_token = {
                    "descriptor": (
                        f"{type(item).__module__}.{type(item).__qualname__}"
                    ),
                    "name": _canonical(getattr(item, "__name__", None), seen),
                    "qualname": _canonical(
                        getattr(item, "__qualname__", None), seen
                    ),
                    "owner": _class_token(descriptor_owner, seen),
                    "doc": _canonical(getattr(item, "__doc__", None), seen),
                }
            elif (
                type(item).__module__ == "_collections"
                and type(item).__qualname__ == "_tuplegetter"
            ):
                item_token = {
                    "namedtuple_getter": repr(item),
                }
            elif isinstance(
                item,
                (types.FunctionType, staticmethod, classmethod, property),
            ):
                if (
                    namedtuple_fields is not None
                    and key in _NAMEDTUPLE_GENERATED_MEMBER_NAMES
                ):
                    item_token = _namedtuple_generated_member_token(
                        key,
                        item,
                        namedtuple_fields,
                        namedtuple_owner_prefix,
                        seen,
                    )
                elif _is_original_trusted_meep_member(value, key, item):
                    item_token = _trusted_meep_snapshot_member_token(
                        item, seen
                    )
                else:
                    item_token = _shallow_function_token(
                        item,
                        seen,
                        transitive=not _is_original_trusted_stdlib_function(
                            (
                                item.__func__
                                if type(item) in (staticmethod, classmethod)
                                else item
                            ),
                            seen,
                        ),
                    )
            else:
                item_token = _canonical(item, seen)
            attributes.append({"name": key, "value": item_token})
        metaclass = type(value)
        token = {
            "class": name,
            "bases": [_class_token(base, seen) for base in bases],
            "metaclass": _class_token(metaclass, seen),
            "attributes": attributes,
        }
        if trusted_meep_dependency and not (
            _trusted_meep_dependency_is_unchanged(value, seen)
        ):
            raise BackendRecommendationError(
                f"trusted Meep class {name} package dependency changed "
                "during canonicalization"
            )
        if isinstance(seen, _CanonicalTraversal):
            seen.class_tokens[id(value)] = token
        return token
    finally:
        if isinstance(seen, _CanonicalTraversal):
            seen.active_classes.remove(id(value))
        else:
            seen.remove(id(value))


def _static_binding_attribute_names(
    code: types.CodeType,
    binding_name: str,
    *,
    root_fast: bool,
    seen: set[int] | None = None,
) -> tuple[str, ...] | None:
    """Return first-hop attributes when every binding use is statically explicit."""
    attributes: set[str] = set()
    saw_binding = False
    for nested_index, nested in enumerate(_nested_code_objects(code)):
        instructions = list(_code_instructions(nested, seen))
        for index, instruction in enumerate(instructions):
            if str(instruction.argval) != binding_name:
                continue
            if instruction.opname == "LOAD_CLOSURE":
                continue
            allowed_load = instruction.opname == "LOAD_DEREF" or (
                root_fast
                and nested_index == 0
                and instruction.opname == "LOAD_FAST"
            )
            if not allowed_load:
                if instruction.opname in (
                    "STORE_DEREF",
                    "DELETE_DEREF",
                    "STORE_FAST",
                    "DELETE_FAST",
                    "LOAD_CLASSDEREF",
                ):
                    return None
                continue
            saw_binding = True
            if index + 1 >= len(instructions) or instructions[
                index + 1
            ].opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                return None
            attribute = instructions[index + 1].argval
            if not isinstance(attribute, str):
                return None
            if attribute in {
                "__hash__",
                "__repr__",
                "__str__",
                "__format__",
            }:
                raise BackendRecommendationError(
                    "identity-sensitive special method access cannot be "
                    "canonicalized exactly"
                )
            attributes.add(attribute)
    if not saw_binding or not attributes:
        return None
    return tuple(sorted(attributes))


def _static_nested_attribute_names(
    code: types.CodeType,
    binding_name: str,
    first_attribute: str,
    *,
    root_fast: bool,
    seen: set[int] | None = None,
) -> tuple[str, ...] | None:
    """Return authenticated second-hop reads for one projected attribute."""

    attributes: set[str] = set()
    saw_first_attribute = False
    for nested_index, nested in enumerate(_nested_code_objects(code)):
        instructions = list(_code_instructions(nested, seen))
        for index, instruction in enumerate(instructions):
            if str(instruction.argval) != binding_name:
                continue
            allowed_load = instruction.opname == "LOAD_DEREF" or (
                root_fast
                and nested_index == 0
                and instruction.opname == "LOAD_FAST"
            )
            if not allowed_load or index + 1 >= len(instructions):
                continue
            first = instructions[index + 1]
            if (
                first.opname not in ("LOAD_ATTR", "LOAD_METHOD")
                or first.argval != first_attribute
            ):
                continue
            saw_first_attribute = True
            if index + 2 >= len(instructions):
                return None
            second = instructions[index + 2]
            if second.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                return None
            attribute = second.argval
            if not isinstance(attribute, str):
                return None
            if attribute in {
                "__hash__",
                "__repr__",
                "__str__",
                "__format__",
            }:
                raise BackendRecommendationError(
                    "identity-sensitive special method access cannot be "
                    "canonicalized exactly"
                )
            if attribute in {"fields", "structure"}:
                # These native handles are sealed by their ordered rank-local
                # manifests rather than by recursive Python-object
                # canonicalization.  Permit the established terminal
                # projection, but do not let a newly-supported holder route
                # expose unsealed Python attributes or aliases of the handle.
                following = index + 3
                while (
                    following < len(instructions)
                    and instructions[following].opname
                    in {"CACHE", "EXTENDED_ARG", "NOP"}
                ):
                    following += 1
                if (
                    following >= len(instructions)
                    or instructions[following].opname
                    not in {"POP_TOP", "RETURN_VALUE", "RETURN_CONST"}
                ):
                    return None
            attributes.add(attribute)
    if not saw_first_attribute or not attributes:
        return None
    return tuple(sorted(attributes))


def _projected_binding_token(
    value: Any,
    attributes: tuple[str, ...],
    seen: set[int],
    active: set[tuple[int, int]] | None = None,
    instance_value_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Bind only statically used instance-dict attributes under default lookup."""
    value_type = type(value)
    dictionary = getattr(value, "__dict__", None)
    if not isinstance(dictionary, dict):
        return None
    try:
        getattribute = inspect.getattr_static(value_type, "__getattribute__")
    except AttributeError:
        return None
    if getattribute is not object.__getattribute__:
        return None
    missing = object()
    try:
        getattr_fallback = inspect.getattr_static(
            value_type, "__getattr__", missing
        )
    except Exception:
        return None
    if getattr_fallback is not missing:
        return None
    projected: dict[str, Any] = {}
    if active is None:
        active = set()
    for name in attributes:
        try:
            class_attribute = inspect.getattr_static(value_type, name, missing)
        except Exception:
            return None
        if name in dictionary:
            if class_attribute is not missing and inspect.isdatadescriptor(
                class_attribute
            ):
                return None
            projected[name] = {
                "instance_value": (
                    instance_value_overrides[name]
                    if instance_value_overrides is not None
                    and name in instance_value_overrides
                    else _canonical(dictionary[name], seen)
                ),
                "shadowed_class_attribute": (
                    None
                    if class_attribute is missing
                    else _canonical(class_attribute, seen)
                ),
            }
            continue
        if class_attribute is missing or inspect.isdatadescriptor(class_attribute):
            return None
        if isinstance(class_attribute, types.FunctionType):
            code = class_attribute.__code__
            receiver_name = (
                str(code.co_varnames[0]) if code.co_argcount > 0 else "self"
            )
            receiver_attributes = _static_binding_attribute_names(
                code,
                receiver_name,
                root_fast=True,
                seen=seen,
            )
            if receiver_attributes is None:
                return None
            recursion_key = (id(value), id(class_attribute))
            if recursion_key in active:
                return None
            active.add(recursion_key)
            try:
                receiver_token = _projected_binding_token(
                    value,
                    receiver_attributes,
                    seen,
                    active,
                )
            finally:
                active.remove(recursion_key)
            if receiver_token is None:
                return None
            projected[name] = {
                "bound_class_function": _canonical(class_attribute, seen),
                "receiver": receiver_token,
            }
            continue
        if hasattr(class_attribute, "__get__"):
            return None
        projected[name] = {
            "class_value": _canonical(class_attribute, seen),
        }
    return {
        "static_attribute_projection": {
            "owner_type": f"{value_type.__module__}.{value_type.__qualname__}",
            "owner_mro": [
                f"{owner.__module__}.{owner.__qualname__}"
                for owner in value_type.__mro__
            ],
            "lookup": "builtins.object.__getattribute__",
            "attributes": projected,
        }
    }


def _validate_guarded_simulation_projection(
    simulation: Any,
    attributes: tuple[str, ...],
    active: set[tuple[int, str]] | None = None,
) -> None:
    """Prove that a root projection cannot escape its sealed state routes."""
    dictionary = object.__getattribute__(simulation, "__dict__")
    if type(dictionary) is not dict:
        raise BackendRecommendationError(
            "guarded Simulation root dictionary is malformed"
        )
    value_type = type(simulation)
    missing = object()
    if active is None:
        active = set()
    for name in attributes:
        if name.startswith("_gpmeep_") or name in _SIMULATION_UNBOUND_RUNTIME_FIELDS:
            raise BackendRecommendationError(
                "guarded Simulation callback reads runtime state that is not "
                "bound by the workload or native digest: " + name
            )
        if name in {"fields", "structure"}:
            continue
        try:
            class_attribute = inspect.getattr_static(value_type, name, missing)
        except Exception as error:
            raise BackendRecommendationError(
                "guarded Simulation root lookup cannot be authenticated"
            ) from error
        if name in dictionary:
            if class_attribute is not missing and inspect.isdatadescriptor(
                class_attribute
            ):
                raise BackendRecommendationError(
                    "guarded Simulation root data-descriptor lookup cannot be "
                    "projected exactly"
                )
            projected_value = dictionary[name]
        elif isinstance(class_attribute, types.FunctionType):
            if _is_original_trusted_meep_function(class_attribute):
                # The guarded root's complete semantic Python state and its
                # ordered per-rank native digest are already bound by
                # ``_workload_record``.  Original Simulation methods may
                # legitimately pass ``self`` through several private helpers
                # (for example the delayed DFT monitor initializers), which
                # cannot be represented as a finite first-hop projection.
                # Authenticate the exact import-time implementation and
                # reject direct reads of policy-only or otherwise unbound
                # runtime fields; the method code itself is carried by the
                # enclosing callable token.
                direct_attributes = {
                    str(instruction.argval)
                    for nested in _nested_code_objects(
                        class_attribute.__code__
                    )
                    for instruction in dis.get_instructions(nested)
                    if instruction.opname in ("LOAD_ATTR", "LOAD_METHOD")
                }
                forbidden = sorted(
                    attribute
                    for attribute in direct_attributes
                    if attribute.startswith("_gpmeep_")
                    or attribute in _SIMULATION_UNBOUND_RUNTIME_FIELDS
                )
                if forbidden:
                    raise BackendRecommendationError(
                        "guarded Simulation method reads runtime state that "
                        "is not bound by the workload or native digest: "
                        + ", ".join(forbidden)
                    )
                continue
            recursion_key = (id(class_attribute), name)
            if recursion_key in active:
                raise BackendRecommendationError(
                    "recursive guarded Simulation method projection cannot be "
                    "canonicalized exactly"
                )
            code = class_attribute.__code__
            receiver_name = (
                str(code.co_varnames[0]) if code.co_argcount > 0 else "self"
            )
            receiver_attributes = _static_binding_attribute_names(
                code, receiver_name, root_fast=True
            )
            if receiver_attributes is None:
                raise BackendRecommendationError(
                    "guarded Simulation method usage cannot be projected exactly"
                )
            active.add(recursion_key)
            try:
                _validate_guarded_simulation_projection(
                    simulation, receiver_attributes, active
                )
            finally:
                active.remove(recursion_key)
            continue
        elif class_attribute is missing or hasattr(class_attribute, "__get__"):
            raise BackendRecommendationError(
                "guarded Simulation descriptor usage cannot be projected exactly"
            )
        else:
            projected_value = class_attribute

        audit = _CanonicalTraversal()
        audit.native_simulation_roots.add(id(simulation))
        audit.reject_unprojected_native_roots = True
        _canonical(projected_value, audit)


def _binding_token(
    value: Any,
    code: types.CodeType,
    binding_name: str,
    seen: set[int],
    *,
    root_fast: bool,
) -> Any:
    if isinstance(value, types.ModuleType):
        return {"module_binding": value.__name__}
    if (
        isinstance(seen, _CanonicalTraversal)
        and id(value) in seen.native_simulation_roots
        and _is_exact_trusted_meep_type(
            value, "meep.simulation", "Simulation"
        )
    ):
        attributes = _static_binding_attribute_names(
            code, binding_name, root_fast=root_fast, seen=seen
        )
        if attributes is None:
            raise BackendRecommendationError(
                "guarded Simulation root usage cannot be projected exactly"
            )
        _validate_guarded_simulation_projection(value, attributes)
        # All ordinary Simulation attributes are already present in the
        # common workload token. ``fields`` and ``structure`` are present in
        # the ordered per-rank native digest. Keep a projection description so
        # code changes cannot silently broaden which root state is consumed.
        return {
            "guarded_simulation_root_reference": True,
            "static_attributes": list(attributes),
        }
    attributes = _static_binding_attribute_names(
        code, binding_name, root_fast=root_fast, seen=seen
    )
    if attributes is not None:
        overrides: dict[str, Any] = {}
        dictionary = getattr(value, "__dict__", None)
        if isinstance(dictionary, dict) and isinstance(
            seen, _CanonicalTraversal
        ):
            for name in attributes:
                projected_value = dictionary.get(name)
                if not (
                    id(projected_value) in seen.native_simulation_roots
                    and _is_exact_trusted_meep_type(
                        projected_value,
                        "meep.simulation",
                        "Simulation",
                    )
                ):
                    continue
                nested_attributes = _static_nested_attribute_names(
                    code,
                    binding_name,
                    name,
                    root_fast=root_fast,
                    seen=seen,
                )
                if nested_attributes is None:
                    raise BackendRecommendationError(
                        "guarded Simulation root escaped nested static "
                        "callback projection"
                    )
                _validate_guarded_simulation_projection(
                    projected_value, nested_attributes
                )
                overrides[name] = {
                    "guarded_simulation_root_reference": True,
                    "static_attributes": list(nested_attributes),
                }
        projection = _projected_binding_token(
            value,
            attributes,
            seen,
            instance_value_overrides=overrides,
        )
        if projection is not None:
            return projection
    return _canonical(value, seen)


def _callable_token(
    value: Any,
    seen: set[int],
    *,
    allow_format_value: bool = False,
) -> dict[str, Any]:
    if type(value) is functools.partial:
        return {
            "partial": _canonical(value.func, seen),
            "args": _canonical(value.args, seen),
            "keywords": _canonical(value.keywords or {}, seen),
            "attributes": _canonical(getattr(value, "__dict__", {}), seen),
        }
    if isinstance(value, functools.partial):
        raise BackendRecommendationError(
            "functools.partial subclasses cannot be canonicalized exactly"
        )
    if isinstance(value, types.MethodType):
        function = value.__func__
        code = getattr(function, "__code__", None)
        receiver_name = (
            str(code.co_varnames[0])
            if code is not None and code.co_argcount > 0
            else "self"
        )
        return {
            "bound_method": _canonical(function, seen),
            "bound_self": (
                _binding_token(
                    value.__self__,
                    code,
                    receiver_name,
                    seen,
                    root_fast=True,
                )
                if code is not None
                else _canonical(value.__self__, seen)
            ),
        }
    trusted_meep_function = _is_original_trusted_meep_function(value)
    trusted_stdlib_function = (
        not trusted_meep_function
        and _is_original_trusted_stdlib_function(value, seen)
    )
    if trusted_meep_function or trusted_stdlib_function:
        package_dependency = (
            trusted_meep_function
            and value in _TRUSTED_MEEP_DEPENDENCY_GRAPH["functions"]
        )
        if package_dependency and not (
            _trusted_meep_dependency_is_unchanged(value, seen)
        ):
            raise BackendRecommendationError(
                "trusted Meep function package dependency was modified"
            )
        token = {
            "original_trusted_class_member": (
                _trusted_meep_snapshot_member_token(value, seen)
                if package_dependency
                else _shallow_function_token(
                    value, seen, transitive=False
                )
            )
        }
        if trusted_stdlib_function:
            token["trusted_mutable_dependency_state"] = (
                _trusted_stdlib_mutable_dependency_state(
                    _TRUSTED_STDLIB_DEPENDENCY_GRAPH, value
                )
            )
        if trusted_stdlib_function and not (
            _trusted_dependency_graph_is_unchanged(
                _TRUSTED_STDLIB_DEPENDENCY_GRAPH, root=value
            )
        ):
            raise BackendRecommendationError(
                "trusted standard-library function changed during "
                "canonicalization"
            )
        if package_dependency and not (
            _trusted_meep_dependency_is_unchanged(value, seen)
        ):
            raise BackendRecommendationError(
                "trusted Meep function package dependency changed during "
                "canonicalization"
            )
        return token
    function = value.__func__ if isinstance(value, types.MethodType) else value
    code = getattr(function, "__code__", None)
    token: dict[str, Any] = {
        "callable": f"{getattr(function, '__module__', '')}."
        f"{getattr(function, '__qualname__', type(function).__qualname__)}"
    }
    if code is not None:
        if type(function) is not types.FunctionType:
            raise BackendRecommendationError(
                "callables with spoofed function code cannot be canonicalized "
                "exactly"
            )
        _reject_implicit_namespace_operations(
            code,
            allow_format_value=(
                allow_format_value
                or _format_values_are_builtin_print_only(function, seen)
            ),
            seen=seen,
        )
        referenced_globals: dict[str, Any] = {}
        referenced_builtins: dict[str, Any] = {}
        module_attributes: dict[str, Any] = {}
        namespace = getattr(function, "__globals__", None)
        closure_cells = getattr(function, "__closure__", None) or ()
        closure_values = tuple(cell.cell_contents for cell in closure_cells)
        closure_bindings = dict(zip(code.co_freevars, closure_values))
        default_bindings: dict[str, Any] = {}
        positional_defaults = function.__defaults__ or ()
        keyword_defaults = function.__kwdefaults__ or {}
        if (
            type(positional_defaults) is not tuple
            or len(positional_defaults) > code.co_argcount
            or type(keyword_defaults) is not dict
        ):
            raise BackendRecommendationError(
                "function defaults cannot be canonicalized exactly"
            )
        positional_names = code.co_varnames[: code.co_argcount]
        positional_default_names = positional_names[
            len(positional_names) - len(positional_defaults) :
        ]
        default_bindings.update(
            zip(
                positional_default_names,
                positional_defaults,
            )
        )
        keyword_only_names = set(
            code.co_varnames[
                code.co_argcount :
                code.co_argcount + code.co_kwonlyargcount
            ]
        )
        if any(
            type(name) is not str or name not in keyword_only_names
            for name in keyword_defaults
        ):
            raise BackendRecommendationError(
                "function keyword defaults cannot be canonicalized exactly"
            )
        default_bindings.update(keyword_defaults)
        if isinstance(namespace, dict):
            referenced_names = _referenced_global_names(code, seen)
            referenced_globals, referenced_builtins = (
                _referenced_namespace_dependencies(
                    referenced_names,
                    namespace,
                    getattr(function, "__builtins__", None),
                    seen,
                )
            )
            module_attributes = _module_attribute_dependencies(
                code,
                namespace,
                {
                    **closure_bindings,
                    **{
                        name: default_bindings[name]
                        for name in code.co_cellvars
                        if name in default_bindings
                    },
                },
                default_bindings,
                seen,
            )
        token["code"] = {
            **_code_token(code, seen),
            "defaults": [
                (
                    _binding_token(
                        item,
                        code,
                        name,
                        seen,
                        root_fast=True,
                    )
                    if (
                        isinstance(seen, _CanonicalTraversal)
                        and id(item) in seen.native_simulation_roots
                    )
                    else _direct_binding_token(item, seen)
                )
                for name, item in zip(
                    positional_default_names, positional_defaults
                )
            ],
            "kwdefaults": {
                key: (
                    _binding_token(
                        item,
                        code,
                        key,
                        seen,
                        root_fast=True,
                    )
                    if (
                        isinstance(seen, _CanonicalTraversal)
                        and id(item) in seen.native_simulation_roots
                    )
                    else _direct_binding_token(item, seen)
                )
                for key, item in sorted(
                    keyword_defaults.items()
                )
            },
            "closure": [
                _binding_token(
                    item,
                    code,
                    name,
                    seen,
                    root_fast=False,
                )
                for name, item in zip(code.co_freevars, closure_values)
            ],
            "referenced_globals": referenced_globals,
            "referenced_builtins": referenced_builtins,
            "module_attributes": module_attributes,
        }
        token["function_metadata"] = {
            "name": _canonical(function.__name__, seen),
            "qualname": _canonical(function.__qualname__, seen),
            "module": _canonical(function.__module__, seen),
            "doc": _canonical(function.__doc__, seen),
            "type_params": _canonical(
                getattr(function, "__type_params__", ()), seen
            ),
        }
        token["function_state"] = {
            "attributes": _canonical(getattr(function, "__dict__", {}), seen),
            "annotations": _annotation_token(
                getattr(function, "__annotations__", {}), seen
            ),
        }
        token["numpy_ufunc_context"] = _numpy_ufunc_context_token(seen)
    else:
        state = _instance_state(value, seen)
        if state is None:
            raise BackendRecommendationError(
                "callable state cannot be canonicalized exactly: "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        token["callable_state"] = state
        token["callable_class"] = _class_token(type(value), seen)
        implementation = getattr(type(value), "__call__", None)
        if type(implementation) is not types.FunctionType:
            raise BackendRecommendationError(
                "callables with hidden extension state cannot be "
                "canonicalized exactly"
            )
        token["call_implementation"] = _global_dependency_token(
            implementation, seen
        )
    return token


def _canonical(value: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = _CanonicalTraversal()
    if any(
        value is unsupported
        for unsupported in _INTRINSIC_UNSUPPORTED_CALLABLE_VALUES
    ):
        raise BackendRecommendationError(
            "identity-sensitive or nondeterministic callable cannot be "
            "canonicalized exactly"
        )
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        return _float_token(value)
    if value_type is complex:
        return {"complex": [_float_token(value.real), _float_token(value.imag)]}
    if value is Ellipsis:
        return {"singleton": "builtins.Ellipsis"}
    if value is NotImplemented:
        return {"singleton": "builtins.NotImplemented"}
    if value is inspect.Signature.empty:
        return {"singleton": "inspect.Signature.empty"}
    if value_type is bytes:
        payload = bytes(value)
        return {
            "bytes": {
                "type": "bytes",
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
    if isinstance(value, (bytearray, memoryview)):
        raise BackendRecommendationError(
            "mutable buffer exporter state cannot be canonicalized exactly"
        )
    if value_type is range:
        return {"range": [value.start, value.stop, value.step]}
    if value_type is slice:
        return {
            "slice": [
                _canonical(value.start, seen),
                _canonical(value.stop, seen),
                _canonical(value.step, seen),
            ]
        }
    if isinstance(value, Path):
        if value_type is not _ORIGINAL_CONCRETE_PATH_TYPE:
            raise BackendRecommendationError(
                "Path subclasses cannot be canonicalized exactly"
            )
        return {"path": str(value)}
    if isinstance(value, types.ModuleType):
        raise BackendRecommendationError(
            "nested module state cannot be canonicalized exactly: "
            f"{value.__name__}"
        )

    if (
        isinstance(seen, _CanonicalTraversal)
        and seen.reject_unprojected_native_roots
        and id(value) in seen.native_simulation_roots
        and _is_exact_trusted_meep_type(
            value, "meep.simulation", "Simulation"
        )
    ):
        raise BackendRecommendationError(
            "guarded Simulation root escaped static callback projection"
        )

    identity = id(value)
    reference = None
    if isinstance(seen, _CanonicalTraversal):
        if identity in seen.object_references:
            if seen.object_values.get(identity) is not value:
                raise BackendRecommendationError(
                    "canonical traversal object identity was recycled"
                )
            return {"object_reference": seen.object_references[identity]}
        reference = len(seen.object_references)
        seen.object_references[identity] = reference
        seen.object_values[identity] = value
    elif identity in seen:
        return {"cycle": f"{value_type.__module__}.{value_type.__qualname__}"}
    seen.add(identity)

    def finish(token: Any) -> Any:
        if reference is None:
            return token
        return {"object_definition": reference, "value": token}

    try:
        if value_type is contextvars.ContextVar:
            current_context = contextvars.copy_context()
            current_bound = value in current_context
            missing_context_value = object()
            effective = value.get(missing_context_value)
            fresh_context = contextvars.Context()
            missing_default = object()
            try:
                default = fresh_context.run(value.get)
            except LookupError:
                default = missing_default
            numpy_ufunc_context = value is _ORIGINAL_NUMPY_UFUNC_CONTEXT
            if numpy_ufunc_context:
                module = sys.modules.get(_ORIGINAL_NUMPY_UFUNC_CONTEXT_MODULE)
                namespace = getattr(module, "__dict__", None)
                numpy_module = sys.modules.get("numpy")
                numpy_namespace = getattr(numpy_module, "__dict__", None)
                multiarray_module = sys.modules.get(
                    "numpy._core._multiarray_umath"
                )
                multiarray_namespace = getattr(
                    multiarray_module, "__dict__", None
                )
                if (
                    not isinstance(namespace, dict)
                    or namespace.get(_ORIGINAL_NUMPY_UFUNC_CONTEXT_EXPORT)
                    is not value
                    or default is not _ORIGINAL_NUMPY_UFUNC_CONTEXT_DEFAULT
                    or not isinstance(numpy_namespace, dict)
                    or numpy_namespace.get("geterr")
                    is not _ORIGINAL_NUMPY_GETERR
                    or numpy_namespace.get("geterrcall")
                    is not _ORIGINAL_NUMPY_GETERRCALL
                    or getattr(_ORIGINAL_NUMPY_GETERR, "__code__", None)
                    is not _ORIGINAL_NUMPY_GETERR_CODE
                    or getattr(_ORIGINAL_NUMPY_GETERRCALL, "__code__", None)
                    is not _ORIGINAL_NUMPY_GETERRCALL_CODE
                    or not isinstance(_ORIGINAL_NUMPY_GETERR, types.FunctionType)
                    or _ORIGINAL_NUMPY_GETERR.__globals__.get(
                        "_get_extobj_dict"
                    )
                    is not _ORIGINAL_NUMPY_GET_EXTOBJ_DICT
                    or not isinstance(
                        _ORIGINAL_NUMPY_GETERRCALL, types.FunctionType
                    )
                    or _ORIGINAL_NUMPY_GETERRCALL.__globals__.get(
                        "_get_extobj_dict"
                    )
                    is not _ORIGINAL_NUMPY_GET_EXTOBJ_DICT
                    or namespace.get("_get_extobj_dict")
                    is not _ORIGINAL_NUMPY_GET_EXTOBJ_DICT
                    or not isinstance(multiarray_namespace, dict)
                    or multiarray_namespace.get("_get_extobj_dict")
                    is not _ORIGINAL_NUMPY_GET_EXTOBJ_DICT
                    or getattr(
                        _ORIGINAL_NUMPY_GET_EXTOBJ_DICT, "__code__", None
                    )
                    is not _ORIGINAL_NUMPY_GET_EXTOBJ_DICT_CODE
                ):
                    raise BackendRecommendationError(
                        "NumPy ufunc context differs from its import-time "
                        "dependency"
                    )

            def context_item_token(item: Any) -> Any:
                if (
                    numpy_ufunc_context
                    and item is _ORIGINAL_NUMPY_UFUNC_CONTEXT_DEFAULT
                ):
                    return {"numpy_ufunc_extobj": "import-time-default"}
                if numpy_ufunc_context:
                    return {"numpy_ufunc_extobj": "current-context-value"}
                return _canonical(item, seen)

            numpy_semantics = None
            if numpy_ufunc_context:
                try:
                    numpy_semantics = {
                        "extobj": _canonical(
                            _ORIGINAL_NUMPY_GET_EXTOBJ_DICT(), seen
                        )
                    }
                except Exception as error:
                    raise BackendRecommendationError(
                        "NumPy ufunc error context cannot be canonicalized "
                        "exactly"
                    ) from error

            return finish(
                {
                    "context_var": {
                        "name": value.name,
                        "current_context_bound": current_bound,
                        "has_effective_value": (
                            effective is not missing_context_value
                        ),
                        "effective_value": (
                            None
                            if effective is missing_context_value
                            else context_item_token(effective)
                        ),
                        "has_default": default is not missing_default,
                        "default": (
                            None
                            if default is missing_default
                            else context_item_token(default)
                        ),
                        "numpy_semantics": numpy_semantics,
                    }
                }
            )
        if value_type is type(re.compile("")):
            if int(value.flags) & int(re.LOCALE):
                raise BackendRecommendationError(
                    "locale-dependent regular expressions cannot be "
                    "canonicalized exactly"
                )
            return finish(
                {
                    "regular_expression": {
                        "pattern": _canonical(value.pattern, seen),
                        "flags": int(value.flags),
                    }
                }
            )
        if isinstance(value, type):
            return finish(_class_token(value, seen))
        if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            return finish(_builtin_callable_token(value, seen))
        if isinstance(value, _IMMUTABLE_DESCRIPTOR_TYPES):
            return finish(_builtin_callable_token(value, seen))
        if isinstance(value, enum.Enum):
            return finish(
                {
                    "enum_class": _class_token(value_type, seen),
                    "name": value.name,
                    "value": _canonical(value.value, seen),
                }
            )
        if value_type is inspect.Parameter:
            return finish(
                {
                    "inspect_parameter": {
                        "name": value.name,
                        "kind": value.kind.name,
                        "default": _canonical(value.default, seen),
                        "annotation": _canonical(value.annotation, seen),
                    }
                }
            )
        if value_type is inspect.Signature:
            return finish(
                {
                    "inspect_signature": {
                        "parameters": [
                            {
                                "name": name,
                                "parameter": _canonical(parameter, seen),
                            }
                            for name, parameter in value.parameters.items()
                        ],
                        "return_annotation": _canonical(
                            value.return_annotation, seen
                        ),
                    }
                }
            )
        if _is_exact_trusted_meep_type(value, "meep", "dft_ldos"):
            # The public simulation.dft_ldos factory closes over this SWIG
            # proxy. Pointer addresses in ``this`` are rank/process-local and
            # have no workload semantics, while the frequency vector is the
            # immutable user input which determines the LDOS work. Accept
            # this projection only for the exact package-generated class with
            # every original descriptor still installed.
            attributes = getattr(value, "__dict__", None)
            members = _TRUSTED_MEEP_CLASS_MEMBERS.get(value_type, {})
            frequency_property = members.get("freq")
            if (
                not isinstance(attributes, dict)
                or "this" not in attributes
                or set(value_type.__dict__) != set(members)
                or any(
                    value_type.__dict__.get(name) is not member
                    for name, member in members.items()
                )
                or type(frequency_property) is not property
                or frequency_property.fget is None
            ):
                raise BackendRecommendationError(
                    "Meep dft_ldos Python proxy state is malformed"
                )
            try:
                frequencies = [
                    float(item)
                    for item in frequency_property.__get__(value, value_type)
                ]
            except Exception as error:
                raise BackendRecommendationError(
                    "Meep dft_ldos frequencies cannot be canonicalized exactly"
                ) from error
            python_state = {
                key: item for key, item in attributes.items() if key != "this"
            }
            return finish(
                {
                    "meep_dft_ldos": {
                        "frequencies": _canonical(frequencies, seen),
                        "python_state": _canonical(python_state, seen),
                    }
                }
            )
        if _is_exact_trusted_meep_type(value, "meep", "sourcedata"):
            try:
                sourcedata_manifest = json.loads(value.gpmeep_manifest())
            except Exception as error:
                raise BackendRecommendationError(
                    "Meep sourcedata cannot be canonicalized exactly"
                ) from error
            if (
                not isinstance(sourcedata_manifest, dict)
                or sourcedata_manifest.get("schema") != "gpmeep-sourcedata-v1"
            ):
                raise BackendRecommendationError(
                    "Meep sourcedata manifest is malformed"
                )
            attributes = getattr(value, "__dict__", None)
            if not isinstance(attributes, dict) or "this" not in attributes:
                raise BackendRecommendationError(
                    "Meep sourcedata Python proxy state is malformed"
                )
            python_state = {
                key: item for key, item in attributes.items() if key != "this"
            }
            return finish(
                {
                    "meep_sourcedata": sourcedata_manifest,
                    "python_state": _canonical(python_state, seen),
                }
            )
        try:
            import numpy as np

            if (
                np.ndarray is not _ORIGINAL_NUMPY_NDARRAY_TYPE
                or np.dtype is not _ORIGINAL_NUMPY_DTYPE_TYPE
                or np.generic is not _ORIGINAL_NUMPY_GENERIC_TYPE
                or np.ufunc is not _ORIGINAL_NUMPY_UFUNC_TYPE
                or np.ndarray.tobytes is not _ORIGINAL_NUMPY_NDARRAY_TOBYTES
                or np.ndarray.tolist is not _ORIGINAL_NUMPY_NDARRAY_TOLIST
            ):
                raise BackendRecommendationError(
                    "NumPy primitive types differ from import-time dependencies"
                )

            if (
                _ORIGINAL_NUMPY_GENERATOR_TYPE is not None
                and _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE is not None
                and (
                    isinstance(value, _ORIGINAL_NUMPY_GENERATOR_TYPE)
                    or isinstance(value, _ORIGINAL_NUMPY_BIT_GENERATOR_TYPE)
                    or isinstance(value, _ORIGINAL_NUMPY_SEED_SEQUENCE_TYPE)
                )
            ):
                return finish(_numpy_rng_token(np, value, seen))

            if value_type is _ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHER_TYPE:
                original = _ORIGINAL_NUMPY_ARRAY_FUNCTION_DISPATCHERS.get(
                    id(value)
                )
                implementation = getattr(value, "_implementation", None)
                attributes = getattr(value, "__dict__", None)
                if (
                    original is None
                    or original["dispatcher"] is not value
                    or original["implementation"] is not implementation
                    or not isinstance(attributes, dict)
                ):
                    raise BackendRecommendationError(
                        "NumPy array-function dispatcher is not an original "
                        "import-time dependency"
                    )
                for module_name, export_name in original["exports"]:
                    module = sys.modules.get(module_name)
                    namespace = getattr(module, "__dict__", None)
                    if (
                        not isinstance(namespace, dict)
                        or namespace.get(export_name) is not value
                    ):
                        raise BackendRecommendationError(
                            "NumPy array-function dispatcher export differs "
                            "from its import-time dependency"
                        )
                hidden_token = None
                if isinstance(implementation, types.FunctionType):
                    if (
                        implementation.__code__
                        is not original["implementation_code"]
                        or implementation.__defaults__
                        is not original["implementation_defaults"]
                        or implementation.__kwdefaults__
                        is not original["implementation_kwdefaults"]
                        or implementation.__closure__
                        is not original["implementation_closure"]
                    ):
                        raise BackendRecommendationError(
                            "NumPy array-function implementation differs "
                            "from its import-time dependency"
                        )
                    hidden = original["hidden"]
                    if hidden is None:
                        hidden_token = {
                            "implicit_dispatcher": "finalize_array_function_like"
                        }
                    else:
                        if (
                            implementation.__globals__.get(hidden["name"])
                            is not hidden["function"]
                            or hidden["function"].__code__ is not hidden["code"]
                            or hidden["function"].__defaults__
                            is not hidden["defaults"]
                            or hidden["function"].__kwdefaults__
                            is not hidden["kwdefaults"]
                            or hidden["function"].__closure__
                            is not hidden["closure"]
                        ):
                            raise BackendRecommendationError(
                                "NumPy array-function hidden dispatcher differs "
                                "from its import-time dependency: "
                                f"{original['exports'][0][0]}."
                                f"{original['exports'][0][1]}"
                            )
                        hidden_token = {
                            "name": hidden["name"],
                            "function": _canonical(hidden["function"], seen),
                        }
                return finish(
                    {
                        "numpy_array_function_dispatcher": {
                            "exports": [
                                f"{module_name}.{export_name}"
                                for module_name, export_name in original["exports"]
                            ],
                            "numpy_version": _ORIGINAL_NUMPY_VERSION,
                            "implementation": _canonical(
                                implementation, seen
                            ),
                            "hidden_dispatcher": hidden_token,
                            "public_state": _canonical(attributes, seen),
                        }
                    }
                )
            if value_type is _ORIGINAL_NUMPY_NDARRAY_TYPE:
                return finish({
                    "ndarray": _numpy_array_token(np, value, seen)
                })
            if isinstance(value, _ORIGINAL_NUMPY_UFUNC_TYPE):
                original_ufunc = _ORIGINAL_NUMPY_UFUNCS.get(id(value))
                if (
                    original_ufunc is None
                    or original_ufunc["ufunc"] is not value
                ):
                    raise BackendRecommendationError(
                        "NumPy ufunc is not an original import-time dependency"
                    )
                for module_name, export_name in original_ufunc["exports"]:
                    module = sys.modules.get(module_name)
                    namespace = getattr(module, "__dict__", None)
                    if (
                        not isinstance(namespace, dict)
                        or namespace.get(export_name) is not value
                    ):
                        raise BackendRecommendationError(
                            "NumPy ufunc export differs from its import-time "
                            f"dependency: {module_name}.{export_name}"
                        )
                return finish({
                    "numpy_ufunc": {
                        "exports": [
                            f"{module_name}.{export_name}"
                            for module_name, export_name in original_ufunc[
                                "exports"
                            ]
                        ],
                        "name": value.__name__,
                        "nin": value.nin,
                        "nout": value.nout,
                        "nargs": value.nargs,
                        "identity": _canonical(value.identity, seen),
                        "signature": value.signature,
                        "types": list(value.types),
                        "error_context": (
                            _numpy_ufunc_context_token(seen)
                        ),
                    }
                })
            if isinstance(value, _ORIGINAL_NUMPY_GENERIC_TYPE):
                if (
                    value_type not in _ORIGINAL_NUMPY_BUILTIN_SCALAR_TYPES
                    or value_type is not value.dtype.type
                ):
                    raise BackendRecommendationError(
                        "NumPy scalar subclasses cannot be canonicalized "
                        "exactly"
                    )
                dtype_token = _numpy_dtype_token(np, value.dtype, seen)
                if value.dtype.hasobject:
                    scalar_value = {
                        "object_value": _canonical(value.item(), seen)
                    }
                else:
                    tobytes = _ORIGINAL_NUMPY_SCALAR_TOBYTES.get(value_type)
                    if (
                        tobytes is None
                        or getattr(value_type, "tobytes", None) is not tobytes
                    ):
                        raise BackendRecommendationError(
                            "NumPy scalar byte serializer differs from its "
                            "import-time dependency"
                        )
                    payload = tobytes(value)
                    scalar_value = {
                        "raw_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                return finish({
                    "numpy_scalar": scalar_value,
                    "dtype": dtype_token,
                })
            if isinstance(value, _ORIGINAL_NUMPY_NDARRAY_TYPE):
                raise BackendRecommendationError(
                    "ndarray subclasses cannot be canonicalized exactly"
                )
        except ImportError:
            pass
        if isinstance(value, dict):
            if value_type is not dict:
                raise BackendRecommendationError(
                    "dict subclasses cannot be canonicalized exactly"
                )
            items = [
                (_canonical(key, seen), _canonical(item, seen))
                for key, item in dict.items(value)
            ]
            return finish({"mapping": items})
        if isinstance(value, collections.abc.Mapping):
            raise BackendRecommendationError(
                "mapping views and non-dict mappings cannot be "
                "canonicalized exactly: "
                f"{value_type.__module__}.{value_type.__qualname__}"
            )
        if isinstance(value, (list, tuple)):
            named_tuple_fields = _namedtuple_class_fields(value_type)
            named_tuple = (
                isinstance(value, tuple)
                and named_tuple_fields is not None
                and len(named_tuple_fields) == len(value)
                and tuple.__iter__ is getattr(value_type, "__iter__", None)
            )
            if value_type is list:
                iterator = list.__iter__(value)
            elif value_type is tuple:
                iterator = tuple.__iter__(value)
            elif named_tuple:
                iterator = tuple.__iter__(value)
            else:
                raise BackendRecommendationError(
                    "sequence subclasses cannot be canonicalized exactly"
                )
            token = {
                "sequence": type(value).__name__,
                "items": [_canonical(item, seen) for item in iterator],
            }
            if value_type not in (list, tuple):
                state = (
                    {"namedtuple_fields": list(named_tuple_fields)}
                    if named_tuple
                    else _instance_state(value, seen)
                )
                if state is None:
                    raise BackendRecommendationError(
                        "sequence subclass state cannot be canonicalized exactly: "
                        f"{value_type.__module__}.{value_type.__qualname__}"
                    )
                token.update(
                    {
                        "subclass": _class_token(value_type, seen),
                        "subclass_state": state,
                    }
                )
            return finish(token)
        if isinstance(value, (set, frozenset)):
            if value_type is set:
                iterator = set.__iter__(value)
            elif value_type is frozenset:
                iterator = frozenset.__iter__(value)
            else:
                raise BackendRecommendationError(
                    "set subclasses cannot be canonicalized exactly"
                )
            items = [_canonical(item, seen) for item in iterator]
            token = {
                "set": {
                    "type": value_type.__name__,
                    "items": items,
                }
            }
            return finish(token)
        if isinstance(value, collections.deque):
            if value_type is not collections.deque:
                raise BackendRecommendationError(
                    "deque subclasses cannot be canonicalized exactly"
                )
            token = {
                "deque": [
                    _canonical(item, seen)
                    for item in collections.deque.__iter__(value)
                ],
                "maxlen": value.maxlen,
            }
            return finish(token)
        if isinstance(value, (int, float, complex, str, bytes, bytearray, memoryview)):
            raise BackendRecommendationError(
                "primitive subclasses cannot be canonicalized exactly"
            )
        if isinstance(value, types.GeneratorType):
            _reject_implicit_namespace_operations(value.gi_code, seen=seen)
            generator_state = inspect.getgeneratorstate(value)
            if generator_state not in (
                inspect.GEN_CREATED,
                inspect.GEN_CLOSED,
            ):
                raise BackendRecommendationError(
                    "suspended or running generator state cannot be "
                    "canonicalized exactly"
                )
            frame = value.gi_frame
            generator_globals: dict[str, Any] = {}
            generator_builtins: dict[str, Any] = {}
            generator_modules: dict[str, Any] = {}
            if frame is not None:
                referenced_names = _referenced_global_names(value.gi_code, seen)
                generator_globals, generator_builtins = (
                    _referenced_namespace_dependencies(
                        referenced_names,
                        frame.f_globals,
                        frame.f_builtins,
                        seen,
                    )
                )
                generator_modules = _module_attribute_dependencies(
                    value.gi_code,
                    frame.f_globals,
                    frame.f_locals,
                    frame.f_locals,
                    seen,
                )
            return finish({
                "generator": {
                    "state": generator_state,
                    "code": _code_token(value.gi_code, seen),
                    "lasti": None if frame is None else frame.f_lasti,
                    "lineno": None if frame is None else frame.f_lineno,
                    "locals": (
                        None if frame is None else _canonical(frame.f_locals, seen)
                    ),
                    "referenced_globals": generator_globals,
                    "referenced_builtins": generator_builtins,
                    "module_attributes": generator_modules,
                }
            })
        if hasattr(value, "__next__"):
            iterator_type = type(value)
            if iterator_type in _SAFE_BUILTIN_ITERATOR_TYPES:
                try:
                    reduced = value.__reduce__()
                except Exception as error:
                    raise BackendRecommendationError(
                        "iterator state cannot be canonicalized exactly"
                    ) from error
                return finish({
                    "iterator": f"{type(value).__module__}.{type(value).__qualname__}",
                    "state": _canonical(reduced, seen),
                })
            raise BackendRecommendationError(
                "iterator type cannot be canonicalized exactly"
            )
        if callable(value):
            if isinstance(seen, _CanonicalTraversal):
                seen.callable_dependency_count += 1
                if seen.callable_dependency_count > MAXIMUM_CALLABLE_DEPENDENCIES:
                    raise BackendRecommendationError(
                        "callable dependency graph exceeds the fail-closed limit"
                    )
            return finish(_callable_token(value, seen))
        state = _instance_state(value, seen)
        if state is None:
            raise BackendRecommendationError(
                "object state cannot be canonicalized exactly: "
                f"{value_type.__module__}.{value_type.__qualname__}"
            )
        return finish(
            {
                "object": f"{value_type.__module__}.{value_type.__qualname__}",
                "class_state": _class_token(value_type, seen),
                "state": state,
            }
        )
    finally:
        seen.remove(identity)


def _collective_records(
    mp: Any, local: dict[str, Any], *, verify_count: bool = True
) -> list[dict[str, Any]]:
    serialized = json.dumps(
        local, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if len(serialized.encode("utf-8")) >= 4096:
        raise BackendRecommendationError(
            "recommendation collective record exceeds 4095 UTF-8 bytes"
        )
    raw_records = mp._gpu_collective_rank_records(serialized)
    try:
        records = [json.loads(record) for record in raw_records]
    except (TypeError, ValueError) as error:
        raise BackendRecommendationError(
            "recommendation collective returned malformed rank data"
        ) from error
    if not records:
        raise BackendRecommendationError(
            "recommendation collective returned no rank data"
        )
    if verify_count and len(records) != int(mp.count_processors()):
        raise BackendRecommendationError(
            "recommendation collective rank count changed"
        )
    return records


def _positive_grid_extent(value: Any, resolution: float) -> int:
    numeric = abs(float(value))
    if not math.isfinite(numeric):
        raise BackendRecommendationError(
            "simulation cell extents must be finite"
        )
    return max(1, int(math.ceil(numeric * resolution))) if numeric > 0 else 1


def _global_cells(simulation: Any) -> int:
    resolution = float(simulation.resolution)
    if not math.isfinite(resolution) or resolution <= 0:
        raise BackendRecommendationError("simulation resolution must be positive")
    size = simulation.cell_size
    raw_components = [
        float(getattr(size, "x", 0)),
        float(getattr(size, "y", 0)),
        float(getattr(size, "z", 0)),
    ]
    if not all(math.isfinite(component) for component in raw_components):
        raise BackendRecommendationError("simulation cell extents must be finite")
    dimensions = int(simulation.dimensions)
    if dimensions == 1:
        extents = [raw_components[2]]
    elif dimensions == -2:
        extents = [raw_components[0], raw_components[2]]
    elif dimensions == 2:
        extents = raw_components[:2]
    else:
        extents = [component for component in raw_components if abs(component) > 0]
    if not extents:
        # A zero-volume Meep cell is a supported zero-dimensional simulation,
        # notably for run_k_points material-dispersion calculations.  It owns
        # one spatial sample rather than an invalid/empty domain.
        extents = [0.0]
    cells = 1
    try:
        for extent in extents:
            cells *= _positive_grid_extent(extent, resolution)
    except (OverflowError, ValueError) as error:
        raise BackendRecommendationError(
            "simulation cell extent and resolution product is out of range"
        ) from error
    return cells


def _numeric_condition_time(condition: Any) -> float | None:
    if isinstance(condition, bool):
        return None
    if isinstance(condition, numbers.Number):
        value = float(condition)
        return value if math.isfinite(value) and value >= 0 else None
    if isinstance(condition, (list, tuple)):
        candidates = [
            value
            for item in condition
            if (value := _numeric_condition_time(item)) is not None
        ]
        return min(candidates) if candidates else None
    return None


def _source_end_time(source: Any) -> float | None:
    time_source = getattr(source, "src", None)
    if time_source is None:
        return 0.0
    if hasattr(time_source, "end_time"):
        value = float(time_source.end_time)
    elif all(hasattr(time_source, key) for key in ("start_time", "width", "cutoff")):
        value = float(time_source.start_time) + 2.0 * float(time_source.width) * float(
            time_source.cutoff
        )
    else:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _run_length(simulation: Any, until: Any, until_after_sources: Any) -> dict[str, Any]:
    condition = until_after_sources if until_after_sources is not None else until
    additional_time = _numeric_condition_time(condition)
    kind = "until_after_sources" if until_after_sources is not None else "until"
    duration = additional_time
    source_end = None
    if until_after_sources is not None and additional_time is not None:
        ends = [_source_end_time(source) for source in simulation.sources]
        if ends and all(value is not None for value in ends):
            source_end = max(ends)
            duration = source_end + additional_time
        elif not ends:
            source_end = 0.0
    courant = float(simulation.Courant)
    resolution = float(simulation.resolution)
    expected_timesteps = None
    if duration is not None and math.isfinite(courant) and courant > 0:
        expected_timesteps = int(math.ceil(duration * resolution / courant))
    return {
        "kind": kind,
        "numeric_duration": duration,
        "source_end_time": source_end,
        "expected_timesteps": expected_timesteps,
        "dynamic_condition": additional_time is None,
    }


def _monitor_frequency_count(simulation: Any) -> int:
    total = 0
    for monitor in getattr(simulation, "dft_objects", ()):
        args = getattr(monitor, "args", ())
        candidates = []
        if isinstance(args, (list, tuple)):
            candidates.extend(args)
        count = 1
        for candidate in candidates:
            if hasattr(candidate, "size") and not isinstance(candidate, (str, bytes)):
                try:
                    count = max(count, int(candidate.size))
                except (TypeError, ValueError):
                    pass
        total += count
    return total


def _invalidate_device_inventory_cache(controller: Any | None = None) -> None:
    """Forget the guard-only positive device inventory snapshot.

    Public ``mp.gpu.devices()`` remains an uncached discovery API.  This
    process-local cache only removes repeated discovery from the backend
    recommendation guard and is deliberately discarded across fork and every
    explicit backend/device transition.
    """
    global _DEVICE_INVENTORY_CACHE
    _ensure_process_local_state()
    with _LOCK:
        cache = _DEVICE_INVENTORY_CACHE
        if controller is None or cache is None or cache.get("controller") is controller:
            _DEVICE_INVENTORY_CACHE = None


def _run_gpu_controller_transition(
    controller: Any, transition: Any, argument: Any
) -> Any:
    """Serialize a public native GPU transition with guard discovery.

    The native generation token detects private/out-of-band changes, while
    this critical section makes the supported ``mp.gpu`` mutators atomic with
    respect to the recommendation guard's positive inventory cache.
    """
    global _DEVICE_INVENTORY_CACHE
    _ensure_process_local_state()
    if not callable(transition):
        raise TypeError("CUDA controller transition is not callable")
    with _LOCK:
        try:
            return transition(argument)
        finally:
            cache = _DEVICE_INVENTORY_CACHE
            if (
                cache is None
                or cache.get("controller") is controller
            ):
                _DEVICE_INVENTORY_CACHE = None


def _device_inventory_cache_key(
    mp: Any,
    owner: Any | None,
) -> tuple[tuple[Any, ...], Any, Any, Any] | None:
    """Return a stable positive-cache key and strong identity anchors.

    Third-party/fake controllers without the private native generation token
    retain the old uncached behavior.  The production controller exposes that
    token read-only, making direct native backend/device changes observable.
    """
    if owner is None:
        return None
    try:
        weakref.ref(owner)
    except TypeError:
        return None
    controller = mp.gpu
    generation = getattr(controller, "_backend_generation", None)
    if generation is None:
        return None
    if type(generation) is not int or not 0 <= generation <= (1 << 64) - 1:
        raise ValueError("CUDA backend generation is malformed")
    devices_callable = controller.devices
    callable_owner = getattr(devices_callable, "__self__", None)
    callable_implementation = getattr(
        devices_callable, "__func__", devices_callable
    )
    requested_backend = str(controller.requested_backend)
    active_backend = str(getattr(controller, "active_backend", requested_backend))
    selected_ordinal = int(controller.selected_device)
    selected_identifier = str(controller.selected_device_identifier)
    environment = tuple(
        (name, os.environ.get(name))
        for name in _DEVICE_INVENTORY_ENVIRONMENT_NAMES
    )
    topology = (
        int(mp.count_processors()),
        int(mp.my_rank()),
        int(mp.my_global_rank()),
        int(mp.my_node_rank()),
        int(mp.count_node_processors()),
    )
    key = (
        os.getpid(),
        _PROCESS_NONCE,
        id(controller),
        id(callable_owner),
        id(callable_implementation),
        id(owner),
        generation,
        requested_backend,
        active_backend,
        selected_ordinal,
        selected_identifier,
        environment,
        topology,
    )
    return key, controller, callable_owner, callable_implementation


def _normalize_compatible_device_inventory(
    devices: Any,
) -> tuple[tuple[int, str, int, int], ...]:
    if type(devices) not in (list, tuple):
        raise ValueError("CUDA device inventory is not a list")
    normalized: list[tuple[int, str, int, int]] = []
    ordinals: set[int] = set()
    identifiers: set[str] = set()
    for device in devices:
        if type(device) is not dict:
            raise ValueError("CUDA device inventory entry is not a dictionary")
        if not bool(device.get("compatible")):
            continue
        ordinal = device.get("ordinal")
        identifier = device.get("identifier")
        memory = device.get("global_memory_bytes")
        bandwidth = device.get("memory_bandwidth_bytes_per_second", 0)
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("compatible CUDA device has an invalid ordinal")
        if type(identifier) is not str or not identifier:
            raise ValueError("compatible CUDA device has no stable identifier")
        try:
            encoded_identifier = identifier.encode("utf-8", "strict")
        except UnicodeError as error:
            raise ValueError(
                "compatible CUDA device identifier is not valid UTF-8"
            ) from error
        if len(encoded_identifier) > 256 or "\x00" in identifier:
            raise ValueError("compatible CUDA device identifier is malformed")
        if type(memory) is not int or memory <= 0:
            raise ValueError(
                "compatible CUDA device has no positive memory capacity"
            )
        if type(bandwidth) is not int or bandwidth < 0:
            raise ValueError(
                "compatible CUDA device has invalid memory bandwidth"
            )
        if ordinal in ordinals:
            raise ValueError("compatible CUDA device ordinal is not unique")
        if identifier in identifiers:
            raise ValueError("compatible CUDA device identifier is not unique")
        ordinals.add(ordinal)
        identifiers.add(identifier)
        normalized.append((ordinal, identifier, memory, bandwidth))
    return tuple(normalized)


def _materialize_compatible_device_inventory(
    inventory: tuple[tuple[int, str, int, int], ...],
) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": ordinal,
            "compatible": True,
            "identifier": identifier,
            "global_memory_bytes": memory,
            "memory_bandwidth_bytes_per_second": bandwidth,
        }
        for ordinal, identifier, memory, bandwidth in inventory
    ]


def _compatible_device_inventory_unlocked(
    mp: Any, owner: Any | None
) -> list[dict[str, Any]]:
    global _DEVICE_INVENTORY_CACHE
    cache_key = _device_inventory_cache_key(mp, owner)
    if cache_key is None:
        return _materialize_compatible_device_inventory(
            _normalize_compatible_device_inventory(mp.gpu.devices())
        )
    key, controller, callable_owner, callable_implementation = cache_key
    cache = _DEVICE_INVENTORY_CACHE
    if (
        type(cache) is dict
        and cache.get("key") == key
        and cache.get("controller") is controller
        and cache.get("callable_owner") is callable_owner
        and cache.get("callable_implementation") is callable_implementation
        and type(cache.get("owner_ref")) is weakref.ReferenceType
        and cache.get("owner_ref")() is owner
        and type(cache.get("inventory")) is tuple
    ):
        # Re-read native state even on a hit.  This closes the interval in
        # which an out-of-band generation/device transition could otherwise
        # make the first key stale before the cached value is returned.
        after = _device_inventory_cache_key(mp, owner)
        if after is not None:
            after_key, after_controller, after_owner, after_implementation = after
            if (
                after_key == key
                and after_controller is controller
                and after_owner is callable_owner
                and after_implementation is callable_implementation
            ):
                return _materialize_compatible_device_inventory(
                    cache["inventory"]
                )

    # A changed key invalidates the old snapshot before discovery.  Failures
    # and empty inventories are intentionally never negative-cached.
    _DEVICE_INVENTORY_CACHE = None
    inventory = _normalize_compatible_device_inventory(controller.devices())
    after = _device_inventory_cache_key(mp, owner)
    if inventory and after is not None:
        after_key, after_controller, after_owner, after_implementation = after
        if (
            after_key == key
            and after_controller is controller
            and after_owner is callable_owner
            and after_implementation is callable_implementation
        ):
            _DEVICE_INVENTORY_CACHE = {
                "key": key,
                "controller": controller,
                "callable_owner": callable_owner,
                "callable_implementation": callable_implementation,
                "owner_ref": weakref.ref(owner),
                "inventory": inventory,
            }
    return _materialize_compatible_device_inventory(inventory)


def _compatible_device_inventory(
    mp: Any, owner: Any | None = None
) -> list[dict[str, Any]]:
    """Return a positive snapshot scoped to one live Simulation owner."""
    _ensure_process_local_state()
    with _LOCK:
        return _compatible_device_inventory_unlocked(mp, owner)


def _selected_device_record(
    mp: Any,
    simulation: Any,
    local_cells: int,
    size_eligible: bool,
    selected_backend: str,
) -> dict[str, Any]:
    estimated_required_memory = (
        local_cells * BYTES_PER_LOCAL_CELL + FIXED_DEVICE_BYTES
    )
    record: dict[str, Any] = {
        "queried": False,
        "available": False,
        "identifier": "",
        "ordinal": -1,
        "global_memory_bytes": 0,
        "memory_bandwidth_bytes_per_second": 0,
        "estimated_required_memory_bytes": estimated_required_memory,
        "diagnostic": "host-only crossover floor",
    }
    if not size_eligible and selected_backend != "cuda":
        return record
    record["queried"] = True
    try:
        devices = _compatible_device_inventory(mp, simulation)
        if not devices:
            record["diagnostic"] = _safe_bounded_text(
                mp.gpu.runtime_diagnostic, 256
            )
            return record
        if selected_backend == "cuda":
            ordinal = int(mp.gpu.selected_device)
            matches = [
                device
                for device in devices
                if int(device["ordinal"]) == ordinal
            ]
            if len(matches) != 1:
                raise ValueError(
                    "active CUDA backend has no unique compatible device"
                )
            device = matches[0]
            actual_identifier = str(mp.gpu.selected_device_identifier)
            if actual_identifier != str(device.get("identifier", "")):
                raise ValueError(
                    "active CUDA device identifier differs from inventory"
                )
        elif (explicit := os.environ.get("MEEP_GPU_DEVICE")) not in (None, ""):
            if not explicit.isdecimal():
                raise ValueError("MEEP_GPU_DEVICE is not a non-negative integer")
            ordinal = int(explicit)
            matches = [device for device in devices if int(device["ordinal"]) == ordinal]
            if not matches:
                raise ValueError("MEEP_GPU_DEVICE does not name a compatible device")
            device = matches[0]
        else:
            node_rank = int(mp.my_node_rank())
            device = devices[node_rank % len(devices)]
        identifier = _safe_bounded_text(device.get("identifier", ""), 256)
        global_memory_bytes = int(device["global_memory_bytes"])
        if not identifier:
            raise ValueError("compatible CUDA device has no stable identifier")
        if global_memory_bytes <= 0:
            raise ValueError("compatible CUDA device has no positive memory capacity")
        record.update(
            {
                "available": True,
                "identifier": identifier,
                "ordinal": int(device["ordinal"]),
                "global_memory_bytes": global_memory_bytes,
                "memory_bandwidth_bytes_per_second": int(
                    device.get("memory_bandwidth_bytes_per_second", 0)
                ),
                "diagnostic": "compatible CUDA device observed",
            }
        )
    except Exception as error:
        record["diagnostic"] = (
            "device discovery failure: " + _safe_bounded_text(error, 256)
        )
    return record


def _workload_record(
    simulation: Any,
    until: Any,
    until_after_sources: Any,
    step_functions: Any,
    operation: Any,
) -> dict[str, Any]:
    traversal = _CanonicalTraversal(
        memoize_trusted_meep_dependency_graph=True
    )
    traversal.native_simulation_roots.add(id(simulation))
    # Native fields state may legitimately differ by rank (for example an
    # IndexedSource owns rank-local chunks).  guard_run binds its ordered
    # per-rank digest separately while still requiring all semantic Python
    # workload inputs to match across ranks.
    inputs = _simulation_workload_state(simulation, include_native_state=False)
    simulation_token = _canonical(inputs, traversal)
    simulation_class_token = _class_token(type(simulation), traversal)
    numpy_context_token = _numpy_ufunc_context_token(traversal)
    # Callback/default/container traversal must never fall back to the root's
    # reduced instance token. Direct closure/default bindings are accepted
    # only through the static root projection in ``_binding_token``.
    traversal.reject_unprojected_native_roots = True
    result = {
        "simulation": simulation_token,
        # Bind the guarded root class explicitly even when its instance has no
        # DFT wrapper or callback that would otherwise lead traversal back to
        # the class.  Import-time Meep members stay shallow; any added or
        # replaced member is transitive by identity and therefore cannot hide
        # behind an unchanged Simulation instance dictionary.
        "simulation_class": simulation_class_token,
        "numpy_ufunc_context": numpy_context_token,
        "run": {
            "until": _canonical(until, traversal),
            "until_after_sources": _canonical(until_after_sources, traversal),
            "step_functions": _canonical(step_functions, traversal),
            "operation": _canonical(operation, traversal),
        },
    }
    _finalize_trusted_meep_dependency_integrity(traversal, "workload")
    return result


def _policy_facts(rank_records: list[dict[str, Any]]) -> dict[str, Any]:
    size_eligible = all(record["size_eligible"] for record in rank_records)
    run_observations = [record["expected_timesteps"] for record in rank_records]
    known_run = all(value is not None for value in run_observations)
    run_eligible = not known_run or all(
        value >= MINIMUM_EXPECTED_TIMESTEPS for value in run_observations
    )
    devices_available = all(record["device"]["available"] for record in rank_records)
    device_identifiers = [
        record["device"]["identifier"]
        for record in rank_records
        if record["device"]["available"]
    ]
    device_assignment_unique = len(device_identifiers) == len(set(device_identifiers))
    memory_eligible = all(
        record["device"].get("estimated_required_memory_bytes", 2**63)
        <= int(record["device"]["global_memory_bytes"] * DEVICE_CAPACITY_FRACTION)
        for record in rank_records
    )
    monitor_heavy = any(
        record["monitor_count"] >= 64
        and record["rank_local_cells"] < 4 * MINIMUM_RANK_LOCAL_CELLS
        for record in rank_records
    )
    devices_queried = any(
        record["device"]["queried"] for record in rank_records
    )
    return {
        "size_eligible": size_eligible,
        "known_run": known_run,
        "run_eligible": run_eligible,
        "devices_queried": devices_queried,
        "devices_available": devices_available,
        "device_assignment_unique": device_assignment_unique,
        "memory_eligible": memory_eligible,
        "monitor_heavy": monitor_heavy,
    }


def _policy_decision(rank_records: list[dict[str, Any]]) -> tuple[str, str, str]:
    facts = _policy_facts(rank_records)
    if facts["devices_queried"] and (
        not facts["devices_available"]
        or not facts["memory_eligible"]
        or not facts["device_assignment_unique"]
    ):
        return "cpu", "environment", "high"
    if (
        not facts["size_eligible"]
        or not facts["run_eligible"]
        or facts["monitor_heavy"]
    ):
        reason = "hardware_crossover"
        confidence = "high" if facts["known_run"] else "medium"
        return "cpu", reason, confidence
    return (
        "cuda",
        "hardware_crossover",
        "medium" if not facts["known_run"] else "high",
    )


def _acknowledgement_scope_digest(
    rank_records: list[dict[str, Any]],
    selected: str,
    recommended: str,
    reason_class: str,
    confidence: str,
    topology_digest: str,
) -> str:
    """Bind reusable acknowledgement to policy-relevant, not object, state."""
    policy_inputs = [
        {
            "rank": record["rank"],
            "rank_local_cells": record["rank_local_cells"],
            "size_eligible": record["size_eligible"],
            "expected_timesteps": record["expected_timesteps"],
            "monitor_count": record["monitor_count"],
            "monitor_frequency_count": record["monitor_frequency_count"],
            "source_count": record["source_count"],
            "device_available": record["device"]["available"],
            "device_global_memory_bytes": record["device"][
                "global_memory_bytes"
            ],
            "estimated_required_memory_bytes": record["device"].get(
                "estimated_required_memory_bytes", 0
            ),
        }
        for record in rank_records
    ]
    return _sha256_json(
        {
            "schema": "gpmeep-v110-backend-ack-scope-v3",
            "selected_backend": selected,
            "recommended_backend": recommended,
            "reason_class": reason_class,
            "recommendation_confidence": confidence,
            "recommendation_policy_version": POLICY_VERSION,
            "mpi_topology_digest": topology_digest,
            "policy_facts": _policy_facts(rank_records),
            "policy_inputs": policy_inputs,
        }
    )


def _acknowledgement_scope_cache_enabled() -> bool:
    """Return whether an accepted policy class may be reused in this process."""
    return os.environ.get("MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE") is None


def _validated_acknowledged_policy_scopes() -> list[tuple[Any, dict[str, Any]]]:
    """Validate and snapshot the bounded, Simulation-owned reuse cache."""
    if type(_ACKNOWLEDGED_POLICY_SCOPES) is not weakref.WeakKeyDictionary:
        raise BackendRecommendationError(
            "backend acknowledgement scope cache is malformed"
        )
    if (
        type(_ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE) is not int
        or _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE < 0
    ):
        raise BackendRecommendationError(
            "backend acknowledgement scope cache sequence is malformed"
        )
    entries: list[tuple[Any, dict[str, Any]]] = []
    for owner, record in list(_ACKNOWLEDGED_POLICY_SCOPES.items()):
        if (
            type(record) is not dict
            or set(record)
            != {
                "owner_pid",
                "reason",
                "scope_digest",
                "cohort_digest",
                "sequence",
            }
            or type(record["owner_pid"]) is not int
            or record["owner_pid"] != os.getpid()
            or type(record["reason"]) is not str
            or not _is_lower_sha256(record["scope_digest"])
            or not _is_lower_sha256(record["cohort_digest"])
            or type(record["sequence"]) is not int
            or record["sequence"] < 0
        ):
            raise BackendRecommendationError(
                "backend acknowledgement scope cache entry is malformed"
            )
        if record["reason"]:
            _validate_acknowledgement_reason(record["reason"])
        entries.append((owner, record))
    if len(entries) > MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES:
        raise BackendRecommendationError(
            "backend acknowledgement scope cache exceeds its bound"
        )
    sequences = [record["sequence"] for _owner, record in entries]
    if (
        len(sequences) != len(set(sequences))
        or any(
            sequence <= 0
            or sequence > _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE
            for sequence in sequences
        )
    ):
        raise BackendRecommendationError(
            "backend acknowledgement scope cache sequence is inconsistent"
        )
    return entries


def _validated_acknowledged_process_policy_scopes() -> list[dict[str, Any]]:
    """Validate explicitly accepted process-wide policy classes."""
    if type(_ACKNOWLEDGED_PROCESS_POLICY_SCOPES) is not dict:
        raise BackendRecommendationError(
            "backend process-scope acknowledgement cache is malformed"
        )
    if (
        type(_ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE) is not int
        or _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE < 0
    ):
        raise BackendRecommendationError(
            "backend process-scope acknowledgement sequence is malformed"
        )
    entries: list[dict[str, Any]] = []
    for scope_digest, record in _ACKNOWLEDGED_PROCESS_POLICY_SCOPES.items():
        if (
            not _is_lower_sha256(scope_digest)
            or type(record) is not dict
            or set(record)
            != {
                "owner_pid",
                "reason",
                "scope_digest",
                "cohort_digest",
                "sequence",
            }
            or record["scope_digest"] != scope_digest
            or type(record["owner_pid"]) is not int
            or record["owner_pid"] != os.getpid()
            or type(record["reason"]) is not str
            or not _is_lower_sha256(record["cohort_digest"])
            or type(record["sequence"]) is not int
            or record["sequence"] < 0
        ):
            raise BackendRecommendationError(
                "backend process-scope acknowledgement entry is malformed"
            )
        _validate_acknowledgement_reason(record["reason"])
        entries.append(record)
    if len(entries) > MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES:
        raise BackendRecommendationError(
            "backend process-scope acknowledgement cache exceeds its bound"
        )
    sequences = [record["sequence"] for record in entries]
    if (
        len(sequences) != len(set(sequences))
        or any(
            sequence <= 0
            or sequence > _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE
            for sequence in sequences
        )
    ):
        raise BackendRecommendationError(
            "backend process-scope acknowledgement sequence is inconsistent"
        )
    return entries


def _acknowledged_policy_scope(
    simulation: Any, scope_digest: str
) -> dict[str, Any] | None:
    """Return only a scope explicitly accepted for this Simulation object."""
    entries = _validated_acknowledged_policy_scopes()
    for owner, record in entries:
        if owner is simulation and record["scope_digest"] == scope_digest:
            return record
    return None


def _remember_acknowledged_policy_scope(
    simulation: Any, scope_digest: str, reason: str, cohort_digest: str
) -> None:
    """Retain a bounded, Simulation-local policy decision and audit reason."""
    global _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE
    entries = _validated_acknowledged_policy_scopes()
    existing = _ACKNOWLEDGED_POLICY_SCOPES.get(simulation)
    if existing is None:
        while len(entries) >= MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES:
            oldest_owner, _oldest_record = min(
                entries, key=lambda entry: entry[1]["sequence"]
            )
            if _ACKNOWLEDGED_POLICY_SCOPES.pop(oldest_owner, None) is None:
                raise BackendRecommendationError(
                    "backend acknowledgement scope cache changed unexpectedly"
                )
            entries = _validated_acknowledged_policy_scopes()
    _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE += 1
    _ACKNOWLEDGED_POLICY_SCOPES[simulation] = {
        "owner_pid": os.getpid(),
        "reason": reason,
        "scope_digest": scope_digest,
        "cohort_digest": cohort_digest,
        "sequence": _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE,
    }


def _remember_acknowledged_process_policy_scope(
    scope_digest: str, reason: str, cohort_digest: str
) -> None:
    """Retain only a separately and explicitly accepted process-wide class."""
    global _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE
    entries = _validated_acknowledged_process_policy_scopes()
    if scope_digest not in _ACKNOWLEDGED_PROCESS_POLICY_SCOPES:
        while len(entries) >= MAXIMUM_ACKNOWLEDGED_POLICY_SCOPES:
            oldest = min(entries, key=lambda entry: entry["sequence"])
            removed = _ACKNOWLEDGED_PROCESS_POLICY_SCOPES.pop(
                oldest["scope_digest"], None
            )
            if removed is None:
                raise BackendRecommendationError(
                    "backend process-scope acknowledgement cache changed unexpectedly"
                )
            entries = _validated_acknowledged_process_policy_scopes()
    _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE += 1
    _ACKNOWLEDGED_PROCESS_POLICY_SCOPES[scope_digest] = {
        "owner_pid": os.getpid(),
        "reason": reason,
        "scope_digest": scope_digest,
        "cohort_digest": cohort_digest,
        "sequence": _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE,
    }


def _challenge(bindings: dict[str, Any]) -> str:
    return _sha256_json({"schema": "gpmeep-v110-backend-ack-v1", **bindings})


def _is_lower_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _remove_pending(simulation: Any, pending: dict[str, Any]) -> None:
    _PENDING_BY_SIMULATION.pop(simulation, None)
    _PENDING_BY_CHALLENGE.pop(pending["challenge"], None)


def _discard_dead_challenge(
    challenge: str, reference: weakref.ReferenceType[Any]
) -> None:
    with _LOCK:
        if _PENDING_BY_CHALLENGE.get(challenge) is reference:
            _PENDING_BY_CHALLENGE.pop(challenge, None)


def _validate_acknowledgement_reason(reason: Any) -> str:
    if not isinstance(reason, str):
        raise BackendRecommendationError(
            "backend acknowledgement reason must be text"
        )
    try:
        reason_bytes = reason.encode("utf-8")
    except UnicodeEncodeError as error:
        raise BackendRecommendationError(
            "backend acknowledgement reason must be valid UTF-8 text"
        ) from error
    if not MINIMUM_REASON_BYTES <= len(reason_bytes) <= MAXIMUM_REASON_BYTES:
        raise BackendRecommendationError(
            "backend acknowledgement reason must occupy 8 to 512 UTF-8 bytes"
        )
    return reason


def _validate_pending_record(pending: Any) -> dict[str, Any]:
    """Validate local mutable guard state before any rank can branch on it."""
    expected_keys = {
        "owner_pid",
        "stable_key",
        "native_fields_owner",
        "native_structure_owner",
        "run_nonce",
        "challenge",
        "reason",
        "reuse_scope",
        "telemetry",
    }
    if type(pending) is not dict or set(pending) != expected_keys:
        raise BackendRecommendationError(
            "backend acknowledgement pending record is malformed"
        )
    if type(pending["owner_pid"]) is not int:
        raise BackendRecommendationError(
            "backend acknowledgement pending owner is malformed"
        )
    stable_key = pending["stable_key"]
    expected_stable_keys = set(_ACKNOWLEDGEMENT_BINDING_KEYS) - {"run_nonce"}
    if type(stable_key) is not dict or set(stable_key) != expected_stable_keys:
        raise BackendRecommendationError(
            "backend acknowledgement pending binding is malformed"
        )
    if (
        stable_key["selected_backend"] not in {"cpu", "cuda"}
        or stable_key["recommended_backend"] not in {"cpu", "cuda"}
        or stable_key["recommendation_policy_version"] != POLICY_VERSION
        or any(
            not _is_lower_sha256(stable_key[key])
            for key in (
                "workload_fingerprint",
                "mpi_topology_digest",
                "native_state_digest",
                "process_nonce",
            )
        )
        or not _is_lower_sha256(pending["run_nonce"])
        or not _is_lower_sha256(pending["challenge"])
        or type(pending["reason"]) is not str
        or pending["reuse_scope"] not in {"simulation", "process"}
        or type(pending["telemetry"]) is not dict
    ):
        raise BackendRecommendationError(
            "backend acknowledgement pending values are malformed"
        )
    if pending["reason"]:
        _validate_acknowledgement_reason(pending["reason"])
    return pending


def guard_run(
    mp: Any,
    simulation: Any,
    until: Any,
    until_after_sources: Any,
    step_functions: Any = (),
    operation: Any = None,
) -> dict[str, Any]:
    """Guard an explicit backend before initialization mutates simulation state.

    The first mismatch is bound to the exact workload and must be acknowledged
    on an unchanged retry. The default acknowledgement is reusable only by that
    Simulation object. A caller may separately choose process scope when one
    deliberate decision should cover a batch of Simulation objects with the
    same policy class. Neither form ever changes the selected backend.
    """
    global _LAST_TELEMETRY
    _ensure_process_local_state()
    with _LOCK:
        try:
            if (
                type(_PENDING_BY_SIMULATION) is not weakref.WeakKeyDictionary
                or type(_PENDING_BY_CHALLENGE) is not dict
                or type(_ACKNOWLEDGED_POLICY_SCOPES)
                is not weakref.WeakKeyDictionary
                or type(_ACKNOWLEDGED_PROCESS_POLICY_SCOPES) is not dict
            ):
                raise BackendRecommendationError(
                    "backend acknowledgement process state is malformed"
                )
            # Validate every process-local capability before any rank can
            # choose the pending, reuse, matching-selection, or exact-retry
            # branch. A rank-local malformed cache must become one collective
            # preflight failure, never a late asymmetric return or exception.
            _validated_acknowledged_policy_scopes()
            _validated_acknowledged_process_policy_scopes()
            existing = _PENDING_BY_SIMULATION.get(simulation)
            if existing is not None:
                existing = _validate_pending_record(existing)
                if existing["owner_pid"] != os.getpid():
                    _remove_pending(simulation, existing)
                    existing = None
            selected = str(mp.gpu.requested_backend)
            if selected not in ("cpu", "cuda"):
                raise BackendRecommendationError(
                    "gpmeep v1.1.0 accepts only explicit 'cpu' or 'cuda'; "
                    "automatic backend selection is disabled"
                )
            if _is_exact_trusted_meep_type(
                simulation, "meep.simulation", "Simulation"
            ):
                file_input_values = [
                    getattr(simulation, field, None)
                    for field in _SIMULATION_FILE_INPUT_FIELDS
                ]
                if any(
                    value is not None
                    and (type(value) is not str or bool(value))
                    for value in file_input_values
                ):
                    # Configured native inputs retain byte-bound collective
                    # validation. The common unconfigured case does not pay
                    # an extra collective on every short run.
                    seal_simulation_file_inputs(mp, simulation)
            global_cells = _global_cells(simulation)
            ranks = int(mp.count_processors())
            if ranks <= 0:
                raise BackendRecommendationError("MPI communicator has no ranks")
            local_cells = int(math.ceil(global_cells / ranks))
            run_length = _run_length(simulation, until, until_after_sources)
            monitor_count = len(getattr(simulation, "dft_objects", ()))
            frequency_count = _monitor_frequency_count(simulation)
            source_count = len(getattr(simulation, "sources", ()))
            size_eligible = local_cells >= MINIMUM_RANK_LOCAL_CELLS
            device = _selected_device_record(
                mp, simulation, local_cells, size_eligible, selected
            )
            local_rank = int(mp.my_rank())
            topology_local = {
                "rank": local_rank,
                "global_rank": int(mp.my_global_rank()),
                "active_size": ranks,
                "node_rank": int(mp.my_node_rank()),
                "node_size": int(mp.count_node_processors()),
                "device_identifier": device["identifier"],
                "device_ordinal": device["ordinal"],
            }
            preflight_local = {
                "ok": True,
                "active_size": ranks,
                "rank": local_rank,
                "selected_backend": selected,
                "rank_local_cells": local_cells,
                "size_eligible": size_eligible,
                "expected_timesteps": run_length["expected_timesteps"],
                "monitor_count": monitor_count,
                "monitor_frequency_count": frequency_count,
                "source_count": source_count,
                "topology": topology_local,
                "device": device,
            }
        except Exception as error:
            preflight_local = _preflight_failure_record(error)
        preflight_records = _collective_records(
            mp, preflight_local, verify_count=False
        )
        failed_preflights = [
            (rank, record)
            for rank, record in enumerate(preflight_records)
            if record.get("ok") is not True
        ]
        if failed_preflights:
            detail = "; ".join(
                f"rank {rank}: {record.get('error_type', 'unknown')}: "
                f"{record.get('error', 'unspecified preflight failure')}"
                for rank, record in failed_preflights
            )
            raise BackendRecommendationError(
                f"distributed gpmeep workload preflight failed: {detail}"
            )
        declared_sizes = {record.get("active_size") for record in preflight_records}
        if (
            len(declared_sizes) != 1
            or next(iter(declared_sizes)) != len(preflight_records)
            or not 0 <= local_rank < len(preflight_records)
        ):
            raise BackendRecommendationError(
                "distributed gpmeep preflight rank inventory is inconsistent"
            )
        rank_records = preflight_records
        selected_backends = {record["selected_backend"] for record in rank_records}
        if len(selected_backends) != 1:
            raise BackendRecommendationError(
                "distributed gpmeep backend selections differ across MPI ranks"
            )
        selected = str(rank_records[0]["selected_backend"])
        topology_digest = _sha256_json(
            [record["topology"] for record in rank_records]
        )
        recommended, reason_class, confidence = _policy_decision(rank_records)
        scope_digest = _acknowledgement_scope_digest(
            rank_records,
            selected,
            recommended,
            reason_class,
            confidence,
            topology_digest,
        )
        scope_cache_enabled = _acknowledgement_scope_cache_enabled()
        expected_steps = run_length["expected_timesteps"]
        estimated_launches = (
            None
            if expected_steps is None
            else expected_steps * (4 + source_count + monitor_count)
        )
        telemetry: dict[str, Any] = {
            "schema": "gpmeep-v110-backend-recommendation-v3",
            "selected_backend": selected,
            "recommended_backend": recommended,
            "reason_class": reason_class,
            "recommendation_confidence": confidence,
            "rank_local_cells": [
                record["rank_local_cells"] for record in rank_records
            ],
            "run_length": run_length,
            "kernel_launch_count": estimated_launches,
            "transfer_bytes": sum(
                record["device"].get("estimated_required_memory_bytes", 0)
                for record in rank_records
            ),
            "monitor_count": monitor_count,
            "monitor_frequency_count": frequency_count,
            "source_count": source_count,
            "mpi_gpu_topology": [record["topology"] for record in rank_records],
            "profiler_or_ablation_id": PROFILER_OR_ABLATION_ID,
            "acknowledgement": False,
            "acknowledgement_allowed": reason_class != "environment",
            "acknowledgement_reason": "",
            "acknowledgement_reused": False,
            "acknowledgement_scope": "",
            "acknowledgement_origin_cohort_digest": "",
            "acknowledgement_scope_cache_enabled": scope_cache_enabled,
            "acknowledgement_scope_digest": scope_digest,
            "workload_fingerprint_evaluated": False,
            "workload_fingerprint": "",
            "recommendation_policy_version": POLICY_VERSION,
            "mpi_topology_digest": topology_digest,
            "native_state_digest": "",
            "process_nonce": "",
            "run_nonce": "",
            "acknowledgement_digest": "",
        }
        exact_retry_pending = existing is not None
        if ranks > 1:
            control_records = _collective_records(
                mp,
                {
                    "selected_backend": selected,
                    "recommended_backend": recommended,
                    "reason_class": reason_class,
                    "recommendation_confidence": confidence,
                    "scope_digest": scope_digest,
                    "scope_cache_enabled": scope_cache_enabled,
                    "pending_present": exact_retry_pending,
                },
            )
            control_keys = {
                "selected_backend",
                "recommended_backend",
                "reason_class",
                "recommendation_confidence",
                "scope_digest",
                "scope_cache_enabled",
                "pending_present",
            }
            if any(
                set(record) != control_keys
                or record["selected_backend"] not in {"cpu", "cuda"}
                or record["recommended_backend"] not in {"cpu", "cuda"}
                or record["reason_class"]
                not in {"hardware_crossover", "environment"}
                or record["recommendation_confidence"] not in {"medium", "high"}
                or not _is_lower_sha256(record["scope_digest"])
                or type(record["scope_cache_enabled"]) is not bool
                or type(record["pending_present"]) is not bool
                for record in control_records
            ):
                raise BackendRecommendationError(
                    "backend acknowledgement control record is malformed"
                )
            if any(record != control_records[0] for record in control_records):
                if len(
                    {
                        record["scope_cache_enabled"]
                        for record in control_records
                    }
                ) != 1:
                    raise BackendRecommendationError(
                        "backend acknowledgement scope cache configuration differs "
                        "across MPI ranks"
                    )
                if len(
                    {record["pending_present"] for record in control_records}
                ) != 1:
                    raise BackendRecommendationError(
                        "backend acknowledgement pending state differs across MPI ranks"
                    )
                raise BackendRecommendationError(
                    "backend recommendation decision differs across MPI ranks"
                )
            scope_cache_enabled = control_records[0]["scope_cache_enabled"]
            exact_retry_pending = control_records[0]["pending_present"]
            telemetry["acknowledgement_scope_cache_enabled"] = scope_cache_enabled
        if selected == "cuda" and reason_class == "environment":
            if existing:
                _remove_pending(simulation, existing)
            simulation._gpmeep_last_backend_recommendation = dict(telemetry)
            _LAST_TELEMETRY = dict(telemetry)
            raise BackendEnvironmentError(
                "gpmeep CUDA environment/device mapping is invalid or cannot "
                "satisfy the estimated memory requirement; acknowledgement "
                "is not permitted. Correct the environment before running.",
                telemetry,
            )
        # A singleton matching selection has no peer whose Python workload can
        # diverge, so it can return after the cheap collective preflight. MPI
        # matching selections retain the exact semantic fingerprint below;
        # equal cell and source counts do not prove equal rank-local programs.
        if selected == recommended and ranks == 1:
            if existing:
                _remove_pending(simulation, existing)
            simulation._gpmeep_last_backend_recommendation = dict(telemetry)
            _LAST_TELEMETRY = dict(telemetry)
            return telemetry

        deferred_mpi_reuse: dict[str, str] | None = None

        # An explicitly acknowledged exact challenge takes precedence over a
        # scope already owned by this Simulation. It must still prove that its
        # own immediate retry is unchanged before the scope cache is consulted.
        if selected != recommended:
            if scope_cache_enabled and not exact_retry_pending:
                try:
                    cached = _acknowledged_policy_scope(simulation, scope_digest)
                    reuse_scope = "simulation"
                    if cached is None:
                        cached = _ACKNOWLEDGED_PROCESS_POLICY_SCOPES.get(
                            scope_digest
                        )
                        reuse_scope = "process"
                    if cached is not None and (
                        type(cached) is not dict
                        or set(cached)
                        != {
                            "owner_pid",
                            "reason",
                            "scope_digest",
                            "cohort_digest",
                            "sequence",
                        }
                        or type(cached["owner_pid"]) is not int
                        or type(cached["reason"]) is not str
                        or not _is_lower_sha256(cached["scope_digest"])
                        or not _is_lower_sha256(cached["cohort_digest"])
                        or type(cached["sequence"]) is not int
                    ):
                        raise BackendRecommendationError(
                            "backend acknowledgement scope cache entry is malformed"
                        )
                    locally_reusable = bool(
                        cached
                        and cached["owner_pid"] == os.getpid()
                        and cached["reason"]
                    )
                    if locally_reusable:
                        _validate_acknowledgement_reason(cached["reason"])
                    reuse_local = {
                        "ok": True,
                        "reusable": locally_reusable,
                        "scope_digest": scope_digest,
                        "cohort_digest": (
                            cached["cohort_digest"] if cached else ""
                        ),
                        "reason": cached["reason"] if cached else "",
                        "reuse_scope": reuse_scope if cached else "",
                    }
                except Exception as error:
                    reuse_local = _preflight_failure_record(error)
                reuse_records = _collective_records(
                    mp,
                    reuse_local,
                    verify_count=False,
                )
                failed_reuse = [
                    (rank, record)
                    for rank, record in enumerate(reuse_records)
                    if record.get("ok") is not True
                ]
                if failed_reuse:
                    detail = "; ".join(
                        f"rank {rank}: {record.get('error_type', 'unknown')}: "
                        f"{record.get('error', 'unspecified scope cache failure')}"
                        for rank, record in failed_reuse
                    )
                    raise BackendRecommendationError(
                        "distributed backend acknowledgement scope cache "
                        f"preflight failed: {detail}"
                    )
                if any(
                    set(record)
                    != {
                        "ok",
                        "reusable",
                        "scope_digest",
                        "cohort_digest",
                        "reason",
                        "reuse_scope",
                    }
                    or record["ok"] is not True
                    or type(record["reusable"]) is not bool
                    or type(record["scope_digest"]) is not str
                    or type(record["cohort_digest"]) is not str
                    or type(record["reason"]) is not str
                    or record["reuse_scope"] not in {"", "simulation", "process"}
                    for record in reuse_records
                ):
                    raise BackendRecommendationError(
                        "backend acknowledgement scope cache record is malformed"
                    )
                any_reusable = any(
                    record["reusable"] for record in reuse_records
                )
                all_reusable = all(
                    record["reusable"] for record in reuse_records
                )
                if any_reusable and not all_reusable:
                    raise BackendRecommendationError(
                        "backend acknowledgement scope cache differs across MPI ranks"
                    )
                if all_reusable:
                    scope_digests = {
                        record["scope_digest"] for record in reuse_records
                    }
                    reasons = {record["reason"] for record in reuse_records}
                    cohort_digests = {
                        record["cohort_digest"] for record in reuse_records
                    }
                    reuse_scopes = {
                        record["reuse_scope"] for record in reuse_records
                    }
                    if (
                        scope_digests != {scope_digest}
                        or len(reasons) != 1
                        or len(cohort_digests) != 1
                        or len(reuse_scopes) != 1
                        or next(iter(reuse_scopes))
                        not in {"simulation", "process"}
                        or not _is_lower_sha256(next(iter(cohort_digests)))
                    ):
                        raise BackendRecommendationError(
                            "backend acknowledgement scope differs across MPI ranks"
                        )
                    acknowledgement_reason = _validate_acknowledgement_reason(
                        next(iter(reasons))
                    )
                    reuse_scope = next(iter(reuse_scopes))
                    origin_cohort_digest = next(iter(cohort_digests))
                    if ranks > 1:
                        # Every MPI reuse path still proves semantic equality
                        # across ranks. Simulation ownership is not sufficient:
                        # rank-local callbacks or geometry can mutate after the
                        # first accepted run. Continue through the exact
                        # collective fingerprint, then reuse the audited policy
                        # decision without another warning/retry.
                        deferred_mpi_reuse = {
                            "reason": acknowledgement_reason,
                            "origin_cohort_digest": origin_cohort_digest,
                            "scope": reuse_scope,
                        }
                    else:
                        if existing:
                            _remove_pending(simulation, existing)
                        telemetry["acknowledgement"] = True
                        telemetry["acknowledgement_reused"] = True
                        telemetry["acknowledgement_reason"] = (
                            acknowledgement_reason
                        )
                        telemetry["acknowledgement_scope"] = reuse_scope
                        telemetry["acknowledgement_origin_cohort_digest"] = (
                            origin_cohort_digest
                        )
                        simulation._gpmeep_last_backend_recommendation = dict(
                            telemetry
                        )
                        _LAST_TELEMETRY = dict(telemetry)
                        return telemetry

        # A new mismatch retains the exact, fail-before-run handshake.  The
        # expensive graph traversal is intentionally below both fast paths.
        # The validated MPI control records have already proved that every
        # rank has a matching selection and no pending capability here.  Send
        # nonce proposals with the exact fingerprint, without caching or
        # skipping any semantic/native validation. All other paths retain the
        # original pending/retry/acknowledgement exchange.
        fuse_fresh_matching = (
            ranks > 1 and selected == recommended and not exact_retry_pending
        )
        try:
            workload_fingerprint = _sha256_json(
                _workload_record(
                    simulation,
                    until,
                    until_after_sources,
                    step_functions,
                    operation,
                )
            )
            exact_meep_simulation = _is_exact_trusted_meep_type(
                simulation, "meep.simulation", "Simulation"
            )
            native_state_fingerprint = _sha256_json(
                _native_simulation_state(simulation)
                if exact_meep_simulation
                else {
                    "schema": "gpmeep-native-simulation-state-v2",
                    "not_meep": True,
                }
            )
            exact_local = {
                "ok": True,
                "active_size": ranks,
                "rank": local_rank,
                "workload_fingerprint": workload_fingerprint,
                "native_state_fingerprint": native_state_fingerprint,
            }
            if fuse_fresh_matching:
                run_nonce_proposal = secrets.token_hex(32)
                if not (
                    _is_lower_sha256(_PROCESS_NONCE)
                    and _is_lower_sha256(run_nonce_proposal)
                ):
                    raise BackendRecommendationError(
                        "backend acknowledgement nonce proposal is malformed"
                    )
                exact_local.update(
                    process_nonce=_PROCESS_NONCE,
                    run_nonce_proposal=run_nonce_proposal,
                )
        except Exception as error:
            exact_local = _preflight_failure_record(error)
        exact_records = _collective_records(mp, exact_local, verify_count=False)
        failed_exact = [
            (rank, record)
            for rank, record in enumerate(exact_records)
            if record.get("ok") is not True
        ]
        if failed_exact:
            detail = "; ".join(
                f"rank {rank}: {record.get('error_type', 'unknown')}: "
                f"{record.get('error', 'unspecified exact preflight failure')}"
                for rank, record in failed_exact
            )
            raise BackendRecommendationError(
                f"distributed gpmeep exact workload preflight failed: {detail}"
            )
        exact_sizes = {record.get("active_size") for record in exact_records}
        if (
            len(exact_sizes) != 1
            or next(iter(exact_sizes)) != len(exact_records)
            or not 0 <= local_rank < len(exact_records)
        ):
            raise BackendRecommendationError(
                "distributed gpmeep exact preflight rank inventory is inconsistent"
            )
        fingerprints = {
            record["workload_fingerprint"] for record in exact_records
        }
        if len(fingerprints) != 1:
            raise BackendRecommendationError(
                "distributed gpmeep workload fingerprints differ across MPI ranks"
            )
        workload_fingerprint = str(
            exact_records[local_rank]["workload_fingerprint"]
        )
        native_state_digest = _sha256_json(
            [record["native_state_fingerprint"] for record in exact_records]
        )
        if fuse_fresh_matching:
            if any(
                set(record) != {
                    "ok", "active_size", "rank", "workload_fingerprint",
                    "native_state_fingerprint", "process_nonce", "run_nonce_proposal",
                }
                or not _is_lower_sha256(record["process_nonce"])
                or not _is_lower_sha256(record["run_nonce_proposal"])
                for record in exact_records
            ):
                raise BackendRecommendationError(
                    "backend acknowledgement exact nonce record is malformed"
                )
            process_nonce = _sha256_json(
                [record["process_nonce"] for record in exact_records]
            )
        else:
            process_records = _collective_records(mp, {"nonce": _PROCESS_NONCE})
            process_nonce = _sha256_json(
                [record["nonce"] for record in process_records]
            )
        native_fields_owner = getattr(simulation, "fields", None)
        native_structure_owner = getattr(simulation, "structure", None)
        stable_key = {
            "selected_backend": selected,
            "recommended_backend": recommended,
            "workload_fingerprint": workload_fingerprint,
            "recommendation_policy_version": POLICY_VERSION,
            "mpi_topology_digest": topology_digest,
            "native_state_digest": native_state_digest,
            "process_nonce": process_nonce,
        }
        if fuse_fresh_matching:
            run_nonce = exact_records[0]["run_nonce_proposal"]
        else:
            pending_records = _collective_records(
                mp,
                {
                    "stable": bool(
                        existing
                        and existing.get("owner_pid") == os.getpid()
                        and existing["stable_key"] == stable_key
                        and existing.get("native_fields_owner") is native_fields_owner
                        and existing.get("native_structure_owner")
                        is native_structure_owner
                    ),
                    "run_nonce": (
                        existing["run_nonce"]
                        if existing and existing["stable_key"] == stable_key
                        else ""
                    ),
                },
            )
            stable_pending = all(record["stable"] for record in pending_records)
            if any(record["stable"] for record in pending_records) and not stable_pending:
                raise BackendRecommendationError(
                    "backend acknowledgement pending state differs across MPI ranks"
                )
            if stable_pending:
                run_nonces = {record["run_nonce"] for record in pending_records}
                if len(run_nonces) != 1 or not next(iter(run_nonces)):
                    raise BackendRecommendationError(
                        "backend acknowledgement run nonce differs across MPI ranks"
                    )
                run_nonce = next(iter(run_nonces))
            else:
                if existing:
                    _remove_pending(simulation, existing)
                    existing = None
                nonce_records = _collective_records(
                    mp, {"nonce": secrets.token_hex(32)}
                )
                run_nonce = str(nonce_records[0]["nonce"])
        binding_values = {**stable_key, "run_nonce": run_nonce}
        if set(binding_values) != set(_ACKNOWLEDGEMENT_BINDING_KEYS):
            raise BackendRecommendationError(
                "internal backend acknowledgement binding contract drifted"
            )
        bindings = {
            key: binding_values[key] for key in _ACKNOWLEDGEMENT_BINDING_KEYS
        }
        challenge = _challenge(bindings)
        telemetry.update(
            {
                "workload_fingerprint_evaluated": True,
                "workload_fingerprint": workload_fingerprint,
                "native_state_digest": native_state_digest,
                "process_nonce": process_nonce,
                "run_nonce": run_nonce,
                "acknowledgement_digest": challenge,
            }
        )

        if selected == recommended:
            if existing:
                _remove_pending(simulation, existing)
            simulation._gpmeep_last_backend_recommendation = dict(telemetry)
            _LAST_TELEMETRY = dict(telemetry)
            return telemetry

        if deferred_mpi_reuse is not None:
            if existing:
                _remove_pending(simulation, existing)
            telemetry["acknowledgement"] = True
            telemetry["acknowledgement_reused"] = True
            telemetry["acknowledgement_reason"] = deferred_mpi_reuse[
                "reason"
            ]
            telemetry["acknowledgement_scope"] = deferred_mpi_reuse["scope"]
            telemetry["acknowledgement_origin_cohort_digest"] = (
                deferred_mpi_reuse["origin_cohort_digest"]
            )
            simulation._gpmeep_last_backend_recommendation = dict(telemetry)
            _LAST_TELEMETRY = dict(telemetry)
            return telemetry

        acknowledgement_records = _collective_records(
            mp,
            {
                "challenge": existing["challenge"] if existing else "",
                "reason": existing.get("reason", "") if existing else "",
                "reuse_scope": (
                    existing.get("reuse_scope", "") if existing else ""
                ),
            },
        )
        if any(record != acknowledgement_records[0] for record in acknowledgement_records):
            raise BackendRecommendationError(
                "backend acknowledgement differs across MPI ranks"
            )
        collective_acknowledgement = acknowledgement_records[0]
        if (
            collective_acknowledgement["challenge"] == challenge
            and collective_acknowledgement["reason"]
        ):
            acknowledgement_reason = _validate_acknowledgement_reason(
                collective_acknowledgement["reason"]
            )
            reuse_scope = collective_acknowledgement["reuse_scope"]
            if reuse_scope not in {"simulation", "process"}:
                raise BackendRecommendationError(
                    "backend acknowledgement reuse scope is malformed"
                )
            consumed_records = _collective_records(
                mp, {"consumed": run_nonce in _CONSUMED_RUN_NONCES}
            )
            if any(record["consumed"] for record in consumed_records):
                raise BackendRecommendationError(
                    "backend acknowledgement run nonce was already consumed"
                )
            _CONSUMED_RUN_NONCES.add(run_nonce)
            telemetry["acknowledgement"] = True
            telemetry["acknowledgement_reason"] = acknowledgement_reason
            telemetry["acknowledgement_scope"] = reuse_scope
            telemetry["acknowledgement_origin_cohort_digest"] = challenge
            if scope_cache_enabled:
                if reuse_scope == "process":
                    _remember_acknowledged_process_policy_scope(
                        scope_digest, acknowledgement_reason, challenge
                    )
                else:
                    _remember_acknowledged_policy_scope(
                        simulation,
                        scope_digest,
                        acknowledgement_reason,
                        challenge,
                    )
            _remove_pending(simulation, existing)
            simulation._gpmeep_last_backend_recommendation = dict(telemetry)
            _LAST_TELEMETRY = dict(telemetry)
            return telemetry

        pending = {
            "owner_pid": os.getpid(),
            "stable_key": stable_key,
            # Keep the local SWIG owner alive and compare it by identity on the
            # retry.  Pointer values intentionally never enter the collective
            # challenge because they differ across MPI ranks.
            "native_fields_owner": native_fields_owner,
            "native_structure_owner": native_structure_owner,
            "run_nonce": run_nonce,
            "challenge": challenge,
            "reason": existing.get("reason", "") if existing else "",
            "reuse_scope": (
                existing.get("reuse_scope", "simulation")
                if existing
                else "simulation"
            ),
            "telemetry": dict(telemetry),
        }
        _PENDING_BY_SIMULATION[simulation] = pending
        reference = weakref.ref(
            simulation,
            lambda dead_reference, key=challenge: _discard_dead_challenge(
                key, dead_reference
            ),
        )
        _PENDING_BY_CHALLENGE[challenge] = reference
        _LAST_TELEMETRY = dict(telemetry)
        raise BackendRecommendationMismatch(
            "gpmeep backend recommendation mismatch: selected "
            f"{selected!r}, recommended {recommended!r} ({reason_class}, "
            f"confidence={confidence}). No backend was changed. To proceed with "
            "this exact retry and remember the same policy class for this "
            "Simulation object in this process, call "
            f"mp.gpu.acknowledge_backend_mismatch({challenge!r}, "
            "reason='at least 8 UTF-8 bytes'). For an explicitly accepted "
            "batch of equivalent policy decisions, pass scope='process'. "
            "Then repeat Simulation.run.",
            challenge,
            telemetry,
        )


def acknowledge_backend_mismatch(
    challenge: str, reason: str, scope: str = "simulation"
) -> None:
    if not isinstance(challenge, str) or len(challenge) != 64 or any(
        character not in "0123456789abcdef" for character in challenge
    ):
        raise BackendRecommendationError(
            "backend acknowledgement challenge must be 64 lowercase hex characters"
        )
    reason = _validate_acknowledgement_reason(reason)
    if scope not in {"simulation", "process"}:
        raise BackendRecommendationError(
            "backend acknowledgement scope must be 'simulation' or 'process'"
        )
    _ensure_process_local_state()
    with _LOCK:
        reference = _PENDING_BY_CHALLENGE.get(challenge)
        simulation = reference() if reference else None
        if simulation is None:
            raise BackendRecommendationError(
                "backend acknowledgement challenge is stale, unknown, or already consumed"
            )
        pending = _PENDING_BY_SIMULATION.get(simulation)
        if (
            not pending
            or pending.get("owner_pid") != os.getpid()
            or pending["challenge"] != challenge
        ):
            raise BackendRecommendationError(
                "backend acknowledgement challenge is stale or changed"
            )
        pending["reason"] = reason
        pending["reuse_scope"] = scope


def pending_recommendation() -> dict[str, Any] | None:
    _ensure_process_local_state()
    with _LOCK:
        pending = next(iter(_PENDING_BY_SIMULATION.values()), None)
        return dict(pending["telemetry"]) if pending else None


def last_recommendation() -> dict[str, Any] | None:
    _ensure_process_local_state()
    with _LOCK:
        return dict(_LAST_TELEMETRY) if _LAST_TELEMETRY is not None else None


def _reset_for_testing() -> None:
    global _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE
    global _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE, _LAST_TELEMETRY
    global _DEVICE_INVENTORY_CACHE
    with _LOCK:
        _PENDING_BY_SIMULATION.clear()
        _PENDING_BY_CHALLENGE.clear()
        _FILE_INPUT_SEALS_BY_SIMULATION.clear()
        _CONSUMED_RUN_NONCES.clear()
        _ACKNOWLEDGED_POLICY_SCOPES.clear()
        _ACKNOWLEDGED_PROCESS_POLICY_SCOPES.clear()
        _ACKNOWLEDGED_POLICY_SCOPE_SEQUENCE = 0
        _ACKNOWLEDGED_PROCESS_POLICY_SCOPE_SEQUENCE = 0
        _LAST_TELEMETRY = None
        _DEVICE_INVENTORY_CACHE = None
