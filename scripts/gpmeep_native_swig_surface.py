#!/usr/bin/env python3
"""Emit a complete, standalone JSON inventory of the installed Meep surface."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REQUIRED_NATIVE_CLASSES = {
    "continuous_src_time",
    "custom_py_src_time",
    "custom_src_time",
    "dft_chunk",
    "dft_energy",
    "dft_fields",
    "dft_flux",
    "dft_force",
    "dft_near2far",
    "fields",
    "flux_vol",
    "gaussian_src_time",
    "src_time",
    "structure",
    "structure_chunk",
}
REQUIRED_NATIVE_CALLABLES = {"dft_ldos"}


def _member_record(owner: type, name: str) -> dict[str, Any]:
    try:
        value = inspect.getattr_static(owner, name)
    except Exception as error:
        return {"kind": "unreadable", "error": type(error).__name__}
    if isinstance(value, property):
        return {"kind": "property", "writable": value.fset is not None}
    if isinstance(value, (staticmethod, classmethod)):
        return {"kind": type(value).__name__, "callable": True}
    if callable(value):
        return {"kind": "callable", "callable": True}
    return {"kind": type(value).__name__, "callable": False}


def _class_record(value: type) -> dict[str, Any]:
    names = sorted(name for name in dir(value) if not name.startswith("__"))
    return {
        "module": value.__module__,
        "qualname": value.__qualname__,
        "swig_proxy": (
            "thisown" in names or "__swig_destroy__" in value.__dict__
        ),
        "members": {name: _member_record(value, name) for name in names},
    }


def build_inventory(meep: Any) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    callables: dict[str, Any] = {}
    public_names = sorted(name for name in dir(meep) if not name.startswith("_"))
    for name in public_names:
        try:
            value = getattr(meep, name)
        except Exception as error:
            callables[name] = {
                "kind": "unreadable",
                "error": type(error).__name__,
            }
            continue
        if inspect.isclass(value):
            classes[name] = _class_record(value)
        elif callable(value):
            callables[name] = {
                "kind": type(value).__name__,
                "module": getattr(value, "__module__", None),
                "qualname": getattr(value, "__qualname__", None),
            }

    missing_classes = sorted(REQUIRED_NATIVE_CLASSES.difference(classes))
    missing_callables = sorted(REQUIRED_NATIVE_CALLABLES.difference(callables))
    if missing_classes or missing_callables:
        raise RuntimeError(
            "required native public surface is absent: "
            f"classes={missing_classes}, callables={missing_callables}"
        )

    extension_public_names = sorted(
        name for name in dir(meep._meep) if not name.startswith("_")
    )
    extension_exports: dict[str, Any] = {}
    for name in extension_public_names:
        try:
            value = getattr(meep._meep, name)
        except Exception as error:
            extension_exports[name] = {
                "kind": "unreadable",
                "error": type(error).__name__,
            }
            continue
        extension_exports[name] = {
            "kind": type(value).__name__,
            "callable": callable(value),
        }

    raw_owner_setters = {
        "fields": sorted(
            name
            for name in extension_public_names
            if name.startswith("fields_") and name.endswith("_set")
        ),
        "structure": sorted(
            name
            for name in extension_public_names
            if name.startswith("structure_")
            and not name.startswith("structure_chunk_")
            and name.endswith("_set")
        ),
        "dft": sorted(
            name
            for name in extension_public_names
            if name.startswith(
                (
                    "dft_chunk_",
                    "dft_energy_",
                    "dft_fields_",
                    "dft_flux_",
                    "dft_force_",
                    "dft_near2far_",
                )
            )
            and name.endswith("_set")
        ),
        "flux_vol": sorted(
            name
            for name in extension_public_names
            if name.startswith("flux_vol_") and name.endswith("_set")
        ),
        "src_time": sorted(
            name
            for name in extension_public_names
            if name.startswith("src_time_") and name.endswith("_set")
        ),
    }

    flux_init = inspect.getsource(meep.flux_vol.__init__)
    structure_effort = inspect.getattr_static(meep.structure, "effort")
    source_next = inspect.getattr_static(meep.src_time, "next")
    trusted_collector = getattr(
        meep._meep, "_gpmeep_fields_external_source_states", None
    )
    assertions = {
        "internal_halo_transport_exposed": hasattr(
            meep.fields, "process_incoming_chunk_data"
        ),
        "direct_flux_vol_constructor_exposed": (
            "No constructor defined" not in flux_init
        ),
        "opaque_external_source_accessor_exposed": hasattr(
            meep.src_time, "gpmeep_external_state"
        ),
        "trusted_external_source_collector_present": hasattr(
            meep._meep, "_gpmeep_fields_external_source_states"
        ),
        "trusted_external_source_collector_is_builtin": inspect.isbuiltin(
            trusted_collector
        ),
        "structure_effort_proxy_writable": (
            isinstance(structure_effort, property)
            and structure_effort.fset is not None
        ),
        "raw_structure_effort_set_exposed": hasattr(
            meep._meep, "structure_effort_set"
        ),
        "source_next_proxy_writable": (
            isinstance(source_next, property) and source_next.fset is not None
        ),
        "raw_source_next_set_exposed": hasattr(
            meep._meep, "src_time_next_set"
        ),
        "raw_direct_flux_vol_constructor_exposed": hasattr(
            meep._meep, "new_flux_vol"
        ),
        "raw_internal_halo_transport_exposed": hasattr(
            meep._meep, "fields_process_incoming_chunk_data"
        ),
        "raw_external_source_accessor_exposed": any(
            hasattr(meep._meep, name)
            for name in (
                "src_time_gpmeep_external_state",
                "custom_py_src_time_gpmeep_external_state",
            )
        ),
        "raw_owner_setters_are_exactly_manifest_safe": (
            raw_owner_setters
            == {
                "fields": ["fields_t_set"],
                "structure": [],
                "dft": [],
                "flux_vol": [],
                "src_time": [
                    "src_time_id_set",
                    "src_time_is_integrated_set",
                ],
            }
        ),
    }
    if assertions != {
        "internal_halo_transport_exposed": False,
        "direct_flux_vol_constructor_exposed": False,
        "opaque_external_source_accessor_exposed": False,
        "trusted_external_source_collector_present": True,
        "trusted_external_source_collector_is_builtin": True,
        "structure_effort_proxy_writable": False,
        "raw_structure_effort_set_exposed": False,
        "source_next_proxy_writable": False,
        "raw_source_next_set_exposed": False,
        "raw_direct_flux_vol_constructor_exposed": False,
        "raw_internal_halo_transport_exposed": False,
        "raw_external_source_accessor_exposed": False,
        "raw_owner_setters_are_exactly_manifest_safe": True,
    }:
        raise RuntimeError(f"native surface safety assertion failed: {assertions}")

    return {
        "schema": "gpmeep-v110-swig-native-surface-v4",
        "scope": (
            "every public class and callable in the generated meep namespace; "
            "each class includes every non-dunder member; every public raw "
            "meep._meep extension export is also classified"
        ),
        "public_name_count": len(public_names),
        "public_names_sha256": hashlib.sha256(
            "\0".join(public_names).encode("utf-8")
        ).hexdigest(),
        "required_native_classes": sorted(REQUIRED_NATIVE_CLASSES),
        "required_native_callables": sorted(REQUIRED_NATIVE_CALLABLES),
        "classes": classes,
        "callables": callables,
        "extension_public_name_count": len(extension_public_names),
        "extension_public_names_sha256": hashlib.sha256(
            "\0".join(extension_public_names).encode("utf-8")
        ).hexdigest(),
        "extension_exports": extension_exports,
        "raw_owner_setters": raw_owner_setters,
        "safety_assertions": assertions,
    }


def write_inventory(path: Path, inventory: dict[str, Any]) -> None:
    payload = json.dumps(
        inventory, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    # Prove the exact bytes are standalone JSON before publication.
    decoded = json.loads(payload)
    if decoded != inventory:
        raise RuntimeError("native surface JSON self-check changed the payload")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    import meep

    inventory = build_inventory(meep)
    write_inventory(arguments.output.resolve(), inventory)
    print(
        json.dumps(
            {
                "schema": inventory["schema"],
                "output": str(arguments.output.resolve()),
                "class_count": len(inventory["classes"]),
                "callable_count": len(inventory["callables"]),
                "extension_export_count": len(inventory["extension_exports"]),
            },
            sort_keys=True,
        ),
        file=os.sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
