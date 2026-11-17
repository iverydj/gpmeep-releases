from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import py_compile
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPERIMENT = load_script(
    "gpmeep_v110_inversion_experiment_test",
    ROOT / "scripts/v110-inversion-experiment-sample.py",
)
CHILD = load_script(
    "gpmeep_v110_inversion_qualified_child_test",
    ROOT / "scripts/v110-inversion-qualified-child.py",
)
REGISTRY = load_script(
    "gpmeep_v110_inversion_registry_experiment_test",
    ROOT / "scripts/validate-v110-inversion-registry.py",
)
IDENTITY = load_script(
    "gpmeep_v110_inversion_identity_test",
    ROOT / "scripts/python-validation/v110_runtime_identity.py",
)
ISOLATED_M4_REPLAY = EXPERIMENT._isolated_m4_replay


class V110InversionExperimentSampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name).resolve()
        self.repo = self.root / "source"
        self.repo.mkdir()
        (self.repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        self.workload = "python/examples/refl-quartz.py"
        self.gpu = "GPU-00000000-0000-0000-0000-000000000001"
        self.count = 0
        workload = self.repo / self.workload
        workload.parent.mkdir(parents=True)
        workload.write_text(
            "import meep as mp\n"
            "simulation = mp.Simulation(resolution=1.0)\n"
            "simulation.run(until=1.0)\n"
        )
        source_child = self.repo / "scripts/v110-inversion-qualified-child.py"
        source_child.parent.mkdir()
        shutil.copyfile(ROOT / "scripts/v110-inversion-qualified-child.py", source_child)
        shutil.copyfile(
            ROOT / "scripts/v110-final-release-receipt.py",
            self.repo / "scripts/v110-final-release-receipt.py",
        )
        (self.repo / "scripts/verify-gpmeep-package.py").write_bytes(
            b"# fixture installed verifier\n"
        )
        (self.repo / "scripts/gpmeep-verify-provenance-launcher.sh").write_bytes(
            b"#!/bin/sh\nexit 0\n"
        )
        source_validation = self.repo / "scripts/python-validation"
        source_validation.mkdir()
        for name in ("v110_runtime_identity.py", "run_m4_equal_work_matrix.py"):
            shutil.copyfile(
                ROOT / f"scripts/python-validation/{name}", source_validation / name
            )
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "gpmeep test"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True)
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        self._build_runtime("normal")
        self.experiment_loader = EXPERIMENT._load_module
        self.registry_loader = REGISTRY._load_module
        self.experiment_patch = mock.patch.object(
            EXPERIMENT, "_load_module", side_effect=self._authority_module
        )
        self.registry_patch = mock.patch.object(
            REGISTRY, "_load_module", side_effect=self._registry_module
        )
        self.m4_patch = mock.patch.object(
            EXPERIMENT,
            "_isolated_m4_replay",
            side_effect=lambda _controller, output, _python, _prefix: self._replay_m4(
                output
            ),
        )
        self.experiment_patch.start()
        self.registry_patch.start()
        self.m4_patch.start()

    @staticmethod
    def record(path: pathlib.Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @staticmethod
    def _tar(
        entries: dict[str, tuple[bytes, int]],
        softlinks: dict[str, tuple[str, int]] | None = None,
    ) -> bytes:
        output = io.BytesIO()
        softlinks = softlinks or {}
        with tarfile.open(fileobj=output, mode="w:") as archive:
            for name in sorted({*entries, *softlinks}):
                if name in softlinks:
                    target, mode = softlinks[name]
                    item = tarfile.TarInfo(name)
                    item.type = tarfile.SYMTYPE
                    item.linkname = target
                    item.mode = mode
                    item.mtime = 0
                    item.uid = 0
                    item.gid = 0
                    item.uname = ""
                    item.gname = ""
                    archive.addfile(item)
                    continue
                payload, mode = entries[name]
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                item.mode = mode
                item.mtime = 0
                item.uid = 0
                item.gid = 0
                item.uname = ""
                item.gname = ""
                archive.addfile(item, io.BytesIO(payload))
        return output.getvalue()

    @classmethod
    def _zstd(cls, payload: bytes) -> bytes:
        result = subprocess.run(
            ["/usr/bin/zstd", "-q", "-c", "-"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("fixture zstd compression failed")
        return result.stdout

    @classmethod
    def _write_conda(
        cls,
        package: pathlib.Path,
        payload: dict[str, tuple[bytes, int]],
        softlinks: dict[str, tuple[str, int]] | None = None,
    ) -> list[dict[str, object]]:
        softlinks = softlinks or {}
        archive_paths = [
            {
                "_path": name,
                "path_type": "hardlink",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_in_bytes": len(data),
            }
            for name, (data, _mode) in sorted(payload.items())
        ]
        installed_paths = [dict(item) for item in archive_paths]
        for name, (target, _mode) in sorted(softlinks.items()):
            target_name = pathlib.PurePosixPath(
                pathlib.PurePosixPath(name).parent, target
            ).as_posix()
            target_data = payload[target_name][0]
            digest = hashlib.sha256(target_data).hexdigest()
            archive_paths.append(
                {
                    "_path": name,
                    "path_type": "softlink",
                    "sha256": digest,
                    "size_in_bytes": len(target_data),
                }
            )
            installed_paths.append(
                {
                    "_path": name,
                    "path_type": "softlink",
                    "sha256_in_prefix": digest,
                    "size_in_bytes": len(target_data),
                }
            )
        archive_paths.sort(key=lambda item: str(item["_path"]))
        installed_paths.sort(key=lambda item: str(item["_path"]))
        info = {
            "info/index.json": (
                json.dumps(
                    {
                        "name": "gpmeep",
                        "version": "1.1.0",
                        "build": "cuda124_mpi_openmpi_py311_0_development",
                        "build_number": 0,
                        "subdir": "linux-64",
                    },
                    sort_keys=True,
                ).encode(),
                0o644,
            ),
            "info/paths.json": (
                json.dumps(
                    {"paths_version": 1, "paths": archive_paths}, sort_keys=True
                ).encode(),
                0o644,
            ),
            "info/files": (
                ("\n".join(sorted({*payload, *softlinks})) + "\n").encode(),
                0o644,
            ),
        }
        with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "metadata.json",
                json.dumps({"conda_pkg_format_version": 2}, sort_keys=True),
            )
            archive.writestr("info-fixture.tar.zst", cls._zstd(cls._tar(info)))
            archive.writestr(
                "pkg-fixture.tar.zst",
                cls._zstd(cls._tar(payload, softlinks)),
            )
        return installed_paths

    def _authority_module(self, name: str, path: pathlib.Path):
        if path.name == "v110_runtime_identity.py":
            return types.SimpleNamespace(
                validate_report=IDENTITY.validate_report,
                runtime_paths=IDENTITY.runtime_paths,
                live_verify=lambda report, _paths: self.live_verification,
            )
        return self.experiment_loader(name, path)

    def _registry_module(self, name: str, path: pathlib.Path):
        if pathlib.Path(path).name == "v110-inversion-experiment-sample.py":
            return EXPERIMENT
        return self.registry_loader(name, path)

    def _replay_m4(self, output: pathlib.Path) -> dict[str, object]:
        self.assertEqual(output, self.m4_output)
        return self.m4_report

    def _fabricate_placeholder_authority(self) -> None:
        self.package.write_bytes(b"not-a-conda-package\n")
        package_sha256 = hashlib.sha256(self.package.read_bytes()).hexdigest()
        self._write_nonpublishable_marker(package_sha256)
        conda_path = pathlib.Path(json.loads(self.provenance.read_text())["conda_record"]["path"])
        conda = json.loads(conda_path.read_text())
        conda["sha256"] = package_sha256
        conda_path.write_text(json.dumps(conda, sort_keys=True))
        report = json.loads(self.provenance.read_text())
        report["conda_record"] = self.record(conda_path)
        report["package_sha256"] = package_sha256
        report.pop("report_id", None)
        report["report_id"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.provenance.write_text(json.dumps(report, sort_keys=True))
        self.runtime_id = report["report_id"]
        self.live_verification = {
            **self.live_verification,
            "report_id": self.runtime_id,
            "package_sha256": package_sha256,
        }
        inputs = self.m4_report["contract"]["runtime_inputs"]
        inputs.update(
            {
                "v110_package": self.record(self.package),
                "v110_provenance_report": self.record(self.provenance),
                "v110_package_sha256": package_sha256,
                "v110_installed_report_id": self.runtime_id,
                "v110_live_verification": self.live_verification,
                "v110_archive_verification": {"fabricated": True},
            }
        )

    def _write_nonpublishable_marker(self, package_sha256: str) -> None:
        marker = pathlib.Path(
            f"{self.package}{IDENTITY.NONPUBLISHABLE_MARKER_SUFFIX}"
        )
        if marker.exists():
            marker.chmod(0o600)
        marker.write_text(
            json.dumps(
                {
                    "schema": IDENTITY.NONPUBLISHABLE_MARKER_SCHEMA,
                    "state": "NON_PUBLISHABLE",
                    "source_commit": self.source_commit,
                    "package": {
                        "file_name": self.package.name,
                        "size_bytes": self.package.stat().st_size,
                        "sha256": package_sha256,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        marker.chmod(0o400)
        self.nonpublishable_marker = marker

    def _build_runtime(self, behavior: str) -> None:
        self.prefix = self.root / f"prefix-{behavior}"
        module = self.prefix / "lib/python3.11/site-packages/meep/__init__.py"
        backend_policy = module.parent / "backend_policy.py"
        extension = module.parent / "_meep.so"
        python = self.prefix / "bin/python3.11"
        mpirun = self.prefix / "bin/mpirun"
        verifier = self.prefix / "bin/gpmeep-verify-provenance"
        verifier_source = (
            self.prefix / "share/gpmeep/libexec/gpmeep-verify-provenance.py"
        )
        libmeep = self.prefix / "lib/libmeep.so.38"
        conda_record = self.prefix / "conda-meta/gpmeep-1.1.0-0.json"
        manifest_file = self.prefix / "share/gpmeep/release.json"
        candidate_receipt = (
            self.prefix / "share/gpmeep/v110-candidate-build-receipt.json"
        )
        fontconfig = self.prefix / "etc/fonts/fonts.conf"
        child = self.prefix.joinpath(*EXPERIMENT.CHILD_RELATIVE.parts)
        for directory in {
            path.parent
            for path in (
                module, backend_policy, extension, python, mpirun, verifier,
                verifier_source,
                libmeep, conda_record, manifest_file, candidate_receipt,
                fontconfig, child,
            )
        }:
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pathlib.Path(sys.executable).resolve(), python)
        python.chmod(python.stat().st_mode | stat.S_IXUSR)
        shutil.copyfile(ROOT / "scripts/v110-inversion-qualified-child.py", child)
        active_expression = (
            repr("cpu")
            if behavior == "cpu_lie"
            else "os.environ.get('MEEP_GPU_BACKEND', 'cpu')"
        )
        device_expression = (
            repr("GPU-00000000-0000-0000-0000-000000009099")
            if behavior == "uuid_lie"
            else "os.environ.get('CUDA_VISIBLE_DEVICES', '')"
        )
        fake = (
            "import os, time\n"
            "from . import backend_policy\n"
            f"_statistics_keys = {CHILD.RUN_STATISTICS_KEYS!r}\n"
            "_calls = 0\n"
            "_availability_probes = 0\n"
            "_device_enumerations = 0\n"
            "_acknowledged = False\n"
            "_pending_scope_ack = False\n"
            "_scope_learned = False\n"
            "_recommendation = None\n"
            "class Simulation:\n"
            "    def __init__(self, *, resolution): self.resolution = resolution; self.fields = None\n"
            "    def run(self, *, until):\n"
            "        global _calls, _acknowledged, _pending_scope_ack, _scope_learned, _recommendation\n"
            "        _calls += 1\n"
            "        cuda = bool(os.environ.get('CUDA_VISIBLE_DEVICES'))\n"
            "        disabled = bool(os.environ.get('MEEP_GPU_DISABLE_CURL_PHASE_REPLAY') or os.environ.get('MEEP_GPU_DISABLE_PHASE_BATCHED_CURL') or os.environ.get('MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY'))\n"
            "        enabled = bool(os.environ.get('MEEP_GPU_ENABLE_CURL_PHASE_REPLAY'))\n"
            "        snapshot_disabled = bool(os.environ.get('MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT'))\n"
            "        boundary_disabled = 'MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY' in os.environ\n"
            "        scope_cache_disabled = bool(os.environ.get('MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE'))\n"
            "        exact_ack = False\n"
            "        reused_ack = False\n"
            f"        if {behavior == 'ack_scope_cache'!r}:\n"
            "            if _pending_scope_ack:\n"
            "                _pending_scope_ack = False\n"
            "                exact_ack = True\n"
            "                if not scope_cache_disabled: _scope_learned = True\n"
            "            elif scope_cache_disabled or not _scope_learned:\n"
            "                raise backend_policy.BackendRecommendationMismatch('challenge-1')\n"
            "            else:\n"
            "                reused_ack = True\n"
            f"        if {behavior == 'ablation_ack_drift'!r} and snapshot_disabled and not _acknowledged:\n"
            "            raise backend_policy.BackendRecommendationMismatch('challenge-1')\n"
            f"        if {behavior == 'ack_scope_cache'!r}:\n"
            "            delay = 0.03 if scope_cache_disabled else 0.005\n"
            f"        elif {behavior == 'probe_confounded'!r}:\n"
            "            delay = 0.005 if snapshot_disabled else 0.02\n"
            "        elif os.environ.get('MEEP_GPU_DISABLE_PHASE_MATERIAL_MIRROR_REUSE'):\n"
            "            delay = 0.15\n"
            "        elif os.environ.get('MEEP_GPU_DISABLE_BOUNDARY_SPAN_CACHE'):\n"
            "            delay = 0.15\n"
            "        elif boundary_disabled:\n"
            "            delay = 0.15\n"
            "        else:\n"
            "            delay = 0.10 if disabled else (0.005 if enabled else (0.07 if (cuda == (self.resolution < 2)) else 0.005))\n"
            f"        if {behavior == 'ack_scope_cache'!r}:\n"
            "            _recommendation = {\n"
            "                'schema': 'gpmeep-v110-backend-recommendation-v3',\n"
            "                'selected_backend': 'cuda', 'recommended_backend': 'cpu',\n"
            "                'reason_class': 'hardware_crossover',\n"
            "                'recommendation_confidence': 'high',\n"
            "                'acknowledgement': True, 'acknowledgement_allowed': True,\n"
            "                'acknowledgement_reason': 'receipt-bound inversion experiment intentionally exercises the selected backend',\n"
            "                'acknowledgement_reused': reused_ack,\n"
            "                'acknowledgement_scope': 'process',\n"
            "                'acknowledgement_origin_cohort_digest': 'd' * 64,\n"
            "                'acknowledgement_scope_cache_enabled': not scope_cache_disabled,\n"
            "                'acknowledgement_scope_digest': 'a' * 64,\n"
            "                'workload_fingerprint_evaluated': exact_ack,\n"
            "                'workload_fingerprint': 'c' * 64 if exact_ack else '',\n"
            "                'recommendation_policy_version': 'gpmeep-v110-explicit-backend-guard-v24',\n"
            "                'mpi_topology_digest': 'b' * 64,\n"
            "                'acknowledgement_digest': 'd' * 64 if exact_ack else '',\n"
            "            }\n"
            "        else:\n"
            "            selected = 'cuda' if cuda else 'cpu'\n"
            "            _recommendation = {\n"
            "                'schema': 'gpmeep-v110-backend-recommendation-v3',\n"
            "                'selected_backend': selected, 'recommended_backend': selected,\n"
            "                'reason_class': 'hardware_crossover',\n"
            "                'recommendation_confidence': 'high',\n"
            "                'acknowledgement': False, 'acknowledgement_allowed': True,\n"
            "                'acknowledgement_reason': '', 'acknowledgement_reused': False,\n"
            "                'acknowledgement_scope': '', 'acknowledgement_origin_cohort_digest': '',\n"
            "                'acknowledgement_scope_cache_enabled': not scope_cache_disabled,\n"
            "                'acknowledgement_scope_digest': 'a' * 64,\n"
            "                'workload_fingerprint_evaluated': False, 'workload_fingerprint': '',\n"
            "                'recommendation_policy_version': 'gpmeep-v110-explicit-backend-guard-v24',\n"
            "                'mpi_topology_digest': 'b' * 64, 'acknowledgement_digest': '',\n"
            "            }\n"
            "        time.sleep(delay * float(until))\n"
            "class GPU:\n"
            "    def reset_statistics(self):\n"
            "        global _calls, _availability_probes, _device_enumerations\n"
            "        _calls = 0; _availability_probes = 0; _device_enumerations = 0\n"
            "    def devices(self):\n"
            "        global _availability_probes, _device_enumerations\n"
            "        _availability_probes += 1\n"
            "        count = int(bool(os.environ.get('CUDA_VISIBLE_DEVICES')))\n"
            "        if count:\n"
            "            legacy = bool(os.environ.get('MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT'))\n"
            "            _device_enumerations += 1 + 10 * count if legacy else 1\n"
            f"            if {behavior == 'probe_confounded'!r} and legacy:\n"
            "                time.sleep(0.10)\n"
            f"        if {behavior == 'inventory_drift'!r}:\n"
            "            state = 'off' if os.environ.get('MEEP_GPU_DISABLE_DEVICE_INVENTORY_SNAPSHOT') else 'on'\n"
            "            return [{'state': state} for _ in range(count)]\n"
            "        return [{} for _ in range(count)]\n"
            "    def acknowledge_backend_mismatch(self, challenge, reason, scope='simulation'):\n"
            "        global _acknowledged, _pending_scope_ack\n"
            "        if challenge != 'challenge-1' or len(reason) < 8 or scope != 'process': raise ValueError('bad acknowledgement')\n"
            "        _acknowledged = True\n"
            "        _pending_scope_ack = True\n"
            "    @property\n"
            "    def requested_backend(self): return os.environ.get('MEEP_GPU_BACKEND', 'cpu')\n"
            "    @property\n"
            f"    def active_backend(self): return {active_expression}\n"
            "    @property\n"
            f"    def selected_device_identifier(self): return {device_expression}\n"
            "    @property\n"
            "    def recommendation(self): return dict(_recommendation) if _recommendation else None\n"
            "    def statistics(self):\n"
            "        active = self.active_backend\n"
            "        values = {group: dict.fromkeys(names, 0) for group, names in _statistics_keys.items()}\n"
            "        for group, stem in [('dispatch','curl'),('field_updates','update_eh'),('polarizations','polarization'),('sources','source'),('boundaries','boundary'),('dfts','dft')]:\n"
            "            for backend in ('cpu','cuda'):\n"
            f"                amount = _calls if backend == active or ({behavior == 'fallback'!r} and backend != active) else 0\n"
            "                values[group][backend + '_' + stem + '_calls'] = amount\n"
            "                values[group][backend + '_' + stem + '_points'] = amount * 64\n"
            "        replay_disabled = bool(os.environ.get('MEEP_GPU_DISABLE_CURL_PHASE_REPLAY'))\n"
            "        replay = int(bool(os.environ.get('MEEP_GPU_ENABLE_CURL_PHASE_REPLAY')) and not replay_disabled "
            f"and {behavior != 'no_replay_hit'!r}) * _calls\n"
            "        forced = int(bool(os.environ.get('MEEP_GPU_ENABLE_PHASE_BATCHED_CURL'))) * _calls\n"
            "        values['phase_batch_policy'].update({\n"
            "            'phase_curl_replay_checks': replay, 'phase_curl_replay_hits': replay,\n"
            "            'phase_curl_replay_unready': 0, 'phase_curl_replay_generation_misses': 0,\n"
            "            'phase_curl_replay_mirror_misses': 0, 'phase_curl_forced_batches': forced})\n"
            f"        if active == 'cuda' and {behavior in ('phase_material_reuse', 'phase_material_reuse_no_hits')!r}:\n"
            "            phasing = values['material_phasing']\n"
            "            phasing['checks'] = 5 * _calls\n"
            "            if 'MEEP_GPU_DISABLE_PHASE_MATERIAL_MIRROR_REUSE' in os.environ:\n"
            "                phasing['disabled'] = 5 * _calls\n"
            f"            elif {behavior == 'phase_material_reuse_no_hits'!r}:\n"
            "                phasing['ineligible'] = 5 * _calls\n"
            "            else:\n"
            "                phasing.update(reused=4*_calls, ineligible=_calls)\n"
            "        if active == 'cuda':\n"
            "            boundary_disabled = 'MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY' in os.environ\n"
            "            boundary = values['boundary_execution']\n"
            "            boundary.update(local_checks=_calls, prepare_calls=_calls, prepared_arrays=10*_calls, prepare_host_nanoseconds=100*_calls, plan_direct_local=_calls)\n"
            "            if 'MEEP_GPU_DISABLE_BOUNDARY_SPAN_CACHE' in os.environ:\n"
            "                boundary.update(span_cache_bypasses=10*_calls)\n"
            f"            elif {behavior != 'span_cache_no_hits'!r}:\n"
            "                boundary.update(span_cache_hits=9*_calls, span_cache_misses=_calls)\n"
            "            if boundary_disabled:\n"
            "                boundary.update(local_disabled=_calls, local_descriptor_rebuild_attempts=_calls)\n"
            f"            elif {behavior == 'boundary_no_replay'!r}:\n"
            "                boundary.update(local_replay_empty=_calls)\n"
            "            else:\n"
            "                boundary.update(local_replay_work=_calls)\n"
            "        values['runtime'] = {\n"
            "            'runtime_availability_probes': _availability_probes,\n"
            "            'runtime_device_enumerations': _device_enumerations,\n"
            "            'runtime_device_selections': 0}\n"
            "        return values\n"
            "gpu = GPU()\n"
        )
        module.write_text(fake)
        backend_policy.write_text(
            "class BackendRecommendationMismatch(Exception):\n"
            "    def __init__(self, challenge):\n"
            "        super().__init__('backend recommendation differs')\n"
            "        self.challenge = challenge\n"
        )
        for path, payload in (
            (extension, b"extension\n"),
            (libmeep, b"libmeep\n"),
            (mpirun, b"#!/bin/sh\nexit 0\n"),
            (verifier, b"#!/bin/sh\nexit 0\n"),
            (verifier_source, b"# fixture installed verifier\n"),
            (manifest_file, b"{}\n"),
            (candidate_receipt, b'{"build_mode":"development"}\n'),
            (fontconfig, b"fonts\n"),
        ):
            path.write_bytes(payload)
        for path in (mpirun, verifier):
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        libmeep_alias = self.prefix / "lib/libmeep.so"
        libmeep_alias.symlink_to(libmeep.name)
        self.package = (
            self.root / f"gpmeep-1.1.0-fixture-{behavior}_development.conda"
        )
        package_paths = (
            module,
            backend_policy,
            extension,
            libmeep,
            verifier,
            verifier_source,
            manifest_file,
            candidate_receipt,
            child,
        )
        package_payload = {
            path.relative_to(self.prefix).as_posix(): (
                path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
            )
            for path in package_paths
        }
        package_softlinks = {
            libmeep_alias.relative_to(self.prefix).as_posix(): (
                os.readlink(libmeep_alias),
                stat.S_IMODE(libmeep_alias.lstat().st_mode),
            )
        }
        paths_data = self._write_conda(
            self.package, package_payload, package_softlinks
        )
        package_sha256 = hashlib.sha256(self.package.read_bytes()).hexdigest()
        self._write_nonpublishable_marker(package_sha256)
        conda_record.write_text(
            json.dumps(
                {
                    "name": "gpmeep",
                    "version": "1.1.0",
                    "build": "cuda124_mpi_openmpi_py311_0_development",
                    "build_number": 0,
                    "subdir": "linux-64",
                    "sha256": package_sha256,
                    "url": self.package.as_uri(),
                    "files": [item["_path"] for item in paths_data],
                    "paths_data": {"paths_version": 1, "paths": paths_data},
                }
            )
        )
        self.provenance = self.root / f"installed-{behavior}.json"
        report = {
            "schema_version": 1,
            "state": "PASS",
            "prefix": str(self.prefix),
            "conda_record": self.record(conda_record),
            "package_url": self.package.as_uri(),
            "package_sha256": package_sha256,
            "ownership": {
                "owned_paths": len(paths_data),
                "regular_files": len(package_payload),
                "symlinks": len(package_softlinks),
                "content_hashes_verified": len(paths_data),
                "content_sizes_verified": len(paths_data),
                "prefix_rewritten_sizes": 0,
            },
            "manifest": {
                "distribution": "gpmeep", "distribution_version": "1.1.0",
                "source_commit": self.source_commit, "precision": "fp32",
                "mpi": "openmpi", "cuda_fast_math": False, "scheme": False,
                "file": self.record(manifest_file),
            },
            "runtime": {
                "python": self.record(python), "module": self.record(module),
                "extension": self.record(extension), "libmeep": self.record(libmeep),
                "meep_version": "1.31.0", "cuda_compiled": True,
                "mpi_enabled": True, "single_precision": True,
            },
            "v110_candidate_gate": {
                "build_mode": "development", "state": "DEVELOPMENT_ONLY",
                "source_commit": self.source_commit, "receipt_id": "a" * 64,
            },
        }
        report["report_id"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.provenance.write_text(json.dumps(report, sort_keys=True))
        self.runtime_id = report["report_id"]
        installed_report, installed = IDENTITY.validate_report(
            self.provenance, self.package, self.prefix, self.source_commit
        )
        runtime = IDENTITY.runtime_paths(installed_report, installed)
        self.archive_verification, _payload, _semantics = (
            EXPERIMENT._installed_archive_authority(
                repo=self.repo,
                package=self.package,
                prefix=self.prefix,
                report=installed_report,
                runtime=runtime,
            )
        )
        self.live_verification = {
            "report_id": self.runtime_id,
            "candidate_receipt_id": "a" * 64,
            "build_mode": "development",
            "package_sha256": package_sha256,
            "verifier": self.record(verifier),
        }
        self.m4_output = self.root / f"m4-{behavior}"
        self.m4_output.mkdir()
        (self.m4_output / "M4_COMPLETE").write_text("{}\n")
        runtime_paths = {
            name: str(runtime[name])
            for name in (
                "python", "build_python", "install_prefix", "build_receipt",
                "mpiexec", "package", "provenance_report", "verifier",
            )
        }
        self.m4_report = {
            "schema": "gpmeep-v110-m4-equal-work-matrix-v1",
            "outcome": "PASS",
            "contract": {
                "paths": {
                    "repo": str(self.repo),
                    "v110_package": str(self.package),
                    "v110_provenance_report": str(self.provenance),
                },
                "runtime_inputs": {
                    "v110_package": self.record(self.package),
                    "v110_provenance_report": self.record(self.provenance),
                    "v110_package_sha256": package_sha256,
                    "v110_installed_report_id": self.runtime_id,
                    "v110_candidate_receipt_id": "a" * 64,
                    "v110_installed_runtime": runtime_paths,
                    "v110_live_verification": self.live_verification,
                    "v110_archive_verification": self.archive_verification,
                    "v110_source_identity": {"git_head": self.source_commit},
                },
            },
        }

    def tearDown(self) -> None:
        self.registry_patch.stop()
        self.experiment_patch.stop()
        self.m4_patch.stop()
        self.temporary.cleanup()

    def sample(
        self,
        topology: str,
        cycle: int,
        arguments: list[str] | None = None,
        *,
        environment_label: str = "baseline",
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.count += 1
        output = self.root / f"sample-{self.count:02d}"
        devices = [] if topology.startswith("cpu") else [self.gpu]
        report = EXPERIMENT.create(
            output=output,
            repo=self.repo,
            source_commit=self.source_commit,
            runtime_provenance=self.provenance,
            package=self.package,
            install_prefix=self.prefix,
            m4_output=self.m4_output,
            workload_id=self.workload,
            cycle_index=cycle,
            topology=topology,
            gpu_devices=devices,
            environment_label=environment_label,
            experiment_arguments=arguments or [],
            timeout_seconds=5,
        )
        report_path = output / "report.json"
        reference = {
            "kind": "experiment_sample",
            "schema": EXPERIMENT.SCHEMA,
            "cycle_index": cycle,
            "path": report_path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
        return reference, report

    @staticmethod
    def identities(
        inputs: list[dict[str, object]], reports: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return [
            {
                "kind": reference["kind"],
                "cycle_index": reference["cycle_index"],
                "sha256": reference["sha256"],
                "report_id": report["report_id"],
                "topology": report["topology"],
                "gpu_devices": report["gpu_devices"],
                "lane_devices": report["lane"]["spec"]["devices"],
                "environment_sha256": report["lane"]["environment_sha256"],
                "primary_seconds": report["timing"]["primary_seconds"],
                "timing_contract": report["timing"]["contract"],
                "run_count": report["timing"]["run_count"],
                "command_sha256": hashlib.sha256(
                    json.dumps(report["lane"]["command"], separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for reference, report in zip(inputs, reports, strict=True)
        ]

    def expected_result(
        self,
        kind: str,
        arguments: dict[str, object],
        inputs: list[dict[str, object]],
        reports: list[dict[str, object]],
    ) -> dict[str, object]:
        identities = self.identities(inputs, reports)
        cycles = [1, 2]
        topologies = sorted({report["topology"] for report in reports})
        if kind == "timing":
            semantics = {
                "topologies": topologies,
                "mean_primary_seconds": {
                    topology: math.fsum(
                        item["primary_seconds"]
                        for item in identities
                        if item["topology"] == topology
                    )
                    / sum(item["topology"] == topology for item in identities)
                    for topology in topologies
                },
            }
        elif kind == "phase_profile":
            semantics = {
                "phase_records": [
                    {
                        "report_id": report["receipt_id"],
                        "available_seconds": {
                            name: report["phase_profile"]["phases"][name]["seconds"]
                            for name in sorted(REGISTRY.PHASES)
                            if report["phase_profile"]["phases"][name]["available"]
                        },
                    }
                    for report in reports
                ]
            }
        elif kind == "size_run_sweep":
            points: dict[tuple[float, float, str], list[float]] = {}
            for report in reports:
                command = report["lane"]["command"]
                size = float(command[command.index(arguments["size_flag"]) + 1])
                run = float(command[command.index(arguments["run_length_flag"]) + 1])
                backend = "cpu" if report["topology"].startswith("cpu") else "cuda"
                points.setdefault((size, run, backend), []).append(
                    report["timing"]["primary_seconds"]
                )
            sizes = sorted({key[0] for key in points})
            runs = sorted({key[1] for key in points})
            normalized = list(reports[0]["lane"]["command"])
            normalized[normalized.index(arguments["size_flag"]) + 1] = "<size>"
            normalized[normalized.index(arguments["run_length_flag"]) + 1] = "<run-length>"
            semantics = {
                "sizes": sizes,
                "run_lengths": runs,
                "cpu_over_cuda": {
                    f"{size:g}:{run:g}": math.fsum(points[(size, run, "cpu")])
                    / len(points[(size, run, "cpu")])
                    / (
                        math.fsum(points[(size, run, "cuda")])
                        / len(points[(size, run, "cuda")])
                    )
                    for size in sizes
                    for run in runs
                },
                "runtime_id": self.runtime_id,
                "normalized_command_sha256": hashlib.sha256(
                    json.dumps(normalized, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        elif kind in {"profiler_or_ablation", "same_binary_ablation"}:
            flag = arguments["intervention_flag"]
            intervention = arguments["intervention"]
            states = {"off": [], "on": []}
            paired = {1: {}, 2: {}}
            normalized = []
            for report in reports:
                command = report["lane"]["command"]
                state = command[command.index(flag) + 1].rsplit("=", 1)[1]
                states[state].append(report["timing"]["primary_seconds"])
                paired[report["cycle_index"]][state] = report["timing"]["primary_seconds"]
                normalized = list(command)
                normalized[normalized.index(flag) + 1] = f"{intervention}=<state>"
            normalized_json = json.dumps(normalized, separators=(",", ":"))
            off_mean = math.fsum(states["off"]) / len(states["off"])
            on_mean = math.fsum(states["on"]) / len(states["on"])
            semantics = {
                "intervention": intervention,
                "runtime_id": self.runtime_id,
                "topology": "cuda1",
                "gpu_identity": {
                    "gpu_devices": [self.gpu],
                    "lane_devices": [self.gpu],
                },
                "environment_sha256": reports[0]["lane"]["environment_sha256"],
                "normalized_command_sha256": hashlib.sha256(
                    normalized_json.encode()
                ).hexdigest(),
                "timing_contract": reports[0]["timing"]["contract"],
                "run_count": reports[0]["timing"]["run_count"],
                "backend_mismatch_acknowledgements": {
                    state: next(
                        report["runtime_snapshot"]["feature_counters"][
                            "backend_mismatch_acknowledgements"
                        ]
                        for report in reports
                        if report["lane"]["command"][
                            report["lane"]["command"].index(flag) + 1
                        ].endswith(f"={state}")
                    )
                    for state in ("off", "on")
                },
                "off_seconds": states["off"],
                "on_seconds": states["on"],
                "relative_improvement": (off_mean - on_mean) / off_mean,
                "paired_relative_improvement": [
                    (paired[cycle]["off"] - paired[cycle]["on"])
                    / paired[cycle]["off"]
                    for cycle in (1, 2)
                ],
            }
            if intervention == "phase-material-mirror-reuse":
                semantics["material_phasing_by_cycle"] = {"1": {}, "2": {}}
                for report in reports:
                    command = report["lane"]["command"]
                    state = command[command.index(flag) + 1].rsplit("=", 1)[1]
                    observations = report["runtime_snapshot"]["run_observations"]
                    semantics["material_phasing_by_cycle"][str(report["cycle_index"])][state] = {
                        name: sum(item["statistics_after"]["material_phasing"][name]
                                  - item["statistics_before"]["material_phasing"][name]
                                  for item in observations)
                        for name in ("checks", "reused", "disabled", "ineligible")}
            if intervention in {"boundary-plan-replay", "boundary-span-cache"}:
                semantics["boundary_execution_by_cycle"] = {"1": {}, "2": {}}
                for report in reports:
                    command = report["lane"]["command"]
                    state = command[command.index(flag) + 1].rsplit("=", 1)[1]
                    observations = report["runtime_snapshot"]["run_observations"]
                    semantics["boundary_execution_by_cycle"][str(report["cycle_index"])][state] = {
                        key: sum(row["statistics_after"]["boundary_execution"][key]
                                 - row["statistics_before"]["boundary_execution"][key]
                                 for row in observations)
                        for key in CHILD.BOUNDARY_EXECUTION_KEYS
                    }
            if intervention == "device-inventory-snapshot":
                probe = reports[0]["runtime_snapshot"]["feature_counters"][
                    "device_inventory_probe"
                ]
                semantics["device_inventory"] = {
                    "device_count": probe["device_count"],
                    "inventory_sha256": probe["inventory_sha256"],
                }
            if intervention == "backend-ack-scope-cache":
                policy = reports[0]["runtime_snapshot"]["backend_policy_runs"][0]
                semantics["backend_policy_reuse"] = {
                    "schema": policy["schema"],
                    "policy_version": policy["recommendation_policy_version"],
                    "scope_digest": policy["acknowledgement_scope_digest"],
                    "mpi_topology_digest": policy["mpi_topology_digest"],
                    "reason_sha256": hashlib.sha256(
                        policy["acknowledgement_reason"].encode("utf-8")
                    ).hexdigest(),
                    "off": {"exact": reports[0]["timing"]["run_count"], "reused": 0},
                    "on": {
                        "exact": 1,
                        "reused": reports[0]["timing"]["run_count"] - 1,
                    },
                }
        else:
            commands = {
                json.dumps(report["lane"]["command"], separators=(",", ":"))
                for report in reports
            }
            semantics = {
                "environment_sha256": sorted(
                    {report["lane"]["environment_sha256"] for report in reports}
                ),
                "runtime_ids": [self.runtime_id],
                "topology": "cuda1",
                "command_sha256": hashlib.sha256(next(iter(commands)).encode()).hexdigest(),
            }
        return {
            "producer_input_count": len(inputs),
            "cycles": cycles,
            "producer_reports": identities,
            "semantics": semantics,
        }

    def verify_wrapper(
        self,
        kind: str,
        classification: str,
        arguments: dict[str, object],
        samples: list[tuple[dict[str, object], dict[str, object]]],
    ) -> dict[str, object]:
        inputs = [item[0] for item in samples]
        reports = [item[1] for item in samples]
        payload = {
            "schema": REGISTRY.EVIDENCE_SCHEMAS[kind],
            "state": "PASS",
            "kind": kind,
            "workload_id": self.workload,
            "source_campaign_id": "fixture-campaign",
            "source_inventory_sha256": "b" * 64,
            "classification": classification,
            "arguments": arguments,
            "inputs": inputs,
            "result": self.expected_result(kind, arguments, inputs, reports),
        }
        payload["evidence_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path = self.root / f"{kind}.json"
        path.write_text(json.dumps(payload, sort_keys=True))
        return REGISTRY._verify_typed_evidence(
            path,
            self.root,
            expected_kind=kind,
            expected_workload_id=self.workload,
            expected_campaign_id="fixture-campaign",
            expected_inventory_sha256="b" * 64,
            expected_classification=classification,
        )

    def test_experiment_producer_round_trip_and_log_tamper(self) -> None:
        _reference, report = self.sample("cuda1", 1)
        output = self.root / "sample-01"
        self.assertEqual(EXPERIMENT.verify_complete(output), report)
        (output / "stdout.log").write_text("tampered\n")
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "stdout differs"):
            EXPERIMENT.verify_complete(output)

    def test_qualified_child_explicitly_acknowledges_one_policy_mismatch(self) -> None:
        class Mismatch(Exception):
            def __init__(self, challenge: str):
                super().__init__("backend recommendation differs")
                self.challenge = challenge

        class Simulation:
            def __init__(self, *, resolution: float):
                self.resolution = resolution
                self.calls: list[float] = []

            def run(self, *, until: float):
                self.calls.append(until)
                if len(self.calls) == 1:
                    raise Mismatch("challenge-1")
                return "completed"

        acknowledgements: list[tuple[str, str, str]] = []
        gpu = types.SimpleNamespace(
            acknowledge_backend_mismatch=lambda challenge, reason, scope="simulation": acknowledgements.append(
                (challenge, reason, scope)
            ),
            recommendation={
                "schema": "gpmeep-v110-backend-recommendation-v3",
                "selected_backend": "cuda",
                "recommended_backend": "cpu",
                "reason_class": "hardware_crossover",
                "recommendation_confidence": "high",
                "acknowledgement": True,
                "acknowledgement_allowed": True,
                "acknowledgement_reason": "explicit mismatch test reason",
                "acknowledgement_reused": False,
                "acknowledgement_scope": "process",
                "acknowledgement_origin_cohort_digest": "d" * 64,
                "acknowledgement_scope_cache_enabled": True,
                "acknowledgement_scope_digest": "a" * 64,
                "workload_fingerprint_evaluated": True,
                "workload_fingerprint": "b" * 64,
                "recommendation_policy_version": (
                    "gpmeep-v110-explicit-backend-guard-v24"
                ),
                "mpi_topology_digest": "c" * 64,
                "acknowledgement_digest": "d" * 64,
            },
        )
        mp = types.SimpleNamespace(Simulation=Simulation, gpu=gpu)
        phase_seconds: list[float] = []
        recorded_reasons: list[str] = []
        policy_runs: list[dict[str, object]] = []

        CHILD._install_scaling(
            mp,
            2.0,
            3.0,
            phase_seconds,
            Mismatch,
            recorded_reasons,
            policy_runs,
        )
        simulation = mp.Simulation(resolution=4.0)
        result = simulation.run(until=5.0)

        self.assertEqual("completed", result)
        self.assertEqual(8.0, simulation.resolution)
        self.assertEqual([15.0, 15.0], simulation.calls)
        self.assertEqual(1, len(acknowledgements))
        self.assertEqual("challenge-1", acknowledgements[0][0])
        self.assertGreaterEqual(len(acknowledgements[0][1]), 8)
        self.assertEqual("process", acknowledgements[0][2])
        self.assertEqual([acknowledgements[0][1]], recorded_reasons)
        self.assertEqual(1, len(phase_seconds))
        self.assertEqual([gpu.recommendation], policy_runs)
        self.assertGreaterEqual(phase_seconds[0], 0.0)

    def test_serial_inversion_policy_receipts_reject_mode_and_lineage_substitution(
        self,
    ) -> None:
        exact = {
            "schema": "gpmeep-v110-backend-recommendation-v3",
            "selected_backend": "cuda",
            "recommended_backend": "cpu",
            "reason_class": "hardware_crossover",
            "recommendation_confidence": "high",
            "acknowledgement": True,
            "acknowledgement_allowed": True,
            "acknowledgement_reason": "explicit serial inversion receipt test",
            "acknowledgement_reused": False,
            "acknowledgement_scope": "process",
            "acknowledgement_origin_cohort_digest": "d" * 64,
            "acknowledgement_scope_cache_enabled": True,
            "acknowledgement_scope_digest": "a" * 64,
            "workload_fingerprint_evaluated": True,
            "workload_fingerprint": "b" * 64,
            "recommendation_policy_version": (
                "gpmeep-v110-explicit-backend-guard-v24"
            ),
            "mpi_topology_digest": "c" * 64,
            "acknowledgement_digest": "d" * 64,
        }
        reused = dict(exact)
        reused.update(
            {
                "acknowledgement_reused": True,
                "workload_fingerprint_evaluated": False,
                "workload_fingerprint": "",
                "acknowledgement_digest": "",
            }
        )
        self.assertEqual(
            [exact, reused],
            EXPERIMENT._backend_policy_runs(
                [exact, reused], run_count=2, expected_backend="cuda"
            ),
        )
        self.assertEqual(
            [exact, reused],
            CHILD._validate_backend_policy_run_sequence([exact, reused]),
        )

        wrong_first_origin = dict(exact)
        wrong_first_origin["acknowledgement_origin_cohort_digest"] = "e" * 64
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "exact acknowledgement differs"
        ):
            EXPERIMENT._backend_policy_runs(
                [wrong_first_origin], run_count=1, expected_backend="cuda"
            )

        forged_serial_exact_reuse = dict(reused)
        forged_serial_exact_reuse.update(
            {
                "workload_fingerprint_evaluated": True,
                "workload_fingerprint": "b" * 64,
                "acknowledgement_digest": "d" * 64,
            }
        )
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "reused acknowledgement differs"
        ):
            EXPERIMENT._backend_policy_runs(
                [exact, forged_serial_exact_reuse],
                run_count=2,
                expected_backend="cuda",
            )

        wrong_reuse_origin = dict(reused)
        wrong_reuse_origin["acknowledgement_origin_cohort_digest"] = "e" * 64
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "lineage differs"
        ):
            EXPERIMENT._backend_policy_runs(
                [exact, wrong_reuse_origin],
                run_count=2,
                expected_backend="cuda",
            )
        with self.assertRaisesRegex(CHILD.ChildError, "lineage differs"):
            CHILD._validate_backend_policy_run_sequence(
                [exact, wrong_reuse_origin]
            )

    def test_device_inventory_snapshot_ablation_proves_native_counts(self) -> None:
        inventory_digests: set[str] = set()
        for state, expected_enumerations in (("off", 11), ("on", 1)):
            _reference, report = self.sample(
                "cuda1",
                1,
                [
                    "--gpmeep-ablation",
                    f"device-inventory-snapshot={state}",
                ],
            )
            output = self.root / f"sample-{self.count:02d}"
            terminal = json.loads(
                (output / report["child_runtime"]["path"]).read_text()
            )
            probe = dict(terminal["feature_counters"]["device_inventory_probe"])
            inventory_digest = probe.pop("inventory_sha256")
            self.assertRegex(inventory_digest, r"^[0-9a-f]{64}$")
            inventory_digests.add(inventory_digest)
            self.assertEqual(
                {
                    "device_count": 1,
                    "runtime_availability_probes": 1,
                    "runtime_device_enumerations": expected_enumerations,
                },
                probe,
            )
            self.assertEqual(
                terminal["timing"]["primary_seconds"],
                report["timing"]["primary_seconds"],
            )
            self.assertEqual(
                terminal["timing"]["run_count"],
                report["timing"]["run_count"],
            )
        self.assertEqual(1, len(inventory_digests))

    def test_backend_ack_scope_cache_ablation_is_allowlisted(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            child_off = CHILD._apply_ablation("backend-ack-scope-cache=off")
            self.assertEqual(
                {"MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE": "1"},
                child_off,
            )
            self.assertEqual(
                "1", os.environ["MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE"]
            )
            child_on = CHILD._apply_ablation("backend-ack-scope-cache=on")
            self.assertEqual({}, child_on)
            self.assertNotIn(
                "MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE", os.environ
            )
        arguments, parsed = EXPERIMENT._experiment_arguments(
            ["--gpmeep-ablation", "backend-ack-scope-cache=off"]
        )
        self.assertEqual(
            ["--gpmeep-ablation", "backend-ack-scope-cache=off"],
            arguments,
        )
        self.assertEqual(
            {"MEEP_GPU_DISABLE_BACKEND_ACK_SCOPE_CACHE": "1"},
            parsed["ablation_environment"],
        )

    def test_backend_ack_scope_cache_ablation_proves_reuse(self) -> None:
        (self.repo / self.workload).write_text(
            "import meep as mp\n"
            "simulation = mp.Simulation(resolution=1.0)\n"
            "simulation.run(until=1.0)\n"
            "simulation.run(until=1.0)\n"
            "simulation.run(until=1.0)\n"
        )
        subprocess.run(["git", "-C", str(self.repo), "add", self.workload], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "three-run fixture"],
            check=True,
        )
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        # Rebuild the package authority so the runtime provenance is bound to
        # the modified three-run workload's exact source commit.
        self._build_runtime("ack_scope_cache")

        samples = []
        by_state: dict[str, list[dict[str, object]]] = {"off": [], "on": []}
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                sample = self.sample(
                    "cuda1",
                    cycle,
                    [
                        "--gpmeep-ablation",
                        f"backend-ack-scope-cache={state}",
                    ],
                )
                samples.append(sample)
                by_state[state].append(sample[1])

        result = self.verify_wrapper(
            "same_binary_ablation",
            "engine_overhead",
            {
                "intervention_flag": "--gpmeep-ablation",
                "intervention": "backend-ack-scope-cache",
                "minimum_cycles": 2,
                "require_reversed_order": True,
            },
            samples,
        )
        self.assertEqual("PASS", result["state"])
        for report in by_state["off"]:
            self.assertEqual(3, report["timing"]["run_count"])
            self.assertEqual(
                3,
                report["runtime_snapshot"]["feature_counters"][
                    "backend_mismatch_acknowledgements"
                ],
            )
        for report in by_state["on"]:
            self.assertEqual(3, report["timing"]["run_count"])
            self.assertEqual(
                1,
                report["runtime_snapshot"]["feature_counters"][
                    "backend_mismatch_acknowledgements"
                ],
            )

    def test_backend_ack_scope_cache_rejects_stale_fresh_telemetry(self) -> None:
        (self.repo / self.workload).write_text(
            "import meep as mp\n"
            "simulation = mp.Simulation(resolution=1.0)\n"
            "simulation.run(until=1.0)\n"
            "simulation.run(until=1.0)\n"
            "simulation.run(until=1.0)\n"
        )
        subprocess.run(["git", "-C", str(self.repo), "add", self.workload], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "three-run fixture"],
            check=True,
        )
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        self._build_runtime("ack_scope_cache")
        originals = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                reference, report = self.sample(
                    "cuda1",
                    cycle,
                    [
                        "--gpmeep-ablation",
                        f"backend-ack-scope-cache={state}",
                    ],
                )
                originals.append((reference, report, state))

        for mutation, expected_error in (
            ("stale-fresh", "cache ON did not use audited reuse"),
            ("scope-drift", "reuse identity differs"),
        ):
            replayed: dict[str, dict[str, object]] = {}
            samples = []
            for reference, report, state in originals:
                altered = json.loads(json.dumps(report))
                if state == "on" and mutation == "stale-fresh":
                    fresh = altered["runtime_snapshot"]["backend_policy_runs"][0]
                    altered["runtime_snapshot"]["backend_policy_runs"] = [
                        dict(fresh) for _ in range(3)
                    ]
                elif state == "on" and mutation == "scope-drift":
                    altered["runtime_snapshot"]["backend_policy_runs"][1][
                        "acknowledgement_scope_digest"
                    ] = "e" * 64
                replayed[
                    pathlib.PurePosixPath(reference["path"]).parent.name
                ] = altered
                samples.append((reference, altered))

            with self.subTest(mutation=mutation), mock.patch.object(
                EXPERIMENT,
                "verify_complete",
                side_effect=lambda output: replayed[pathlib.Path(output).name],
            ):
                with self.assertRaisesRegex(
                    REGISTRY.RegistryError,
                    expected_error,
                ):
                    self.verify_wrapper(
                        "same_binary_ablation",
                        "engine_overhead",
                        {
                            "intervention_flag": "--gpmeep-ablation",
                            "intervention": "backend-ack-scope-cache",
                            "minimum_cycles": 2,
                            "require_reversed_order": True,
                        },
                        samples,
                    )

    def test_inventory_probe_cannot_create_a_false_performance_benefit(self) -> None:
        self._build_runtime("probe_confounded")
        samples = []
        by_cycle: dict[int, dict[str, dict[str, object]]] = {1: {}, 2: {}}
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                sample = self.sample(
                    "cuda1",
                    cycle,
                    [
                        "--gpmeep-ablation",
                        f"device-inventory-snapshot={state}",
                    ],
                )
                samples.append(sample)
                by_cycle[cycle][state] = sample[1]
        for cycle in (1, 2):
            off = by_cycle[cycle]["off"]["timing"]
            on = by_cycle[cycle]["on"]["timing"]
            self.assertGreater(off["process_seconds"], on["process_seconds"])
            self.assertLess(off["primary_seconds"], on["primary_seconds"])
        with self.assertRaisesRegex(
            REGISTRY.RegistryError,
            "intervention has no repeatable measured benefit",
        ):
            self.verify_wrapper(
                "same_binary_ablation",
                "engine_overhead",
                {
                    "intervention_flag": "--gpmeep-ablation",
                    "intervention": "device-inventory-snapshot",
                    "minimum_cycles": 2,
                    "require_reversed_order": True,
                },
                samples,
            )

    def test_same_binary_ablation_rejects_acknowledgement_path_drift(self) -> None:
        self._build_runtime("ablation_ack_drift")
        samples = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                samples.append(
                    self.sample(
                        "cuda1",
                        cycle,
                        [
                            "--gpmeep-ablation",
                            f"device-inventory-snapshot={state}",
                        ],
                    )
                )
        with self.assertRaisesRegex(
            REGISTRY.RegistryError,
            "OFF/ON acknowledgement path differs",
        ):
            self.verify_wrapper(
                "same_binary_ablation",
                "engine_overhead",
                {
                    "intervention_flag": "--gpmeep-ablation",
                    "intervention": "device-inventory-snapshot",
                    "minimum_cycles": 2,
                    "require_reversed_order": True,
                },
                samples,
            )

    def test_same_binary_ablation_rejects_public_inventory_drift(self) -> None:
        self._build_runtime("inventory_drift")
        samples = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                samples.append(
                    self.sample(
                        "cuda1",
                        cycle,
                        [
                            "--gpmeep-ablation",
                            f"device-inventory-snapshot={state}",
                        ],
                    )
                )
        with self.assertRaisesRegex(
            REGISTRY.RegistryError,
            "OFF/ON public device inventory differs",
        ):
            self.verify_wrapper(
                "same_binary_ablation",
                "engine_overhead",
                {
                    "intervention_flag": "--gpmeep-ablation",
                    "intervention": "device-inventory-snapshot",
                    "minimum_cycles": 2,
                    "require_reversed_order": True,
                },
                samples,
            )

    def test_timing_rejects_cpu_cuda_simulation_run_count_drift(self) -> None:
        # Count workload entrypoints, not recursive calls inside an original
        # method while the authenticated Simulation class is restored.
        (self.repo / self.workload).write_text(
            "import meep as mp\n"
            "simulation = mp.Simulation(resolution=1.0)\n"
            "simulation.run(until=1.0)\n"
            "if mp.gpu.active_backend == 'cuda':\n"
            "    simulation.run(until=1.0)\n"
        )
        subprocess.run(["git", "-C", str(self.repo), "add", self.workload], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "unequal run fixture"], check=True)
        self.source_commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        self._build_runtime("run_count_drift")
        samples = [
            self.sample(topology, cycle)
            for cycle in (1, 2)
            for topology in ("cpu1", "cuda1")
        ]
        self.assertEqual(
            {1, 2},
            {sample[1]["timing"]["run_count"] for sample in samples},
        )
        with self.assertRaisesRegex(
            REGISTRY.RegistryError,
            "experiment CPU/CUDA workload count differs",
        ):
            self.verify_wrapper(
                "timing",
                "unknown",
                {"comparison": "cpu_cuda_equal_work", "minimum_cycles": 2},
                samples,
            )

    def test_four_fabricated_cuda_ablation_samples_reject_placeholder_package(self) -> None:
        self._fabricate_placeholder_authority()
        states = ((1, "off"), (1, "on"), (2, "on"), (2, "off"))
        for index, (cycle, state) in enumerate(states, start=1):
            output = self.root / f"forged-ablation-{index}"
            with self.assertRaisesRegex(
                EXPERIMENT.ExperimentError, "genuine Conda archive"
            ):
                EXPERIMENT.create(
                    output=output,
                    repo=self.repo,
                    source_commit=self.source_commit,
                    runtime_provenance=self.provenance,
                    package=self.package,
                    install_prefix=self.prefix,
                    m4_output=self.m4_output,
                    workload_id=self.workload,
                    cycle_index=cycle,
                    topology="cuda1",
                    gpu_devices=[self.gpu],
                    environment_label="baseline",
                    experiment_arguments=[
                        "--gpmeep-ablation",
                        f"curl-phase-replay={state}",
                    ],
                    timeout_seconds=5,
                )
            self.assertFalse(output.exists())

    def test_sample_rejects_package_identity_not_bound_by_m4(self) -> None:
        self.m4_report["contract"]["runtime_inputs"]["v110_package_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "final M4 package authority"
        ):
            EXPERIMENT.create(
                output=self.root / "wrong-m4-package",
                repo=self.repo,
                source_commit=self.source_commit,
                runtime_provenance=self.provenance,
                package=self.package,
                install_prefix=self.prefix,
                m4_output=self.m4_output,
                workload_id=self.workload,
                cycle_index=1,
                topology="cuda1",
                gpu_devices=[self.gpu],
                environment_label="baseline",
                experiment_arguments=[],
                timeout_seconds=5,
            )

    def test_archive_rejects_package_owned_verifier_not_equal_to_source(self) -> None:
        (self.repo / "scripts/verify-gpmeep-package.py").write_text(
            "# changed source verifier\n"
        )
        report, installed = IDENTITY.validate_report(
            self.provenance, self.package, self.prefix, self.source_commit
        )
        runtime = IDENTITY.runtime_paths(report, installed)
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "verifier differs from the exact source"
        ):
            EXPERIMENT._installed_archive_authority(
                repo=self.repo,
                package=self.package,
                prefix=self.prefix,
                report=report,
                runtime=runtime,
            )

    def test_archive_rejects_softlink_retarget_and_absolute_target(self) -> None:
        report, installed = IDENTITY.validate_report(
            self.provenance, self.package, self.prefix, self.source_commit
        )
        runtime = IDENTITY.runtime_paths(report, installed)
        alias = self.prefix / "lib/libmeep.so"
        target = self.prefix / "lib/libmeep.so.38"
        alias.unlink()
        alias.symlink_to(f"./{target.name}")
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "installed softlink is unsafe"
        ):
            EXPERIMENT._installed_archive_authority(
                repo=self.repo,
                package=self.package,
                prefix=self.prefix,
                report=report,
                runtime=runtime,
            )
        alias.unlink()
        alias.symlink_to(target)
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "installed softlink is unsafe"
        ):
            EXPERIMENT._installed_archive_authority(
                repo=self.repo,
                package=self.package,
                prefix=self.prefix,
                report=report,
                runtime=runtime,
            )
        alias.unlink()
        copy = self.prefix / "lib/libmeep-copy.so"
        copy.write_bytes(target.read_bytes())
        alias.symlink_to(copy.name)
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError,
            "installed softlink target differs from exact archive",
        ):
            EXPERIMENT._installed_archive_authority(
                repo=self.repo,
                package=self.package,
                prefix=self.prefix,
                report=report,
                runtime=runtime,
            )

    def test_archive_rejects_package_owned_mode_tamper(self) -> None:
        report, installed = IDENTITY.validate_report(
            self.provenance, self.package, self.prefix, self.source_commit
        )
        runtime = IDENTITY.runtime_paths(report, installed)
        backend_policy = (
            self.prefix / "lib/python3.11/site-packages/meep/backend_policy.py"
        )
        original_mode = stat.S_IMODE(backend_policy.stat().st_mode)
        tampered_mode = original_mode ^ stat.S_IXUSR
        backend_policy.chmod(tampered_mode)
        self.assertNotEqual(original_mode, stat.S_IMODE(backend_policy.stat().st_mode))
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError,
            "installed path mode differs from exact archive",
        ):
            EXPERIMENT._installed_archive_authority(
                repo=self.repo,
                package=self.package,
                prefix=self.prefix,
                report=report,
                runtime=runtime,
            )

    def test_m4_authority_rejects_any_replay_tree_mutation(self) -> None:
        def mutate(_controller, output, _python, _prefix):
            (output / "unexpected").write_text("replay side effect\n")
            return self.m4_report

        with mock.patch.object(
            EXPERIMENT, "_isolated_m4_replay", side_effect=mutate
        ):
            with self.assertRaisesRegex(
                EXPERIMENT.ExperimentError, "evidence tree changed"
            ):
                self.sample("cuda1", 1)

    def test_cuda_evidence_rejects_queried_backend_uuid_and_fallback_lies(self) -> None:
        cases = (
            ("cpu_lie", "child runtime identity"),
            ("uuid_lie", "child runtime identity"),
            ("fallback", "forbidden backend fallback"),
        )
        for behavior, message in cases:
            with self.subTest(behavior=behavior):
                self._build_runtime(behavior)
                self.count += 1
                output = self.root / f"rejected-{self.count:02d}"
                with self.assertRaisesRegex(EXPERIMENT.ExperimentError, message):
                    EXPERIMENT.create(
                        output=output,
                        repo=self.repo,
                        source_commit=self.source_commit,
                        runtime_provenance=self.provenance,
                        package=self.package,
                        install_prefix=self.prefix,
                        m4_output=self.m4_output,
                        workload_id=self.workload,
                        cycle_index=1,
                        topology="cuda1",
                        gpu_devices=[self.gpu],
                        environment_label="baseline",
                        experiment_arguments=[],
                        timeout_seconds=5,
                    )
                self.assertFalse((output / "COMPLETE").exists())

    def test_arbitrary_or_forging_child_cannot_replace_package_owned_runner(self) -> None:
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'command'"):
            EXPERIMENT.create(
                output=self.root / "arbitrary",
                repo=self.repo,
                source_commit=self.source_commit,
                runtime_provenance=self.provenance,
                package=self.package,
                install_prefix=self.prefix,
                m4_output=self.m4_output,
                workload_id=self.workload,
                cycle_index=1,
                topology="cuda1",
                gpu_devices=[self.gpu],
                environment_label="baseline",
                experiment_arguments=[],
                timeout_seconds=5,
                command=["/bin/true"],
            )
        child = self.prefix.joinpath(*EXPERIMENT.CHILD_RELATIVE.parts)
        child.write_text("# forged child\n")
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError,
            "(qualified child differs|installed bytes differ from exact archive)",
        ):
            EXPERIMENT.create(
                output=self.root / "forged",
                repo=self.repo,
                source_commit=self.source_commit,
                runtime_provenance=self.provenance,
                package=self.package,
                install_prefix=self.prefix,
                m4_output=self.m4_output,
                workload_id=self.workload,
                cycle_index=1,
                topology="cuda1",
                gpu_devices=[self.gpu],
                environment_label="baseline",
                experiment_arguments=[],
                timeout_seconds=5,
            )

    def test_boundary_plan_replay_has_two_cycle_native_producers(self) -> None:
        samples = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                reference, report = self.sample(
                    "cuda1", cycle, ["--gpmeep-ablation", "boundary-plan-replay=" + state]
                )
                snapshots = report["runtime_snapshot"]["run_observations"]
                totals = EXPERIMENT._boundary_observation_totals(snapshots)
                self.assertGreater(totals["local_checks"], 0)
                if state == "off":
                    self.assertGreater(totals["local_disabled"], 0)
                    self.assertEqual(totals["local_replay_work"], 0)
                else:
                    self.assertGreater(totals["local_replay_work"], 0)
                    self.assertEqual(totals["local_disabled"], 0)
                samples.append((reference, report))
        arguments = {"intervention_flag": "--gpmeep-ablation", "intervention": "boundary-plan-replay",
                     "minimum_cycles": 2, "require_reversed_order": True}
        for kind in ("profiler_or_ablation", "same_binary_ablation"):
            self.verify_wrapper(kind, "engine_overhead", arguments, samples)

    def test_material_phasing_reuse_has_two_cycle_native_producers(self) -> None:
        self._build_runtime("phase_material_reuse")
        samples = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                reference, report = self.sample(
                    "cuda1", cycle, ["--gpmeep-ablation", "phase-material-mirror-reuse=" + state])
                totals = EXPERIMENT._validate_material_phasing_control(
                    "phase-material-mirror-reuse=" + state, "cuda",
                    report["runtime_snapshot"]["run_observations"])
                self.assertGreater(totals["disabled" if state == "off" else "reused"], 0)
                samples.append((reference, report))
        arguments = {"intervention_flag": "--gpmeep-ablation", "intervention": "phase-material-mirror-reuse",
                     "minimum_cycles": 2, "require_reversed_order": True}
        for kind in ("profiler_or_ablation", "same_binary_ablation"):
            self.verify_wrapper(kind, "engine_overhead", arguments, samples)

    def test_material_phasing_reuse_rejects_cpu_and_only_ineligible_steps(self) -> None:
        self._build_runtime("phase_material_reuse")
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            self.sample("cpu1", 1, ["--gpmeep-ablation", "phase-material-mirror-reuse=on"])
        failed = self.root / f"sample-{self.count:02d}"
        self.assertIn("requires CUDA1", (failed / "stderr.log").read_text())
        self.assertFalse((failed / "COMPLETE").exists())
        self._build_runtime("phase_material_reuse_no_hits")
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            self.sample("cuda1", 1, ["--gpmeep-ablation", "phase-material-mirror-reuse=on"])
        failed = self.root / f"sample-{self.count:02d}"
        self.assertIn("ON did not reuse stable mirrors", (failed / "stderr.log").read_text())
        self.assertFalse((failed / "COMPLETE").exists())

    def test_boundary_span_cache_has_two_cycle_native_producers(self) -> None:
        samples = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                reference, report = self.sample(
                    "cuda1", cycle, ["--gpmeep-ablation", "boundary-span-cache=" + state])
                totals = EXPERIMENT._boundary_observation_totals(
                    report["runtime_snapshot"]["run_observations"])
                self.assertGreater(totals["span_cache_bypasses" if state == "off" else "span_cache_hits"], 0)
                samples.append((reference, report))
        arguments = {"intervention_flag": "--gpmeep-ablation", "intervention": "boundary-span-cache",
                     "minimum_cycles": 2, "require_reversed_order": True}
        for kind in ("profiler_or_ablation", "same_binary_ablation"):
            self.verify_wrapper(kind, "engine_overhead", arguments, samples)

    def test_boundary_span_cache_rejects_cpu_and_zero_hits(self) -> None:
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            self.sample("cpu1", 1, ["--gpmeep-ablation", "boundary-span-cache=on"])
        failed = self.root / f"sample-{self.count:02d}"
        self.assertIn("requires CUDA1", (failed / "stderr.log").read_text())
        self.assertFalse((failed / "COMPLETE").exists())
        self._build_runtime("span_cache_no_hits")
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            self.sample("cuda1", 1, ["--gpmeep-ablation", "boundary-span-cache=on"])
        failed = self.root / f"sample-{self.count:02d}"
        self.assertIn("ON did not reuse validated spans", (failed / "stderr.log").read_text())
        self.assertFalse((failed / "COMPLETE").exists())

    def test_boundary_plan_replay_rejects_cpu_and_empty_replay(self) -> None:
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            self.sample("cpu1", 1, ["--gpmeep-ablation", "boundary-plan-replay=on"])
        failed = self.root / f"sample-{self.count:02d}"
        self.assertIn("requires CUDA1", (failed / "stderr.log").read_text())
        self.assertFalse((failed / "COMPLETE").exists())
        self._build_runtime("boundary_no_replay")
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            self.sample("cuda1", 1, ["--gpmeep-ablation", "boundary-plan-replay=on"])
        failed = self.root / f"sample-{self.count:02d}"
        self.assertIn("ON did not execute resident replay work", (failed / "stderr.log").read_text())
        self.assertFalse((failed / "COMPLETE").exists())

    def _reseal_modified_runtime(self, report, mutate):
        output = self.root / f"sample-{self.count:02d}"
        child_path = output / report["child_runtime"]["path"]
        child = json.loads(child_path.read_text())
        mutate(child)
        child.pop("terminal_id")
        child["terminal_id"] = EXPERIMENT._canonical_sha256(child)
        child_path.write_text(json.dumps(child, sort_keys=True) + "\n")
        report["child_runtime"] = EXPERIMENT._file_record(child_path, output)
        report["runtime_snapshot"]["child_runtime_terminal_id"] = child["terminal_id"]
        for key in ("backend_policy_runs", "feature_counters", "phase_numerical",
                    "run_observations", "workload_statistics"):
            report["runtime_snapshot"][key] = child[key]
        report.pop("report_id")
        report["report_id"] = EXPERIMENT._canonical_sha256(report)
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
        complete_path = output / "COMPLETE"
        complete = json.loads(complete_path.read_text())
        complete["report"] = EXPERIMENT._file_record(report_path, output)
        complete["report_id"] = report["report_id"]
        complete.pop("terminal_id")
        complete["terminal_id"] = EXPERIMENT._canonical_sha256(complete)
        complete_path.write_text(json.dumps(complete, sort_keys=True) + "\n")
        return output

    def test_rehashed_material_phasing_arm_contradiction_is_rejected(self) -> None:
        self._build_runtime("phase_material_reuse")
        for false_outcome in ("ineligible", "disabled"):
            with self.subTest(false_outcome=false_outcome):
                _, report = self.sample("cuda1", 1, ["--gpmeep-ablation", "phase-material-mirror-reuse=on"])
                def mutate(child):
                    snapshots = list(child["workload_statistics"].values())
                    for row in child["run_observations"]:
                        snapshots.extend((row["statistics_before"], row["statistics_after"]))
                    for snapshot in snapshots:
                        counters = snapshot["material_phasing"]
                        counters.update(reused=0, disabled=0, ineligible=0)
                        counters[false_outcome] = counters["checks"]
                output = self._reseal_modified_runtime(report, mutate)
                with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "ON did not reuse stable mirrors"):
                    EXPERIMENT.verify_complete(output)

    def test_rehashed_span_cache_accounting_contradiction_is_rejected(self) -> None:
        _, report = self.sample("cuda1", 1, ["--gpmeep-ablation", "boundary-span-cache=on"])
        def mutate(child):
            snapshots = list(child["workload_statistics"].values())
            for row in child["run_observations"]:
                snapshots.extend((row["statistics_before"], row["statistics_after"]))
            for snapshot in snapshots:
                snapshot["boundary_execution"]["span_cache_hits"] *= 1000
        output = self._reseal_modified_runtime(report, mutate)
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "outcomes exceed prepared arrays"):
            EXPERIMENT.verify_complete(output)

    def test_rehashed_raw_curl_feature_contradiction_is_rejected(self) -> None:
        _, report = self.sample("cuda1", 1, ["--gpmeep-ablation", "curl-phase-replay=on"])
        def mutate(child):
            snapshots = list(child["workload_statistics"].values())
            for row in child["run_observations"]:
                snapshots.extend((row["statistics_before"], row["statistics_after"]))
            for snapshot in snapshots:
                for key in snapshot["phase_batch_policy"]:
                    if key.startswith("phase_curl_replay_") or key == "phase_curl_forced_batches":
                        snapshot["phase_batch_policy"][key] = 0
        output = self._reseal_modified_runtime(report, mutate)
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "workload statistics and curl feature"):
            EXPERIMENT.verify_complete(output)

    def test_rehashed_global_dispatch_contradiction_is_rejected(self) -> None:
        _, report = self.sample("cuda1", 1)
        def mutate(child):
            child["workload_statistics"]["after"]["dispatch"]["cuda_curl_calls"] += 1
        output = self._reseal_modified_runtime(report, mutate)
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "workload statistics and dispatch"):
            EXPERIMENT.verify_complete(output)

    def test_curl_phase_replay_on_requires_native_replay_hit(self) -> None:
        self._build_runtime("no_replay_hit")
        output = self.root / "missing-replay-hit"
        with self.assertRaisesRegex(EXPERIMENT.ExperimentError, "command failed"):
            EXPERIMENT.create(
                output=output,
                repo=self.repo,
                source_commit=self.source_commit,
                runtime_provenance=self.provenance,
                package=self.package,
                install_prefix=self.prefix,
                m4_output=self.m4_output,
                workload_id=self.workload,
                cycle_index=1,
                topology="cuda1",
                gpu_devices=[self.gpu],
                environment_label="baseline",
                experiment_arguments=[
                    "--gpmeep-ablation",
                    "curl-phase-replay=on",
                ],
                timeout_seconds=5,
            )
        self.assertFalse((output / "COMPLETE").exists())

    def test_nominal_sweep_cannot_close_a_family_without_actual_work_and_numerics(self) -> None:
        sweep = []
        for size in (1, 2, 3):
            for run in (1, 2):
                for cycle in (1, 2):
                    for topology in ("cpu1", "cuda1"):
                        sweep.append(
                            self.sample(
                                topology,
                                cycle,
                                [
                                    "--gpmeep-sweep-size",
                                    str(size),
                                    "--gpmeep-sweep-run-length",
                                    str(run),
                                ],
                            )
                        )
        arguments = {
            "size_flag": "--gpmeep-sweep-size",
            "run_length_flag": "--gpmeep-sweep-run-length",
            "minimum_sizes": 3,
            "minimum_run_lengths": 2,
            "require_observed_crossover": True,
        }
        # These dedicated producer reports satisfy the nominal 3x2 matrix,
        # two-cycle backend coverage, timing crossover, source and runtime
        # checks. They do not establish actual grids/steps or paired numerics.
        message = "actual-work and numerical family verifier"
        with self.assertRaisesRegex(REGISTRY.RegistryError, message):
            self.verify_wrapper(
                "size_run_sweep", "hardware_crossover", arguments, sweep
            )
        # The creation/derivation path must reject the same apparent evidence,
        # rather than emitting a PASS wrapper which only replay would reject.
        with self.assertRaisesRegex(REGISTRY.RegistryError, message):
            REGISTRY._verify_typed_evidence(
                self.root / "size_run_sweep.json", self.root,
                expected_kind="size_run_sweep", expected_workload_id=self.workload,
                expected_campaign_id="fixture-campaign",
                expected_inventory_sha256="b" * 64,
                expected_classification="hardware_crossover", derive_result=True,
            )

    def test_supported_non_numerical_evidence_kinds_have_real_positive_producers(self) -> None:
        common = [
            self.sample(topology, cycle)
            for cycle in (1, 2)
            for topology in ("cpu1", "cuda1")
        ]
        self.verify_wrapper(
            "timing",
            "unknown",
            {"comparison": "cpu_cuda_equal_work", "minimum_cycles": 2},
            common,
        )
        self.verify_wrapper(
            "phase_profile",
            "unknown",
            {"phase_inventory": sorted(REGISTRY.PHASES), "minimum_cycles": 2},
            common,
        )

        ablation_arguments = {
            "intervention_flag": "--gpmeep-ablation",
            "intervention": "curl-phase-replay",
            "minimum_cycles": 2,
            "require_reversed_order": True,
        }
        ablation = []
        for cycle, states in ((1, ("off", "on")), (2, ("on", "off"))):
            for state in states:
                ablation.append(
                    self.sample(
                        "cuda1",
                        cycle,
                        ["--gpmeep-ablation", f"curl-phase-replay={state}"],
                    )
                )
        for kind in ("profiler_or_ablation", "same_binary_ablation"):
            self.verify_wrapper(kind, "engine_overhead", ablation_arguments, ablation)

        environments = [
            self.sample("cuda1", cycle, environment_label=label)
            for cycle in (1, 2)
            for label in ("baseline", "cuda-launch-blocking")
        ]
        self.verify_wrapper(
            "environment_diagnostic",
            "environment",
            {
                "diagnostic_keys": ["environment_sha256", "runtime_id"],
                "minimum_cycles": 2,
            },
            environments,
        )

    def test_producer_rejects_unrelated_command_drift_before_execution(self) -> None:
        with self.assertRaisesRegex(
            EXPERIMENT.ExperimentError, "canonical scaling or ablation"
        ):
            self.sample(
                "cuda1",
                1,
                [
                    "--gpmeep-ablation",
                    "curl-phase-replay=on",
                    "--unrelated-work-change",
                ],
            )


class V110IsolatedM4ReplayTest(unittest.TestCase):
    def test_replay_runs_in_a_fresh_isolated_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            controller = root / "controller.py"
            output = root / "m4"
            output.mkdir()
            (output / "M4_COMPLETE").write_text("{}\n")
            controller.write_text(
                "import pathlib\n"
                "import sys\n"
                "def verify_complete(output):\n"
                "    raise RuntimeError('the expensive release verifier must not run')\n"
                "def verify_experiment_authority(output):\n"
                "    assert (pathlib.Path(output) / 'M4_COMPLETE').is_file()\n"
                "    return {'schema': 'fixture', 'outcome': 'PASS', "
                "'contract': {'isolated': sys.flags.isolated, "
                "'controller': str(pathlib.Path(__file__).resolve())}}\n"
            )
            runtime_python = pathlib.Path(sys.executable).resolve()
            result = ISOLATED_M4_REPLAY(
                controller, output, runtime_python, runtime_python.parents[1]
            )
            self.assertFalse((root / "__pycache__").exists())
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result["contract"]["isolated"], 1)
        self.assertEqual(result["contract"]["controller"], str(controller))

    def test_replay_uses_the_bounded_authority_path_within_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            controller = root / "controller.py"
            output = root / "m4"
            output.mkdir()
            (output / "M4_COMPLETE").write_text("{}\n")
            controller.write_text(
                "import pathlib\n"
                "import time\n"
                "def verify_complete(output):\n"
                "    time.sleep(700)\n"
                "    raise RuntimeError('unreachable full verifier')\n"
                "def verify_experiment_authority(output):\n"
                "    assert (pathlib.Path(output) / 'M4_COMPLETE').is_file()\n"
                "    return {'schema': 'fixture', 'outcome': 'PASS', "
                "'contract': {'path': 'bounded'}}\n"
            )
            runtime_python = pathlib.Path(sys.executable).resolve()
            result = ISOLATED_M4_REPLAY(
                controller, output, runtime_python, runtime_python.parents[1]
            )
        self.assertEqual(result["contract"]["path"], "bounded")

    def test_exact_module_loader_restores_the_prior_module_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary).resolve() / "fixture.py"
            source.write_text(
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class Value:\n"
                "    number: int\n"
            )
            name = "gpmeep_exact_loader_restoration_fixture"
            previous = types.ModuleType(name)
            sys.modules[name] = previous
            try:
                loaded = EXPERIMENT._load_module(name, source)
                self.assertEqual(loaded.Value(7).number, 7)
                self.assertIs(sys.modules[name], previous)
            finally:
                sys.modules.pop(name, None)
            loaded = EXPERIMENT._load_module(name, source)
            self.assertEqual(loaded.Value(9).number, 9)
            self.assertNotIn(name, sys.modules)


class V110QualifiedChildInvocationTest(unittest.TestCase):
    def test_native_gpu_identifier_uses_the_physical_uuid_spelling(self):
        expected = "GPU-00000000-0000-0000-0000-000000000001"
        self.assertEqual(expected, CHILD._physical_gpu_uuid("00000000000000000000000000000001"))
        self.assertEqual(expected, CHILD._physical_gpu_uuid(expected))
        self.assertNotEqual(expected, CHILD._physical_gpu_uuid("00000000000000000000000000000002"))
        for invalid in ("cuda:0", "", "GPU-unknown", "{00000000-0000-0000-0000-000000000001}"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(CHILD.ChildError):
                    CHILD._physical_gpu_uuid(invalid)

    def test_child_ignores_timestamp_valid_workload_support_bytecode(self):
        with tempfile.TemporaryDirectory() as raw:
            tests = pathlib.Path(raw) / "python/tests"
            tests.mkdir(parents=True)
            helper = tests / "utils.py"
            committed = b"def allow(): return False\n"
            cached = b"def allow(): return True \n"
            self.assertEqual(len(committed), len(cached))
            helper.write_bytes(cached)
            status = helper.stat()
            with mock.patch.object(sys, "pycache_prefix", None):
                py_compile.compile(str(helper), doraise=True)
            helper.write_bytes(committed)
            os.utime(helper, ns=(status.st_atime_ns, status.st_mtime_ns))
            (tests / "adjoint_cyl_fixed_work_contract.py").write_text("# helper\n")
            workload = tests / "test_adjoint_cyl.py"
            workload.write_text(
                "import unittest, utils\nclass Case(unittest.TestCase):\n"
                + "".join(f" def test_{n}(self): self.assertTrue(utils.allow())\n"
                          for n in range(4))
            )
            contract = CHILD.workload_contract(workload, "python/tests/test_adjoint_cyl.py")
            with mock.patch.dict(sys.modules):
                sys.modules.pop("utils", None)
                sys.modules.pop("test_adjoint_cyl", None)
                with mock.patch.object(CHILD, "_main", side_effect=lambda _argv: CHILD._run_workload(workload, contract)):
                    with mock.patch.object(sys, "stderr", io.StringIO()):
                        with self.assertRaisesRegex(CHILD.ChildError, "did not pass all four"):
                            CHILD.main([])
            self.assertEqual(contract, CHILD.workload_contract(workload, "python/tests/test_adjoint_cyl.py"))

    def test_workload_arguments_cover_the_real_m4_manifest(self):
        plan = json.loads((ROOT / "packaging/v110-m4-equal-work-plan.json").read_text())
        manifest = json.loads((ROOT / "scripts/python-validation/manifest.json").read_text())
        paths = {case["path"] for case in plan["cases"]}
        self.assertEqual(paths, set(CHILD.WORKLOAD_ARGUMENTS))
        rules = {row["glob"]: row for row in manifest["overrides"] if row.get("glob") in paths}
        for path in paths:
            command = rules[path].get("command", [])
            index = next((i for i, token in enumerate(command)
                          if token in ("{path}", "{repo}/" + path)), None)
            arguments = command[index + 1:] if index is not None else []
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            with self.subTest(path=path):
                self.assertEqual(arguments, list(CHILD.WORKLOAD_ARGUMENTS[path]))
                contract = CHILD.workload_contract(ROOT / path, path)
                operation = command[command.index("--observe-operation") + 1] if "--observe-operation" in command else "run"
                self.assertEqual(operation, contract["observed_method"])
                if path.endswith("absorber-1d.py"):
                    self.assertEqual("absorber_branch_matrix.py",
                                     pathlib.Path(contract["support_files"][0]["path"]).name)

    def test_workload_preserves_argv_identity_and_restores_import_path(self):
        original = sys.argv
        contents = list(original)
        path = list(sys.path)
        workload = ROOT / "python/examples/edge_emitter_3D.py"
        contract = CHILD.workload_contract(workload, "python/examples/edge_emitter_3D.py")
        def execute(target, *, run_name):
            self.assertIs(sys.argv, original)
            self.assertEqual([str(workload), "--validation"], sys.argv)
            self.assertEqual(str(workload.parent), sys.path[0])
            sys.argv = ["example replaced argv"]
            raise RuntimeError("example failed")
        with mock.patch.object(CHILD.runpy, "run_path", side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, "example failed"):
                CHILD._run_workload(workload, contract)
        self.assertIs(sys.argv, original)
        self.assertEqual(contents, sys.argv)
        self.assertEqual(path, sys.path)

    def test_absorber_invokes_the_real_two_branch_adapter(self):
        workload = ROOT / "python/examples/absorber-1d.py"
        contract = CHILD.workload_contract(workload, "python/examples/absorber-1d.py")
        calls = []
        def execute(target, *, run_name):
            calls.append((target, list(sys.argv)))
        with mock.patch.object(CHILD.runpy, "run_path", side_effect=execute):
            CHILD._run_workload(workload, contract)
        adapter = str(ROOT / "scripts/python-validation/absorber_branch_matrix.py")
        self.assertEqual([(adapter, [adapter, str(workload), "--", "--validation"])], calls)

    def test_k_point_observer_scales_duration_with_pristine_methods(self):
        calls = []
        class Simulation:
            def __init__(self, *, resolution):
                self.resolution = resolution
            def run(self, **kwargs):
                raise AssertionError("run_k_point must not need Simulation.run")
            def run_k_point(self, t=None, k=None):
                self_test.assertIs(Simulation.__init__, original_init)
                self_test.assertIs(Simulation.run_k_point, original_run)
                calls.append((t, k))
        self_test = self
        original_init, original_run = Simulation.__init__, Simulation.run_k_point
        mp = types.SimpleNamespace(Simulation=Simulation)
        seconds, policy = [], []
        restore = CHILD._install_scaling(mp, 2, 3, seconds, RuntimeError, [], policy,
                                        observed_method="run_k_point")
        with mock.patch.object(CHILD, "_backend_policy_run_record", return_value={"observed": True}):
            Simulation(resolution=4).run_k_point(5, "k")
        restore()
        self.assertEqual([(15.0, "k")], calls)
        self.assertEqual(1, len(seconds))
        self.assertEqual([{"observed": True}], policy)
        self.assertIs(Simulation.__init__, original_init)
        self.assertIs(Simulation.run_k_point, original_run)

    def test_child_environment_replays_without_home_and_selects_fixed_work(self):
        account_home = str(pathlib.Path(EXPERIMENT.pwd.getpwuid(os.getuid()).pw_dir).resolve())
        with mock.patch.dict(os.environ, {}, clear=True):
            environment = EXPERIMENT._environment("cpu1", [], "baseline", pathlib.Path("/runtime"))
        with mock.patch.dict(os.environ, {"HOME": "/unrelated/environment/home"}, clear=True):
            replay = EXPERIMENT._environment("cpu1", [], "baseline", pathlib.Path("/runtime"))
        self.assertEqual(account_home, environment["HOME"])
        self.assertEqual(environment, replay)
        self.assertEqual("1", environment["GPMEEP_PHASE_IN_MATERIAL_FIXED_WORK"])
        self.assertEqual("1", environment["GPMEEP_ADJOINT_CYL_FIXED_WORK"])


if __name__ == "__main__":
    unittest.main()
