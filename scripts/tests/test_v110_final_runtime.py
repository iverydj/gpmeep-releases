"""Replay real final -> candidate -> registry/host dependencies, without mocks."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("runtime_authority", ROOT / "scripts/v110-release-only-receipt.py")
AUTHORITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTHORITY)


class FinalRuntimeTest(unittest.TestCase):
    def test_real_pinned_child_loads_registry_and_host_dependencies(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixed = root / "fixed.json"
            fixed.write_text("{}")
            records = root / "records"
            records.mkdir()
            (records / "invalid.json").write_text("{}")
            for route in ("registry", "host"):
                verifier = root / f"{route}.py"
                verifier.write_text(f'''import importlib.util, pathlib, sys
root = pathlib.Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location("real_final", root / "scripts/v110-final-release-receipt.py")
final = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = final
spec.loader.exec_module(final)
candidate = final._load_candidate_tool(root / "scripts/v110-release-gate-receipt.py")
if {route!r} == "registry":
    candidate._replay_registry_gate(pathlib.Path({str(fixed)!r}), pathlib.Path({raw!r}), pathlib.Path({str(records)!r}))
else:
    candidate._replay_native_evidence("host_four", pathlib.Path({str(root / 'absent.json')!r}), {{}}, "0" * 40,
        source_policy_path=None, expected_source_policy_sha256=None,
        source_verifier_path=None, expected_source_verifier_sha256=None)
''')
                with self.subTest(route=route), self.assertRaises(AUTHORITY.ReleaseOnlyError) as caught:
                    AUTHORITY._run_pinned_final_verifier(verifier_path=verifier,
                        expected_verifier_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest(),
                        verifier_arguments=[], final_receipt_path=fixed,
                        package_path=fixed, candidate_receipt_path=fixed)
                message = str(caught.exception)
                self.assertNotIn("ModuleNotFoundError", message)
                self.assertIn("schema" if route == "registry" else "absent.json", message)


if __name__ == "__main__":
    unittest.main()
