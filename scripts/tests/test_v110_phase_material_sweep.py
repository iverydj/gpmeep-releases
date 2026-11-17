"""Contract rejection tests with synthetic arrays; not physical qualification."""
import copy
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / 'scripts/python-validation/phase_material_contract.py'
spec = importlib.util.spec_from_file_location('phase_contract_under_test', HELPER_PATH)
H = importlib.util.module_from_spec(spec)
exec(compile(HELPER_PATH.read_bytes(), str(HELPER_PATH), 'exec'), H.__dict__)


def fixture(contract):
    grid = tuple(contract['grid_shape'])
    initial, target = np.full(grid, 3.0), np.full(grid, 2.0)
    times = np.asarray(contract['snapshot_times'], dtype=np.float64)
    snapshots = np.asarray([np.reciprocal((1 - time / contract['phase_time']) / initial
                            + (time / contract['phase_time']) / target) for time in times])
    ez = np.ones((5, *grid), dtype=np.float64)
    arrays = {
        'phase_initial_epsilon': initial, 'phase_target_epsilon': target,
        'phase_snapshot_times': times, 'phase_snapshot_epsilon': snapshots,
        'phase_final_epsilon': target.copy(),
        'phase_reciprocal_relative_error': np.zeros(1),
        'phase_endpoint_relative_error': np.zeros(1),
        'phase_progress': np.asarray([np.linalg.norm(value - initial) for value in snapshots]),
        'phase_snapshot_ez': ez, 'phase_snapshot_dz': ez.copy(),
        'phase_changed_region_ez_l2': np.asarray([np.linalg.norm(value) for value in ez]),
        'phase_static_initial_ez': np.full(grid, 0.5),
        'phase_static_initial_dz': np.full(grid, 0.5),
        'phase_dynamic_static_relative_difference': np.asarray([0.5]),
    }
    start = {'resolution': float(contract['resolution']), 'cell_size': contract['cell_size'],
             'courant': 0.5, 'initialized': True, 'grid_shape': list(grid),
             'dt': contract['dt'], 'timestep': 0, 'meep_time': 0.0}
    end = {**start, 'timestep': contract['steps_per_run'], 'meep_time': contract['phase_time']}
    static_start = {**start, 'initialized': False, 'grid_shape': None, 'dt': None}
    observations = {'dynamic': {'before': start, 'after': end}, 'target': copy.deepcopy(start),
                    'static': {'before': static_start, 'after': copy.deepcopy(end)}}
    namespace = {**arrays, 'phase_sweep_contract': contract, 'phase_run_observations': observations}
    metrics = [{'timestep': contract['steps_per_run'], 'timestep_delta': contract['steps_per_run'],
                'meep_time': contract['phase_time'], 'field_energy': 1.0, 'dft_norm': 0.0}
               for _ in range(2)]
    return namespace, metrics


def update_ez_metrics(namespace):
    snapshots = namespace['phase_snapshot_ez']
    namespace['phase_changed_region_ez_l2'] = np.asarray([np.linalg.norm(value) for value in snapshots])
    namespace['phase_dynamic_static_relative_difference'] = np.asarray([
        np.linalg.norm(snapshots[-1] - namespace['phase_static_initial_ez']) / np.linalg.norm(snapshots[-1])
    ])


class PhaseMaterialSweepContractTests(unittest.TestCase):
    def test_registry_family_dispatch_compares_all_pairs_and_rejects_field_drift(self):
        registry_path = ROOT / 'scripts/validate-v110-inversion-registry.py'
        registry_spec = importlib.util.spec_from_file_location('phase_sweep_registry_test', registry_path)
        registry = importlib.util.module_from_spec(registry_spec)
        exec(compile(registry_path.read_bytes(), str(registry_path), 'exec'), registry.__dict__)
        support = {'path': str(HELPER_PATH), 'size_bytes': HELPER_PATH.stat().st_size,
                   'sha256': hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()}
        inputs, reports, identities = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size in (1, 2, 3):
                for length in (1, 2):
                    contract = H.work_contract(size, length)
                    for cycle in (1, 2):
                        for topology in ('cpu1', 'cuda1'):
                            sample = root / f's{size}-l{length}-c{cycle}-{topology}'
                            sample.mkdir()
                            namespace, metrics = fixture(contract)
                            numerical = H.write_numerical(
                                namespace, sample / 'phase-numerical.npy', contract, metrics
                            )
                            inputs.append({'path': f'{sample.name}/report.json'})
                            reports.append({
                                'runtime_authority': {'repo': str(ROOT),
                                    'workload_contract': {'support_files': [support]}},
                                'lane': {'command': ['fixture', '--gpmeep-sweep-size', str(size),
                                                    '--gpmeep-sweep-run-length', str(length)]},
                                'runtime_snapshot': {'phase_numerical': numerical},
                            })
                            identities.append({'topology': topology, 'cycle_index': cycle})
            result = registry._verify_sweep_family(
                'python/examples/phase_in_material.py', inputs, reports, identities, root
            )
            self.assertEqual(len(result['phase_numerical_pairs']), 12)
            self.assertTrue(all(row['outcome'] == 'PASS' for row in result['phase_numerical_pairs']))

            # A self-consistent, rehashed CUDA artifact with a physical field
            # difference must still fail the CPU/CUDA family oracle.
            contract = H.work_contract()
            namespace, metrics = fixture(contract)
            namespace['phase_snapshot_ez'] += 0.01
            update_ez_metrics(namespace)
            artifact = root / 's1-l1-c1-cuda1/phase-numerical.npy'
            artifact.unlink()
            reports[1]['runtime_snapshot']['phase_numerical'] = H.write_numerical(
                namespace, artifact, contract, metrics
            )
            with self.assertRaisesRegex(registry.RegistryError, 'phase sweep numerical comparison differs'):
                registry._verify_sweep_family(
                    'python/examples/phase_in_material.py', inputs, reports, identities, root
                )

    def test_original_modes_retain_grid_duration_and_steps(self):
        for fixed, duration, steps in ((False, 2.0, 40), (True, 200.0, 4000)):
            contract = H.work_contract(fixed_work=fixed)
            self.assertEqual(contract['grid_shape'], [60, 60])
            self.assertEqual(contract['phase_time'], duration)
            self.assertEqual(contract['steps_per_run'], steps)
            self.assertEqual(contract['run_count'], 2)
            self.assertEqual(contract['snapshot_times'], [duration * index / 4 for index in range(5)])

    def test_rejects_native_overflow_and_float32_stop_misalignment(self):
        with self.assertRaisesRegex(ValueError, 'native signed timestep'):
            H.work_contract(1, 1_000_000)
        with self.assertRaisesRegex(ValueError, 'exact intended native timestep'):
            H.work_contract(1, .007)
        with self.assertRaisesRegex(ValueError, 'original time tolerance'):
            H.work_contract(1, 2.003)
        with self.assertRaisesRegex(ValueError, 'exact positive integer'):
            H.work_contract(1.01, 1)
        with self.assertRaisesRegex(ValueError, 'fixed-work validation'):
            H.work_contract(2, 1, fixed_work=False)

    def test_actual_grid_and_step_observations_cannot_be_relabelled(self):
        contract = H.work_contract()
        namespace, _ = fixture(contract)
        observations = namespace['phase_run_observations']
        for key, value in (('grid_shape', [120, 120]), ('dt', .025), ('timestep', 3999)):
            changed = copy.deepcopy(observations)
            changed['dynamic']['after'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                H.validate_observations(changed, contract)

    def test_missing_snapshot_wrong_time_and_static_control_are_rejected(self):
        contract = H.work_contract()
        namespace, _ = fixture(contract)
        arrays = {key: namespace[key] for key in H.TOLERANCES}
        for name, replacement in (
            ('phase_snapshot_ez', arrays['phase_snapshot_ez'][:-1]),
            ('phase_snapshot_times', np.asarray([0., 50., 100., 151., 200.])),
            ('phase_static_initial_ez', arrays['phase_snapshot_ez'][-1].copy()),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                H.validate_arrays({**arrays, name: replacement}, contract)

    def test_full_field_bundle_roundtrip_and_same_coordinate_comparison(self):
        contract = H.work_contract()
        namespace, metrics = fixture(contract)
        with tempfile.TemporaryDirectory() as temporary:
            cpu, cuda = Path(temporary) / 'cpu.npy', Path(temporary) / 'cuda.npy'
            cpu_record = H.write_numerical(namespace, cpu, contract, metrics)
            cuda_record = H.write_numerical(namespace, cuda, contract, metrics)
            result = H.compare_numerical(cpu_record, cpu, cuda_record, cuda)
            self.assertEqual(result['outcome'], 'PASS')
            self.assertEqual(set(result['arrays']), set(H.TOLERANCES))
            self.assertEqual(set(result['native_run_metrics']), set(metrics[0]))
            arrays, _ = H.read_arrays(cpu, contract)
            for name in H.TOLERANCES:
                np.testing.assert_array_equal(arrays[name], namespace[name])

    def test_pointwise_pass_does_not_replace_original_weighted_summary(self):
        contract = H.work_contract()
        cpu_namespace, metrics = fixture(contract)
        ez = np.zeros((5, 60, 60), dtype=np.float64)
        ez.flat[8191], ez.flat[16383] = -2.0, 1.0
        cpu_namespace['phase_snapshot_ez'] = ez
        static = np.zeros((60, 60), dtype=np.float64)
        static.flat[16383 - 4 * 3600] = -1.0
        cpu_namespace['phase_static_initial_ez'] = static
        update_ez_metrics(cpu_namespace)
        cuda_namespace = copy.deepcopy(cpu_namespace)
        cuda_namespace['phase_snapshot_ez'].flat[16383] += 1e-6
        update_ez_metrics(cuda_namespace)
        rtol, atol = H.TOLERANCES['phase_snapshot_ez']
        difference = np.abs(cpu_namespace['phase_snapshot_ez'] - cuda_namespace['phase_snapshot_ez'])
        self.assertTrue(np.all(difference <= atol + rtol * np.maximum(
            np.abs(cpu_namespace['phase_snapshot_ez']), np.abs(cuda_namespace['phase_snapshot_ez']))))
        with tempfile.TemporaryDirectory() as temporary:
            cpu, cuda = Path(temporary) / 'cpu.npy', Path(temporary) / 'cuda.npy'
            cpu_record = H.write_numerical(cpu_namespace, cpu, contract, metrics)
            cuda_record = H.write_numerical(cuda_namespace, cuda, contract, metrics)
            with self.assertRaisesRegex(ValueError, 'original result summary comparison failed: phase_snapshot_ez'):
                H.compare_numerical(cpu_record, cpu, cuda_record, cuda)

    def test_nonunit_coordinates_keep_complete_evidence_and_actual_work(self):
        for size, length, grid, duration, steps in (
            (2, 1, [120, 120], 200.0, 8000),
            (1, 2, [60, 60], 400.0, 8000),
            (2, 2, [120, 120], 400.0, 16000),
        ):
            with self.subTest(size=size, run_length=length), tempfile.TemporaryDirectory() as temporary:
                contract = H.work_contract(size, length)
                self.assertEqual(contract['grid_shape'], grid)
                self.assertEqual(contract['phase_time'], duration)
                self.assertEqual(contract['steps_per_run'], steps)
                namespace, metrics = fixture(contract)
                cpu, cuda = Path(temporary) / 'cpu.npy', Path(temporary) / 'cuda.npy'
                cpu_record = H.write_numerical(namespace, cpu, contract, metrics)
                cuda_record = H.write_numerical(namespace, cuda, contract, metrics)
                self.assertEqual(H.compare_numerical(cpu_record, cpu, cuda_record, cuda)['outcome'], 'PASS')

    def test_valid_samples_at_different_coordinates_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = []
            for size in (1, 2):
                contract = H.work_contract(size, 1)
                namespace, metrics = fixture(contract)
                path = Path(temporary) / f'size-{size}.npy'
                records.append((H.write_numerical(namespace, path, contract, metrics), path))
            with self.assertRaisesRegex(ValueError, 'coordinates differ'):
                H.compare_numerical(*records[0], *records[1])

    def test_rehashed_record_with_wrong_artifact_sha_is_rejected(self):
        contract = H.work_contract()
        namespace, metrics = fixture(contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fields.npy'
            record = H.write_numerical(namespace, path, contract, metrics)
            record['artifact']['sha256'] = '0' * 64
            record['record_id'] = H.digest({key: value for key, value in record.items() if key != 'record_id'})
            with self.assertRaisesRegex(ValueError, 'artifact binding'):
                H.verify_numerical(record, path, contract)

    def test_finite_numerically_valid_payload_change_cannot_reuse_the_record(self):
        contract = H.work_contract()
        namespace, metrics = fixture(contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fields.npy'
            record = H.write_numerical(namespace, path, contract, metrics)
            contents = bytearray(path.read_bytes())
            # Change only the least significant bit of the last float64 metric.
            # Its value still passes the numerical consistency tolerance.
            contents[-8] ^= 1
            path.write_bytes(contents)
            H.read_arrays(path, contract)
            with self.assertRaisesRegex(ValueError, 'artifact binding'):
                H.verify_numerical(record, path, contract)

    def test_native_energy_is_compared_even_when_all_stored_fields_match(self):
        contract = H.work_contract()
        namespace, metrics = fixture(contract)
        changed = copy.deepcopy(metrics)
        changed[0]['field_energy'] = 1.01
        with tempfile.TemporaryDirectory() as temporary:
            cpu, cuda = Path(temporary) / 'cpu.npy', Path(temporary) / 'cuda.npy'
            cpu_record = H.write_numerical(namespace, cpu, contract, metrics)
            cuda_record = H.write_numerical(namespace, cuda, contract, changed)
            with self.assertRaisesRegex(ValueError, 'per-run metric differs: field_energy'):
                H.compare_numerical(cpu_record, cpu, cuda_record, cuda)

    def test_finite_field_values_cannot_pass_with_overflowed_summary(self):
        contract = H.work_contract()
        namespace, _ = fixture(contract)
        namespace['phase_snapshot_dz'].flat[0] = 1.341e154
        with self.assertRaisesRegex(ValueError, 'finite signal'):
            H.validate_arrays({key: namespace[key] for key in H.TOLERANCES}, contract)
        self.assertFalse(H.close_scalar(float('inf'), 1.0, .001, 1e-8))
        self.assertFalse(H.close_scalar(1e308, -1e308, .001, 1e-8))

    def test_oversized_npy_header_rejected_before_numpy_header_parser(self):
        contract = H.work_contract()
        count = sum(np.prod(shape) for shape in H.array_shapes(contract).values())
        contents = b'\x93NUMPY\x02\x00' + (2**32 - 1).to_bytes(4, 'little') + bytes(int(count) * 8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.npy'
            path.write_bytes(contents)
            with mock.patch.object(np.lib.format, 'read_array_header_2_0', side_effect=AssertionError('parser reached')):
                with self.assertRaisesRegex(ValueError, 'header length'):
                    H.read_arrays(path, contract)

    def test_forged_npy_shape_rejected_before_array_view(self):
        contract = H.work_contract()
        count = sum(np.prod(shape) for shape in H.array_shapes(contract).values())
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(buffer, {
            'descr': '<f8', 'fortran_order': False, 'shape': (2**62,),
        })
        buffer.write(bytes(int(count) * 8))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.npy'
            path.write_bytes(buffer.getvalue())
            with mock.patch.object(np, 'frombuffer', side_effect=AssertionError('array view reached')):
                with self.assertRaisesRegex(ValueError, 'NPY shape'):
                    H.read_arrays(path, contract)

    def test_fifo_is_rejected_without_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fifo'
            os.mkfifo(path)
            program = (
                'import importlib.util,pathlib\n'
                f'p=pathlib.Path({str(HELPER_PATH)!r})\n'
                's=importlib.util.spec_from_file_location("fifo_check",p)\n'
                'm=importlib.util.module_from_spec(s)\n'
                'exec(compile(p.read_bytes(),str(p),"exec"),m.__dict__)\n'
                'try:\n'
                f' m.read_arrays(pathlib.Path({str(path)!r}),m.work_contract())\n'
                'except ValueError:\n'
                ' raise SystemExit(0)\n'
                'raise SystemExit(1)\n'
            )
            result = subprocess.run([sys.executable, '-I', '-B', '-c', program], timeout=5,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
