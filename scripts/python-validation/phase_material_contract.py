"""Exact work and full-field evidence for the material-phasing case family.

This module is an isolated implementation draft, not yet applied or qualified.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct


CONTRACT_SCHEMA = 'gpmeep-v110-phase-material-work-v1'
NUMERICAL_SCHEMA = 'gpmeep-v110-phase-material-numerical-v1'
MAX_ARRAY_BYTES = 512 * 1024**2
MAX_NPY_HEADER_BYTES = 4096
MAX_NATIVE_TIMESTEP = 2**31 - 1
# Exact tolerances of the existing phase entry in the validation manifest.
TOLERANCES = {
    'phase_initial_epsilon': (2e-5, 1e-6),
    'phase_target_epsilon': (2e-5, 1e-6),
    'phase_snapshot_times': (0.0, 1e-6),
    'phase_snapshot_epsilon': (5e-5, 2e-6),
    'phase_final_epsilon': (5e-5, 2e-6),
    'phase_reciprocal_relative_error': (0.1, 1e-7),
    'phase_endpoint_relative_error': (0.1, 1e-7),
    'phase_progress': (5e-5, 2e-6),
    'phase_snapshot_ez': (0.001, 3e-6),
    'phase_snapshot_dz': (0.001, 3e-6),
    'phase_changed_region_ez_l2': (0.001, 2e-6),
    'phase_static_initial_ez': (0.001, 2e-6),
    'phase_static_initial_dz': (0.001, 2e-6),
    'phase_dynamic_static_relative_difference': (0.002, 2e-6),
}
MIN_L2 = {
    'phase_initial_epsilon': 60.0, 'phase_target_epsilon': 60.0,
    'phase_snapshot_times': 1.0, 'phase_snapshot_epsilon': 100.0,
    'phase_final_epsilon': 60.0, 'phase_progress': 1.0,
    'phase_snapshot_ez': 1.0, 'phase_snapshot_dz': 1.0,
    'phase_changed_region_ez_l2': 1.0,
    'phase_static_initial_ez': 1.0, 'phase_static_initial_dz': 1.0,
    'phase_dynamic_static_relative_difference': 0.05,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def work_contract(size=1.0, run_length=1.0, *, fixed_work=True):
    require(type(fixed_work) is bool, 'phase fixed-work mode is not boolean')
    for name, value in (('size', size), ('run_length', run_length)):
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                f'phase {name} must be finite and positive')
    size, run_length = float(size), float(run_length)
    require(fixed_work or (size == run_length == 1.0),
            'nonunit phase coordinates require the fixed-work validation mode')
    resolution = 10.0 * size
    require(math.isfinite(resolution) and resolution.is_integer(),
            'phase resolution must be an exact positive integer')
    resolution = int(resolution)
    duration = (200.0 if fixed_work else 2.0) * run_length
    dt = 0.5 / resolution
    steps = duration / dt
    require(math.isfinite(steps) and steps > 0 and steps.is_integer() and int(steps) % 4 == 0,
            'phase duration and quarter snapshots must align with native timesteps')
    require(steps <= MAX_NATIVE_TIMESTEP, 'phase duration exceeds the native signed timestep range')
    rounded_before = struct.unpack('=f', struct.pack('=f', (int(steps) - 1) * dt))[0]
    rounded_end = struct.unpack('=f', struct.pack('=f', int(steps) * dt))[0]
    require(rounded_before < duration <= rounded_end,
            'phase duration does not stop at the exact intended native timestep')
    require(abs(rounded_end - duration) <= 1e-6,
            'phase rounded endpoint exceeds the original time tolerance')
    require(all(abs((int(steps) // 4 * q) * dt - duration * q / 4) <= 1e-6
                for q in range(5)), 'phase quarter times differ from native timestep times')
    previous_quarter_time = 0.0
    for quarter in range(1, 5):
        quarter_step = int(steps) // 4 * quarter
        before = struct.unpack('=f', struct.pack('=f', (quarter_step - 1) * dt))[0]
        at = struct.unpack('=f', struct.pack('=f', quarter_step * dt))[0]
        threshold = previous_quarter_time + duration / 4 + (-0.5 * dt)
        require(before < threshold <= at,
                'phase quarter callback does not first fire at the intended native step')
        previous_quarter_time = at
    n = 6 * resolution
    # Five full epsilon/Ez/Dz snapshots, five standalone fields, and scalar vectors.
    require((20 * n * n + 24) * 8 <= MAX_ARRAY_BYTES,
            'phase full numerical evidence exceeds the resource bound')
    result = {
        'schema': CONTRACT_SCHEMA,
        'coordinates': {'size': size, 'run_length': run_length},
        'fixed_work': fixed_work, 'resolution': resolution,
        'cell_size': [6.0, 6.0, 0.0], 'grid_shape': [n, n],
        'courant': 0.5, 'dt': dt, 'phase_time': duration,
        'steps_per_run': int(steps), 'run_count': 2,
        'snapshot_times': [duration * quarter / 4 for quarter in range(5)],
        'geometry': {'radius': 1.0, 'index': 3.5,
                     'initial_center': [0.0, 0.0, 0.0],
                     'target_center': [1.0, 1.0, 0.0]},
        'source': {'component': 'Ez', 'frequency': 0.5,
                   'center': [-0.5, 0.0, 0.0], 'end_time': duration},
        'static_control': 'initial-material-same-source-grid-duration',
    }
    result['contract_id'] = digest(result)
    return result


def validate_contract(value):
    require(isinstance(value, dict), 'phase work contract is absent')
    coordinates = value.get('coordinates', {})
    expected = work_contract(coordinates.get('size'), coordinates.get('run_length'),
                             fixed_work=value.get('fixed_work'))
    require(value == expected, 'phase work contract differs from its coordinates')
    return expected


def observe_simulation(simulation):
    fields = simulation.fields
    observation = {
        'resolution': float(simulation.resolution),
        'cell_size': [float(getattr(simulation.cell_size, axis)) for axis in 'xyz'],
        'courant': float(simulation.Courant), 'initialized': fields is not None,
        'grid_shape': None, 'dt': None, 'timestep': 0, 'meep_time': 0.0,
    }
    if fields is not None:
        observation.update(grid_shape=[int(simulation.gv.nx()), int(simulation.gv.ny())],
                           dt=float(fields.dt), timestep=int(fields.t),
                           meep_time=float(fields.time()))
    return observation


def validate_observations(value, contract):
    contract = validate_contract(contract)
    require(isinstance(value, dict) and set(value) == {'dynamic', 'target', 'static'},
            'phase simulation observation inventory differs')
    expected_keys = {'resolution', 'cell_size', 'courant', 'initialized',
                     'grid_shape', 'dt', 'timestep', 'meep_time'}
    for name in ('dynamic', 'static'):
        require(isinstance(value[name], dict) and set(value[name]) == {'before', 'after'},
                'phase run observation pair differs')
    points = [(name, moment, value[name][moment]) for name in ('dynamic', 'static')
              for moment in ('before', 'after')]
    points.append(('target', 'before', value['target']))
    for name, moment, observation in points:
        require(isinstance(observation, dict) and set(observation) == expected_keys,
                'phase simulation observation fields differ')
        require(type(observation['resolution']) in (int, float)
                and observation['resolution'] == contract['resolution']
                and observation['cell_size'] == contract['cell_size']
                and observation['courant'] == contract['courant'],
                'phase actual resolution, cell, or Courant differs')
        initialized = not (name == 'static' and moment == 'before')
        require(observation['initialized'] is initialized,
                'phase initialization placement changed')
        require(observation['grid_shape'] == (contract['grid_shape'] if initialized else None),
                'phase actual grid differs')
        if initialized:
            require(type(observation['dt']) in (int, float)
                    and math.isclose(observation['dt'], contract['dt'], rel_tol=1e-12, abs_tol=0),
                    'phase actual native dt differs')
        else:
            require(observation['dt'] is None, 'uninitialized phase dt is present')
        expected_step = contract['steps_per_run'] if moment == 'after' else 0
        expected_time = contract['phase_time'] if moment == 'after' else 0.0
        require(type(observation['timestep']) is int and observation['timestep'] == expected_step,
                'phase actual timestep count differs')
        require(type(observation['meep_time']) in (int, float)
                and math.isclose(observation['meep_time'], expected_time, rel_tol=0, abs_tol=1e-6),
                'phase actual simulation time differs')
    return value


def array_shapes(contract):
    grid = tuple(validate_contract(contract)['grid_shape'])
    result = {name: grid for name in TOLERANCES}
    for name in ('phase_snapshot_epsilon', 'phase_snapshot_ez', 'phase_snapshot_dz'):
        result[name] = (5, *grid)
    for name in ('phase_snapshot_times', 'phase_progress', 'phase_changed_region_ez_l2'):
        result[name] = (5,)
    for name in ('phase_reciprocal_relative_error', 'phase_endpoint_relative_error',
                 'phase_dynamic_static_relative_difference'):
        result[name] = (1,)
    return result


def validate_run_metrics(records, contract):
    require(isinstance(records, list) and len(records) == contract['run_count'],
            'phase numerical run count differs')
    for record in records:
        require(isinstance(record, dict) and set(record) == {
            'timestep', 'timestep_delta', 'meep_time', 'field_energy', 'dft_norm'},
            'phase native run metric inventory differs')
        for name in ('timestep', 'timestep_delta'):
            require(type(record[name]) is int and record[name] == contract['steps_per_run'],
                    'phase native run metric step count differs')
        for name in ('meep_time', 'field_energy', 'dft_norm'):
            require(type(record[name]) in (int, float) and math.isfinite(record[name]),
                    'phase native run metric is not finite')
        require(abs(record['meep_time'] - contract['phase_time']) <= 1e-6,
                'phase rounded native run time differs')
        require(abs(record['field_energy']) >= 1e-3, 'phase native field energy is below the original floor')
        require(record['dft_norm'] == 0.0, 'phase unexpectedly contains DFT state')
    return records


def scalar_summary(values):
    values = [float(value) for value in values]
    try:
        result = {'final': values[-1], 'sum': math.fsum(values),
                  'l2': math.sqrt(math.fsum(value * value for value in values)),
                  'weighted_checksum': math.fsum((index + 1) * value for index, value in enumerate(values))}
    except OverflowError as error:
        raise ValueError('phase native metric summary overflowed') from error
    require(all(math.isfinite(value) for value in result.values()), 'phase native metric summary is not finite')
    return result


def array_summary(array):
    # Same order and arithmetic as run_example_oracle.numeric_result_vector.
    try:
        result = {'l2': math.sqrt(math.fsum(float(value) * float(value) for value in array.flat)),
                  'max_abs': max(abs(float(value)) for value in array.flat),
                  'weighted_real': math.fsum((index + 1) * float(value) for index, value in enumerate(array.flat)),
                  'weighted_imag': 0.0}
    except OverflowError as error:
        raise ValueError('phase field summary overflowed') from error
    require(all(math.isfinite(value) for value in result.values()), 'phase field summary is not finite')
    return result


def close_scalar(left, right, rtol, atol):
    if not all(math.isfinite(value) for value in (left, right, rtol, atol)) or rtol < 0 or atol < 0:
        return False
    error = abs(left - right)
    relative = rtol * max(abs(left), abs(right))
    allowed = atol + relative
    return all(math.isfinite(value) for value in (error, relative, allowed)) and error <= allowed


def validate_arrays(arrays, contract):
    import numpy as np
    shapes = array_shapes(contract)
    require(isinstance(arrays, dict) and set(arrays) == set(shapes),
            'phase numerical array inventory differs')
    for name, shape in shapes.items():
        array = arrays[name]
        require(isinstance(array, np.ndarray) and array.dtype == np.dtype('float64')
                and array.shape == shape and np.isfinite(array).all(),
                f'phase array {name} type, shape, or finite values differ')
        if name in MIN_L2:
            original_l2 = math.sqrt(math.fsum(float(value) * float(value) for value in array.flat))
            require(math.isfinite(original_l2) and original_l2 >= MIN_L2[name],
                    f'phase array {name} lacks finite signal')
    require(sum(array.nbytes for array in arrays.values()) <= MAX_ARRAY_BYTES,
            'phase numerical array byte bound exceeded')
    a = arrays
    require(np.allclose(a['phase_snapshot_times'], contract['snapshot_times'], rtol=0, atol=1e-6),
            'phase quarter snapshot times differ')
    initial, target = a['phase_initial_epsilon'], a['phase_target_epsilon']
    require((initial > 0).all() and (target > 0).all(), 'phase permittivity is not positive')
    inverse_initial, inverse_target = np.reciprocal(initial), np.reciprocal(target)
    expected = np.asarray([np.reciprocal((1 - t / contract['phase_time']) * inverse_initial
                           + t / contract['phase_time'] * inverse_target)
                           for t in a['phase_snapshot_times']])
    reciprocal = float(np.linalg.norm(a['phase_snapshot_epsilon'] - expected)
                       / max(float(np.linalg.norm(expected)), 1e-30))
    endpoint = float(np.linalg.norm(a['phase_final_epsilon'] - target)
                     / max(float(np.linalg.norm(target)), 1e-30))
    progress = np.asarray([float(np.linalg.norm(s - initial)) for s in a['phase_snapshot_epsilon']])
    changed = np.abs(inverse_target - inverse_initial) > 1e-6
    excitation = np.asarray([float(np.linalg.norm(s[changed])) for s in a['phase_snapshot_ez']])
    separation = float(np.linalg.norm(a['phase_snapshot_ez'][-1] - a['phase_static_initial_ez'])
                       / max(float(np.linalg.norm(a['phase_snapshot_ez'][-1])), 1e-30))
    require(reciprocal <= 2e-5 and endpoint <= 2e-5,
            'phase reciprocal interpolation or material endpoint differs')
    require((np.diff(progress) >= -1e-7).all() and progress[-1] > 0,
            'phase material progression differs')
    require(excitation[-1] > 1e-3 and separation >= 0.05,
            'phase changing region or static negative control is ineffective')
    recomputed = {'phase_reciprocal_relative_error': [reciprocal],
                  'phase_endpoint_relative_error': [endpoint], 'phase_progress': progress,
                  'phase_changed_region_ez_l2': excitation,
                  'phase_dynamic_static_relative_difference': [separation]}
    for name, expected_value in recomputed.items():
        require(np.allclose(a[name], expected_value, rtol=1e-12, atol=1e-12),
                f'phase reported metric {name} disagrees with complete fields')
    return {name: {'shape': list(array.shape), 'dtype': 'float64', 'size_bytes': array.nbytes}
            for name, array in arrays.items()}


def read_arrays(path, contract):
    import numpy as np
    shapes = array_shapes(contract)
    total_values = sum(math.prod(shape) for shape in shapes.values())
    payload_bytes = total_values * 8
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode)
                and payload_bytes + 10 <= info.st_size <= payload_bytes + MAX_NPY_HEADER_BYTES + 12,
                'phase numerical artifact type or byte bound differs')
        contents = stream.read(info.st_size + 1)
        require(len(contents) == info.st_size, 'phase numerical artifact changed while read')
    require(contents[:6] == b'\x93NUMPY', 'phase numerical artifact is not NPY')
    version = tuple(contents[6:8])
    require(version in {(1, 0), (2, 0)}, 'phase NPY version differs')
    prefix = 10 if version == (1, 0) else 12
    header_length = int.from_bytes(contents[8:prefix], 'little')
    require(0 < header_length <= MAX_NPY_HEADER_BYTES
            and prefix + header_length + payload_bytes == len(contents),
            'phase NPY header length or complete payload size differs')
    header = io.BytesIO(contents[:prefix + header_length])
    require(np.lib.format.read_magic(header) == version, 'phase NPY version changed')
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    actual_shape, fortran, dtype = reader(header)
    require(actual_shape == (total_values,) and dtype == np.dtype('<f8') and not fortran
            and header.tell() == prefix + header_length,
            'phase NPY shape, dtype, or storage order differs')
    flat = np.frombuffer(contents, dtype='<f8', count=total_values, offset=prefix + header_length)
    arrays = {}
    offset = 0
    for name, shape in shapes.items():
        count = math.prod(shape)
        arrays[name] = flat[offset:offset + count].reshape(shape)
        offset += count
    validate_arrays(arrays, contract)
    return arrays, {'path': Path(path).name, 'size_bytes': len(contents),
                    'sha256': hashlib.sha256(contents).hexdigest()}


def write_numerical(namespace, path, contract, run_metrics):
    import numpy as np
    require(namespace.get('phase_sweep_contract') == contract,
            'phase workload and child work contracts differ')
    observations = validate_observations(namespace.get('phase_run_observations'), contract)
    run_metrics = validate_run_metrics(run_metrics, contract)
    arrays = {name: np.array(namespace[name], copy=True, order='C') for name in TOLERANCES}
    metadata = validate_arrays(arrays, contract)
    with Path(path).open('xb') as stream:
        flat = np.concatenate([arrays[name].reshape(-1) for name in TOLERANCES]).astype('<f8', copy=False)
        np.save(stream, flat, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    _, artifact = read_arrays(path, contract)
    record = {'schema': NUMERICAL_SCHEMA, 'contract': contract,
              'observations': observations, 'run_metrics': run_metrics,
              'arrays': metadata, 'artifact': artifact}
    record['record_id'] = digest(record)
    return record


def verify_numerical(record, path, expected_contract):
    require(isinstance(record, dict) and set(record) == {
        'schema', 'contract', 'observations', 'run_metrics', 'arrays', 'artifact', 'record_id'},
        'phase numerical record inventory differs')
    unsigned = {k: v for k, v in record.items() if k != 'record_id'}
    require(record['schema'] == NUMERICAL_SCHEMA and record['record_id'] == digest(unsigned)
            and record['contract'] == expected_contract,
            'phase numerical record identity or coordinates differ')
    validate_observations(record['observations'], expected_contract)
    validate_run_metrics(record['run_metrics'], expected_contract)
    arrays, artifact = read_arrays(path, expected_contract)
    require(record['artifact'] == artifact and record['arrays'] == validate_arrays(arrays, expected_contract),
            'phase numerical artifact binding differs')
    return arrays


def compare_numerical(cpu_record, cpu_path, cuda_record, cuda_path):
    import numpy as np
    contract = validate_contract(cpu_record.get('contract'))
    cpu = verify_numerical(cpu_record, cpu_path, contract)
    cuda = verify_numerical(cuda_record, cuda_path, contract)
    require(cpu_record['observations'] == cuda_record['observations'],
            'phase CPU/CUDA actual work observations differ')
    comparisons = {}
    for name, (rtol, atol) in TOLERANCES.items():
        differences = np.abs(cuda[name] - cpu[name])
        passed = bool(np.all(differences <= atol + rtol * np.maximum(np.abs(cpu[name]), np.abs(cuda[name]))))
        require(passed, f'phase CPU/CUDA complete field comparison failed: {name}')
        cpu_summary, cuda_summary = array_summary(cpu[name]), array_summary(cuda[name])
        require(all(close_scalar(cpu_summary[key], cuda_summary[key], rtol, atol) for key in cpu_summary),
                f'phase CPU/CUDA original result summary comparison failed: {name}')
        comparisons[name] = {'element_count': cpu[name].size,
                             'max_abs_error': float(np.max(differences)),
                             'cpu_summary': cpu_summary, 'cuda_summary': cuda_summary,
                             'rtol': rtol, 'atol': atol, 'outcome': 'PASS'}
    run_comparisons = {}
    tolerances = {'timestep': (0.0, 0.0), 'timestep_delta': (0.0, 0.0),
                  'meep_time': (0.0, 1e-6), 'field_energy': (0.001, 1e-8), 'dft_norm': (0.0, 0.0)}
    for name, (rtol, atol) in tolerances.items():
        cpu_values = [row[name] for row in cpu_record['run_metrics']]
        cuda_values = [row[name] for row in cuda_record['run_metrics']]
        require(all(close_scalar(a, b, rtol, atol) for a, b in zip(cpu_values, cuda_values, strict=True)),
                f'phase CPU/CUDA per-run metric differs: {name}')
        cpu_summary, cuda_summary = scalar_summary(cpu_values), scalar_summary(cuda_values)
        require(all(close_scalar(cpu_summary[key], cuda_summary[key], rtol, atol) for key in cpu_summary),
                f'phase CPU/CUDA aggregate native metric differs: {name}')
        run_comparisons[name] = {'cpu': cpu_values, 'cuda': cuda_values,
                                 'cpu_summary': cpu_summary, 'cuda_summary': cuda_summary,
                                 'rtol': rtol, 'atol': atol, 'outcome': 'PASS'}
    return {'contract_id': contract['contract_id'], 'outcome': 'PASS',
            'cpu_numerical_id': cpu_record['record_id'],
            'cuda_numerical_id': cuda_record['record_id'], 'arrays': comparisons,
            'native_run_metrics': run_comparisons}
