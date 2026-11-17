#!/usr/bin/env python3
"""Minimal installed gpmeep two-rank/two-GPU MPI execution example."""

import json

from mpi4py import MPI
import meep as mp
from meep.backend_policy import BackendRecommendationMismatch


ACKNOWLEDGEMENT_REASON = (
    "Installed two-GPU example intentionally demonstrates CUDA below crossover"
)


communicator = MPI.COMM_WORLD
if communicator.size != 2 or mp.count_processors() != 2:
    raise RuntimeError("launch this example with exactly two MPI ranks")
mp.gpu.set_backend("cuda")
mp.gpu.reset_statistics()
simulation = mp.Simulation(
    cell_size=mp.Vector3(6, 4),
    resolution=24,
    boundary_layers=[mp.PML(0.5)],
    sources=[
        mp.Source(
            mp.GaussianSource(0.3, fwidth=0.1),
            component=mp.Ez,
            center=mp.Vector3(-1.5),
        )
    ],
)
communicator.Barrier()
try:
    simulation.run(until=12)
except BackendRecommendationMismatch as mismatch:
    # This is a quick topology smoke, not a speed claim. Each rank confirms
    # the same exact mismatch challenge before retrying the unchanged run.
    mp.gpu.acknowledge_backend_mismatch(
        mismatch.challenge, ACKNOWLEDGEMENT_REASON
    )
    simulation.run(until=12)
local = {
    "rank": communicator.rank,
    "active_backend": mp.gpu.active_backend,
    "selected_device_identifier": mp.gpu.selected_device_identifier,
    "cuda_execution_selected": bool(
        simulation.fields.gpu_cuda_execution_selected()
    ),
    "backend_recommendation": mp.gpu.recommendation,
    "statistics": mp.gpu.statistics(),
}
records = communicator.gather(local, root=0)
simulation.reset_meep()
communicator.Barrier()
if communicator.rank == 0:
    assert records is not None
    identifiers = [record["selected_device_identifier"] for record in records]
    if any(
        record["active_backend"] != "cuda"
        or not record["cuda_execution_selected"]
        for record in records
    ):
        raise RuntimeError("a rank did not execute strict CUDA work")
    if len(set(identifiers)) != 2:
        raise RuntimeError("MPI ranks did not select distinct physical GPUs")
    print(json.dumps({"state": "PASS", "ranks": records}, sort_keys=True))
