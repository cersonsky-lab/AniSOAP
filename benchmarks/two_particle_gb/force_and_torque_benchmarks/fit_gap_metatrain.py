from ase.io import read
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.data import Dataset
from omegaconf import OmegaConf
from metatrain.utils.data.target_info import get_energy_target_info

target_cfg = OmegaConf.create({
    "quantity": "energy",
    "unit": "eV",
})

energy_info = get_energy_target_info(
    target_name="energy",
    target=target_cfg,
    add_position_gradients=True,
)

dataset_info = DatasetInfo(
    length_unit="angstrom",
    atomic_types=[0],  # H, O
    targets={
        "energy": energy_info
    },
)

from metatrain.gap import GAP
lmax=6
nmax=9
HYPER_PARAMETERS = {"soap":{
        "cutoff": {
            "radius": 3.0,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 1.0,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": lmax,
            "radial": {"type": "Gto", "max_radial": nmax},
        },
    },
    "krr": {"num_sparse_points": 10, "degree": 1},
    "zbl": False
}
g = GAP(hypers=HYPER_PARAMETERS, dataset_info=dataset_info)

import torch
from ase.io import read
from metatomic.torch import systems_to_torch
from metatensor.torch import TensorMap, TensorBlock, Labels
from metatrain.utils.data import Dataset

frames = read("../benchmarks/two_particle_gb/force_and_torque_benchmarks/side_to_side.xyz", ":")

systems = systems_to_torch(frames, dtype=torch.float64)

def energy_target(atoms):
    n_atoms = len(atoms)

    block = TensorBlock(
        values=torch.tensor([[atoms.get_potential_energy()]], dtype=torch.float64),
        samples=Labels(
            ["system"],
            torch.tensor([[0]], dtype=torch.int32),
        ),
        components=[],
        properties=Labels(
            ["energy"],
            torch.tensor([[0]], dtype=torch.int32),
        ),
    )

    forces = torch.tensor(atoms.get_forces(), dtype=torch.float64)

    grad_block = TensorBlock(
        values=-forces.reshape(n_atoms, 3, 1),  # dE/dr = -F
        samples=Labels(
            ["sample", "system", "atom"],
            torch.tensor([[0, 0, i] for i in range(n_atoms)], dtype=torch.int32),
        ),
        components=[
            Labels(
                ["xyz"],
                torch.tensor([[0], [1], [2]], dtype=torch.int32),
            )
        ],
        properties=Labels(
            ["energy"],
            torch.tensor([[0]], dtype=torch.int32),
        ),
    )

    block.add_gradient("positions", grad_block)

    return TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )

dataset = Dataset.from_dict({
    "system": systems,
    "energy": [energy_target(atoms) for atoms in frames],
})