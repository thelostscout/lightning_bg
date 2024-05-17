import math
import os
from typing import Iterable, Tuple

import FrEIA.modules as Fm
import bgflow
import bgmol
import mdtraj
import torch
from FrEIA.utils import force_to
from torch import Tensor

# TODO: use a premade backbone searcher
def align_backbone(coordinates, system: bgmol.systems.OpenMMSystem):
    """
    Aligns the backbone of all molecules provided in coordinates to the first molecule.
    :param coordinates: Coordinates of the molecules to align.
    :param system: The system that describes the molecules.
    :return: The coordinates of the aligned molecules.
    """
    # find all the atoms which are not hydrogen
    df = system.mdtraj_topology.to_dataframe()[0]
    atom_idx = (df['element'] != "H").astype(int).to_numpy().nonzero()[0]
    # superpose trajectories
    traj = mdtraj.Trajectory(coordinates, system.mdtraj_topology)
    traj = traj.superpose(traj, frame=0, atom_indices=atom_idx, parallel=True)
    return traj.xyz


def load_data(data_path):
    """
    Loads molecule data from a given path. Currently supported are alanine dipeptide and peptides.
    :param data_path: The path to where the data is stored. If alanine dipeptide is supposed to be used, the path must
    contain "alanine_dipeptide". If peptides are supposed to be used, the path must contain a top.pdb and a traj.h5
    file.
    :return: The coordinates of the molecules and the system that describes them.
    """
    # for the purpose of the project, only two types of data where required: alanine dipeptide and custom peptides
    # TODO: generalize this function to support more types of molecules or other data types
    if "alanine_dipeptide" in data_path:
        is_data_here = os.path.exists(data_path + "/Ala2TSF300.npy")
        ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=data_path)
        system = ala_data.system
        coordinates = ala_data.coordinates
    else:
        with open(data_path.rstrip("/") + "/top.pdb", 'r') as file:
            # read the topology file to get the number of atoms and residues
            lines = file.readlines()
            lastline = lines[-3]
            n_atoms = int(lastline[4:11].strip())
            n_res = int(lastline[22:26].strip())
            # the peptide module requires the number of atoms and residues to be specified
            system = bgmol.systems.peptide(short=False, n_res=n_res, n_atoms=n_atoms, filepath=data_path)
            traj = mdtraj.load_hdf5(data_path.rstrip("/") + "/traj.h5")
            coordinates = traj.xyz
    # need to reinitialize the energy model with n_workers=1 due to a bug that prevents the energy model from
    # shutting down after initialization with n_workers>1 (https://github.com/noegroup/bgflow/issues/35)
    system.reinitialize_energy_model(temperature=300., n_workers=1)
    return coordinates, system


# TODO: this tasked should be done by the model itself in the future
def load_model_kwargs(ModelClass: "BaseTrainable", train_data, val_data, system: bgmol.systems.OpenMMSystem):
    """
    A helper function that loads all extra arguments a class might need.
    :param ModelClass: The class of the model that needs the arguments.
    :param train_data: The training data.
    :param val_data: The validation data.
    :param system: The system that describes the molecules.
    :return: A dictionary containing all the arguments.
    """
    kwargs = dict(train_data=train_data, val_data=val_data)
    # add the energy function, the system, and the alignment penalty if the model needs them
    if ModelClass.needs_energy_function:
        kwargs['energy_function'] = system.energy_model.energy
        if ModelClass.needs_alignment:
            kwargs['alignment_penalty'] = AlignmentIC(system).penalty
    elif ModelClass.needs_system:
        kwargs['system'] = system
    return kwargs


class SingleTensorDataset(torch.utils.data.TensorDataset):
    """A dataset that only contains a single tensor."""
    def __init__(self, tensor):
        super().__init__(tensor)

    def __getitem__(self, index):
        # Override to make the dataset behave like a tensor when accessed
        return self.tensors[0][index]

    def get_tensor(self):
        return self.tensors[0]

    def to(self, *args, **kwargs):
        force_to(self.q, *args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        # Delegate to .to(...) as it moves distributions, too
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        return self.to(device)

    def cpu(self):
        # Delegate to .to(...) as it moves distributions, too
        return self.to("cpu")


class CoordinateDataset(SingleTensorDataset):
    """
    A dataset that contains coordinates of molecules.
    Automatically handles splits and alignment to a reference molecule.
    """
    def __init__(self, coordinates, system, mode, val_split=.1, test_split=.2, seed=42):
        """
        :param coordinates: The coordinates of the molecules.
        :param system: The system that describes the molecules.
        :param mode: train, val, test, or no_split.
        :param val_split: The fraction of the data that should be used for validation.
        :param test_split: The fraction of the data that should be used for testing.
        :param seed: Predefined seed for the random number generator to ensure reproducibility.
        """
        # the dataset should know which mode it is in
        self.mode = mode
        # align the molecules to the first molecule
        N = coordinates.shape[0]
        coordinates = align_backbone(coordinates, system)
        coordinate_tensor = torch.Tensor(coordinates.reshape(N, -1))
        # save the reference molecule
        self.reference_molecule = coordinates[0]
        # split the data
        val_len = int(math.floor(N * val_split))
        test_len = int(math.floor(N * test_split))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(N, generator=generator)
        # select the indices according to the mode
        if mode == 'test':
            idx = indices[:test_len]
        elif mode == 'val':
            idx = indices[test_len: test_len + val_len]
        elif mode == 'train':
            idx = indices[test_len + val_len:]
        elif mode == 'no_split':
            idx = indices
        else:
            raise ValueError(f"{mode} is not a valid mode. Choose from ['train', 'test', 'val', 'no_split']")

        super().__init__(coordinate_tensor[idx])


def dataset_setter(coordinates, system, val_split=.1, test_split=.2, seed=42):
    """
    A helper function that creates the datasets for training, validation, and testing.
    :param coordinates: The coordinates of the molecules.
    :param system: The system that describes the molecules.
    :param val_split: The fraction of the data that should be used for validation.
    :param test_split: The fraction of the data that should be used for testing.
    :param seed: The seed for the random number generator to ensure reproducibility.
    :return: A training, test, and validation dataset.
    """
    modes = ['train', 'val', 'test']
    out = [CoordinateDataset(coordinates, system, mode, val_split, test_split, seed) for mode in modes]
    # print("mode order: " + s[:-2])
    return out


class AlignmentRMS:
    """
    A class that calculates the root-mean-square deviation of the coordinates of a molecule to a reference molecule.
    This alignment method performs badly and should not be used.
    """
    def __init__(self, system, reference_molecule):
        self.reference_molecule = mdtraj.Trajectory(reference_molecule, system.mdtraj_topology)
        df = system.mdtraj_topology.to_dataframe()[0]
        self.atom_idx = (df['element'] != "H").astype(int).to_numpy().nonzero()[0]

    def penalty(self, x: torch.Tensor):
        """
        Calculates the root-mean-square deviation of the coordinates of a molecule to a reference molecule.
        :param x: The coordinates of the molecule.
        :return: The root-mean-square deviation.
        """
        with torch.no_grad():
            x = x.cpu()
            x = x.reshape(*x.shape[:-1], *self.reference_molecule.xyz.shape[-2:])
            traj = mdtraj.Trajectory(x, self.reference_molecule.topology)
            traj = traj.superpose(self.reference_molecule, frame=0, atom_indices=self.atom_idx, parallel=True)
        return torch.mean((x - traj.xyz).reshape(*x.shape[:-2], -1) ** 2, dim=-1)


class AlignmentIC:
    """
    A class that calculates the deviation of a molecule with respect the origin and a default orientation.
    The penalty is designed as a quadratic potential.
    """
    def __init__(self, system):
        # initialize the internal coordinate layer
        zfactory = bgmol.zmatrix.ZMatrixFactory(system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        self.ic_layer = bgflow.GlobalInternalCoordinateTransformation(zmatrix)

    def penalty(self, x: torch.Tensor):
        """
        Calculates the deviation of a molecule with respect the origin and a default orientation.
        :param x: The coordinates of the molecule.
        :return: The deviation as a quadratic potential.
        """
        # get translation and rotation of the molecules from the internal coordinate transform
        _0, _1, _2, loc, rot, _5 = self.ic_layer(x)
        # loc comes in a [1, 3] shape, but we want [3]
        loc = loc.squeeze()
        return torch.sum(loc ** 2, dim=-1), torch.sum(rot ** 2, dim=-1)


class ICTransform(Fm.InvertibleModule):
    """
    A class that wraps the internal coordinate layer of bgflow to make it compatible with FrEIA.
    """
    def __init__(self, dims_in, dims_c=None, system=None):
        super().__init__(dims_in, dims_c)
        # need the zmatrix to initialize the internal coordinate layer
        zfactory = bgmol.zmatrix.ZMatrixFactory(system.mdtraj_topology)
        zmatrix, fixed_atoms = zfactory.build_naive()
        # the internal coordinate layer is fixed and does not depend on learnable parameters
        self.bg_layer = bgflow.GlobalInternalCoordinateTransformation(zmatrix)

    def output_dims(self, input_dims):
        # the internal coordinate layer does not change the shape of the input
        return input_dims

    def forward(
            self, x_or_z: Iterable[Tensor],
            c: Iterable[Tensor] = None,
            rev: bool = False,
            jac: bool = True
    ) -> Tuple[Tuple[Tensor], Tensor]:
        x_or_z = x_or_z[0]
        if not rev:
            # the internal coordinate layer splits the output into many different tensors, but we need one tensor
            # with all dimensions
            bonds, angles, torsions, loc, rot, log_jac_det = self.bg_layer._forward(x_or_z)
            out = torch.cat([bonds, angles, torsions, loc.squeeze(1), rot], dim=1)
        else:
            # need to split up the output into the different tensors again
            bonds = x_or_z[:, :self.bg_layer.dim_bonds]
            angles = x_or_z[:, self.bg_layer.dim_bonds:self.bg_layer.dim_bonds + self.bg_layer.dim_angles]
            torsions = x_or_z[:,
                              self.bg_layer.dim_bonds + self.bg_layer.dim_angles:self.bg_layer.dim_bonds + self.bg_layer.dim_angles + self.bg_layer.dim_torsions]
            # loc needs a [1, 3] shape, but we have [3]
            loc = x_or_z[-6:-3].unsqueeze(1)
            rot = x_or_z[-3:]
            out, log_jac_det = self.bg_layer._inverse(bonds, angles, torsions, loc, rot)
        return (out,), log_jac_det.squeeze(1)
