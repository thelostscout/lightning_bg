import sys
import os

import bgmol
from bgmol.systems.peptide import peptide
import mdtraj
import yaml
import torch

from lightning_bg import models
from lightning_bg.utils import dataset_setter, AlignmentIC

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    # read yaml param file
    param_path = sys.argv[1]
    param_name = os.path.splitext(os.path.basename(param_path))[0]
    with open(param_path) as f:
        params = yaml.load(f, yaml.FullLoader)  # load the parameters

    # read progress bar flag
    if "--disable_progress_bar" in sys.argv:
        params['trainer_kwargs']['enable_progress_bar'] = False

    # create a default datapath if none is given
    data_path = params.get("data_path", None)
    if not data_path:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.realpath(file_dir + "/../data/") + "/"
    else:
        data_path = os.path.expanduser(data_path)
        print("loading data path from yaml file: ", data_path)
        data_path = os.path.realpath(data_path) + "/"

    # set log path
    lightning_logs = os.path.join(data_path, "lightning_logs", params['molecule'].lstrip("/"))
    # import data
    # currently only alanine dipeptide and custom peptides are supported
    # TODO: add support for other molecules
    # TODO: this should be handled by the model class itself in the future or at least by a proper data loader
    molecule_path = os.path.join(data_path, "Molecules", params['molecule'].lstrip("/"))
    if params['molecule'] == "alanine_dipeptide":
        # import alanine data
        is_data_here = os.path.exists(molecule_path + "/Ala2TSF300.npy")
        print(f"Is data here? {is_data_here}, molecule path: {molecule_path}")
        if not os.path.exists(molecule_path):
            os.makedirs(molecule_path)
        ala_data = bgmol.datasets.Ala2TSF300(download=not is_data_here, read=True, root=molecule_path)
        # define system & energy model
        system = ala_data.system
        # need to reinitialize the energy model to set n_workers to 1 due to a bug:
        # (https://github.com/noegroup/bgflow/issues/35)
        system.reinitialize_energy_model(temperature=300., n_workers=1)
        energy_model = system.energy_model
        coordinates = ala_data.coordinates
    else:
        # read the top.pdb file to determine the number of atoms and residues
        with open(molecule_path.rstrip("/") + "/top.pdb", 'r') as file:
            lines = file.readlines()
            lastline = lines[-3]
            n_atoms = int(lastline[4:11].strip())
            n_res = int(lastline[22:26].strip())
            print(f"Number of atoms: {n_atoms}, residues: {n_res}")

        # define system & energy model
        system = peptide(short=False, n_atoms=n_atoms, n_res=n_res, filepath=molecule_path)
        # need to reinitialize the energy model to set n_workers to 1 due to a bug:
        # (https://github.com/noegroup/bgflow/issues/35)
        system.reinitialize_energy_model(temperature=300., n_workers=1)
        energy_model = system.energy_model

        # read coordinates
        traj = mdtraj.load_hdf5(molecule_path + "/traj.h5")
        coordinates = traj.xyz
        assert coordinates.shape[-2] == n_atoms, (f"pdb file ({n_atoms}) atoms does not match the "
                                                  f"data ({coordinates.shape[-2]}).")
    # determine n_dims
    params['network_params']['n_dims'] = len(coordinates[0].flat)
    # adjust learning rate
    try:
        params['network_params']['optimizer']['lr'] /= params['network_params']['n_dims']
        print(f"Dividing learning rate by n_dims. New lr: {params['network_params']['optimizer']['lr']}.")
    except KeyError:
        print("No learning rate found in optimizer dict. Not adjusting learning rate.")

    # load model class and corresponding param class
    ModelClass = getattr(models, params['network_name'])
    ParamClass = ModelClass.hparams_type
    hparams = ParamClass(**params['network_params'])

    # prepare the dataloaders
    train_split = params['training_params']['train_split']
    train_data, val_data, test_data = dataset_setter(
        coordinates, system, val_split=(.8 - train_split), test_split=.2, seed=42
    )
    print(f"{len(train_data)} training data, {len(val_data)} validation data, {len(test_data)} test data.")
    # create model
    # TODO: this should be handled by the model class itself in the future
    if ModelClass.needs_energy_function:
        if ModelClass.needs_alignment:
            alignment = AlignmentIC(system)
            model = ModelClass(
                hparams, energy_model.energy, alignment.penalty, train_data=train_data, val_data=val_data
            )
        else:
            model = ModelClass(
                hparams, energy_model.energy, train_data=train_data, val_data=val_data
            )
    elif ModelClass.needs_system:
        model = ModelClass(
            hparams, system, train_data=train_data, val_data=val_data
        )
    else:
        model = ModelClass(
            hparams, train_data=train_data, val_data=val_data
        )
    # load model state from previous experiment if a checkpoint is given
    load_from_checkpoint = params.get('load_from_checkpoint', None)
    if load_from_checkpoint is not None:
        checkpoint_path = os.path.join(data_path, "lightning_logs", load_from_checkpoint.lstrip("/"))
        print(f"loading state_dict from {checkpoint_path}", os.path.exists(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    # fit the model
    model.fit(
        trainer_kwargs=params['trainer_kwargs'],
        logger_kwargs=dict(save_dir=lightning_logs, name=param_name)
    )
    print("done")
