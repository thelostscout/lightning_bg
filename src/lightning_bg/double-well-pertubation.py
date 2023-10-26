from lightning_bg.architectures import *
from lightning_bg.utils import SingleTensorDataset
import torch
import torch.distributions as D
import sys
import os
import yaml
from collections import OrderedDict
import pandas as pd


# noinspection PyShadowingNames
def remove_logger(model: BaseTrainable):
    def no_logger(*args, **kwargs):
        return None

    model.configure_logger = no_logger
    return model


def set_context():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    if device == torch.device("cuda"):
        torch.set_float32_matmul_precision('medium')
    # a context tensor to send data to the right device and dtype via '.to(ctx)'
    ctx = {"device": device, "dtype": dtype}
    return ctx


def gaussian_mixture_model(means, stds, weights):
    means = torch.Tensor(means)
    stds = torch.Tensor(stds)

    distribution = D.MixtureSameFamily(
        D.Categorical(probs=torch.Tensor(weights)),
        D.Independent(D.Normal(loc=means, scale=stds), 1)
    )
    return distribution


# noinspection PyShadowingNames
def perturb_state_dict(state_dict: OrderedDict, scale):
    new_state_dict = state_dict.copy()
    for k, v in new_state_dict.items():
        if "is_initialized" not in k:
            new_state_dict[k] += scale * torch.randn_like(v)
    return new_state_dict


if __name__ == "__main__":
    # read yaml param file
    param_path = sys.argv[1]
    param_name = os.path.basename(param_path)[-5]
    with open(param_path) as f:
        params = yaml.load(f, yaml.FullLoader)  # load the parameters

    # create a default datapath if none is given
    data_path = params.get("data_path", None)
    if not data_path:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.realpath(file_dir + "/../data/double-well/") + "/"
    else:
        data_path = os.path.expanduser(data_path)
        print("loading data path from yaml file: ", data_path)
        data_path = os.path.realpath(data_path) + "/"

    context = set_context()
    GMM = gaussian_mixture_model(params['means'], params['stds'], params['weights'])
    params['network_params']['n_dims'] = len(torch.Tensor(params['means'])[0].flat)

    # read model and parameter class from params
    ModelClass = get_network_by_name(params['network_name'])
    PreModelClass = get_network_by_name("RNVPfwkl")
    ParamClass = BaseHParams
    hparams = ParamClass(**params['network_params'])
    prehparams = hparams.copy()
    prehparams.early_stopping = 5
    prehparams.max_epochs = 10

    # generate samples
    n_samples = 1_000_000
    train_split = params['training_params']['train_split']
    # noinspection PyTypeChecker
    train_data = SingleTensorDataset(GMM.sample((int(n_samples * train_split),)))
    # noinspection PyTypeChecker
    val_data = SingleTensorDataset(GMM.sample((int(n_samples * (1 - train_split)),)))

    # train with fwkl to get optimal solution
    premodel_path = os.path.join(data_path, "premodels", param_name)
    redo_premodel = params.get('redo_premodel', False)
    if not os.path.exists(premodel_path) or redo_premodel:
        # train the model
        premodel = PreModelClass(prehparams, train_data=train_data, val_data=val_data)
        premodel.fit(logger_kwargs=dict(save_dir=os.path.join(data_path, "premodels"), version=param_name))
    else:
        # load trained model
        premodel = PreModelClass.load_from_checkpoint(premodel_path + "/checkpoints/last.ckpt", hparams=prehparams)
    state_dict = premodel.state_dict()

    # perturb the model x times and log the results
    results = []
    for i in range(params['iterations']):
        # create a new model
        model = ModelClass(hparams, train_data=train_data, val_data=val_data)
        model = remove_logger(model)
        # load perturbed state dict
        perturbed_state_dict = perturb_state_dict(state_dict, params['scale'])
        model.load_state_dict(perturbed_state_dict)
        # train model and log results
        results.append(model.fit())
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(data_path, param_name + ".csv"))