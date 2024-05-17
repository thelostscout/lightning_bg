import os
import warnings
from abc import ABC, abstractmethod
from functools import partial

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import lightning_trainable as lt
import torch
import torch.distributions as D
import torch.nn as nn
from FrEIA.utils import force_to
from lightning_trainable.hparams import AttributeDict

from .utils import AlignmentIC, ICTransform
from .fff_loss import fff_loss


def get_model_by_name(name: str) -> "type(BaseTrainable)":
    """
    Converts a model name into the actual model object. This method is deprecated. Use getattr(models, name) instead.

    :parameter name: Name of the network.
    :type name: str
    :return: Network class.
    :exception ValueError: If the network name is unknown.
    """
    warnings.warn("get_model_by_name is deprecated. Use getattr(models, name) instead.", DeprecationWarning)
    if name == "RNVPfwkl":
        return RNVPfwkl
    elif name == "RNVPICfwkl":
        return RNVPICfwkl
    elif name == "RNVPpseudofwkl":
        return RNVPpseudofwkl
    elif name == "RNVPvar":
        return RNVPvar
    elif name == "RNVPrvkl":
        return RNVPrvkl
    elif name == "RNVPrvklLatent":
        return RNVPrvklLatent
    elif name == "RQSfwkl":
        return RQSfwkl
    else:
        raise ValueError(f"Unknown network name {name}")


def latent_distribution_constructor(n_dims: int, **kwargs) -> D.Distribution:
    """
    Constructs a latent distribution from keyword arguments which describe the distribution.
    Currently supported distributions are:
        - Normal
        - Bimodal

    :param n_dims: the number of dimensions of the latent space.
    :type n_dims: int
    :param kwargs: arguments describing the distribution.
    :return: A distribution object.
    :exception ValueError: If the distribution name is unknown.

    :example:

    >>> latent_distribution_constructor(2, name="Bimodal", mus=[0, 10], sigmas=[1, 1])
    A bimodal distribution with two modes at 0 and 10 with standard deviation 1.
    """
    name = kwargs['name']

    if name == "Normal":
        sigma = kwargs['sigma']
        return D.MultivariateNormal(torch.zeros(n_dims), sigma * torch.eye(n_dims))
    elif name == "Bimodal":
        sigmas = torch.tensor(kwargs['sigmas'])[:, None] * torch.eye(n_dims)
        mus = torch.zeros((2, n_dims))
        mus[0, 0], mus[1, 0] = kwargs['mus']
        gausses = D.MultivariateNormal(mus, sigmas)
        weights = D.Categorical(torch.tensor([1, 1]))
        return D.MixtureSameFamily(weights, gausses)
    else:
        raise ValueError(f"Unknown distribution name {name}")


def exponential_subnet_constructor(
        subnet_max_width: int, subnet_depth: int, subnet_growth_factor: int, dims_in: int, dims_out: int
) -> torch.nn.Sequential:
    """
    Constructs a subnet with exponential growth and decay of the hidden layer widths (diamond shape).
    :param subnet_max_width: the maximum width of the subnet.
    :param subnet_depth: the amount of layers in the subnet.
    :param subnet_growth_factor: determines how fast the hidden layer sizes grow and decay.
    :param dims_in: size of the input layer.
    :param dims_out: size of the output layer.
    :return: a subnet object.
    """
    # Make sure that the subnet is not smaller than the input and output layers.
    assert subnet_max_width >= dims_in and subnet_max_width >= dims_out
    # Create subnet iteratively.
    dims_prev = dims_in
    layers = []
    for i in range(subnet_depth + 1):
        # As long as iterable is before halfway point, increase subnet size by growth factor.
        if i < subnet_depth / 2:
            dims_next = dims_prev * subnet_growth_factor
        # If iterable is at halfway point, keep subnet size the same.
        elif i == subnet_depth / 2:
            dims_next = dims_prev
        # If iterable is beyond halfway point, decrease subnet size by growth factor.
        else:
            dims_next = int(dims_prev / subnet_growth_factor)
        # Special case for last layer: output layer size is fixed and potentially different from input size.
        if i != subnet_depth:
            layers.append(nn.Linear(min(dims_prev, subnet_max_width), min(dims_next, subnet_max_width)))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(min(dims_prev, subnet_max_width), dims_out))
        # Update previous layer size.
        dims_prev = dims_next
    # Create sequential subnet.
    block = nn.Sequential(*layers)
    # Initialize weights and biases of the last layer to zero.
    block[-1].weight.data.zero_()
    block[-1].bias.data.zero_()
    return block


class BaseHParams(lt.TrainableHParams):
    """
    Base class for all hparams classes.
    It contains the parameters that are shared by all models.
    """
    inn_depth: int
    subnet_max_width: int
    subnet_depth: int
    subnet_growth_factor: int = 2
    n_dims: int
    latent_target_distribution: dict


class RvklHParams(BaseHParams):
    """
    Special hparams class for models that require alignment.
    is_molecule controls if an alignment penalty is required (which might not be the case for toy models).
    lambda_alignment controls the magnitude of the alignment penalty.
    """
    is_molecule: bool = True
    lambda_alignment: float | None = None


class RvklLatentHParams(RvklHParams):
    """ Special hparams class for models that use an INN as latent distribution. """
    latent_network_params: BaseHParams

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        # Make sure that the latent network has the same number of dimensions as the main network.
        hparams.latent_network_params.n_dims = hparams.n_dims
        return super().validate_parameters(hparams)


class BaseTrainable(lt.Trainable, ABC):
    """
    Base class for all models.
    Handles the following tasks shared by all models:
        - Construction of the INN and the latent distribution.
        - Distribution utilities like sampling and log probability.
        - Moving the model between devices.
    """
    hparams: BaseHParams
    # flags to indicate if the model requires addtional parameters
    needs_energy_function = False
    needs_system = False

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.inn = self.configure_inn()
        self.q = latent_distribution_constructor(self.hparams.n_dims, **self.hparams.latent_target_distribution)

    @abstractmethod
    def configure_inn(self):
        raise NotImplementedError

    def sample(self, size) -> torch.Tensor:
        """
        Samples from the latent distribution and transforms the samples to the data space.
        :param size: Size of the returned samples.
        :return: Samples in data space.
        """
        # Sampling is an independent operation and should be treated "as is" without influencing the gradients.
        with torch.no_grad():
            z = self.q.sample(size)
            x = self.inn(z, rev=True)[0]
        return x

    def log_prob(self, x):
        """
        Computes the log probability of the data space.
        :param x: points of which the log probability is computed.
        :return: log probability of the points.
        """
        z, log_det_JF = self.inn(x)
        return self.q.log_prob(z) + log_det_JF

    def to(self, *args, **kwargs):
        """
        Wrapper function to move the parameters of the latent distribution between devices.
        This is neccesary because torch distributions can't move their parameters between devices by default.
        """
        # 
        if isinstance(self.q, TrainableDistribution):
            self.q.model.to(*args, **kwargs)
        elif isinstance(self.q, D.Distribution):
            force_to(self.q, *args, **kwargs)
        else:
            raise ValueError(f"Unknown distribution type {type(self.q)} for latent distribution.")
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        # Delegate to .to(...) as it moves distributions, too.
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        return self.to(device)

    def cpu(self):
        # Delegate to .to(...) as it moves distributions, too.
        return self.to("cpu")


class TrainableDistribution(D.Distribution):
    """
    A wrapper to turn a BaseTrainable model into a torch distribution. Provides sampling and log probability methods,
    which is all that is used in the scope of this libary.
    """

    def __init__(self, model: BaseTrainable):
        self.model = model
        super().__init__(validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of points in data space.
        :param value: the points in data space.
        :return: the log probability of the points.
        """
        return self.model.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Samples from the latent distribution and maps the samples to the data space.
        :param sample_shape: a shape input for the latent distribution.
        :return: samples in data space.
        """
        return self.model.sample(sample_shape)


class BaseRNVP(BaseTrainable):
    """ Base class for all models that use a RealNVP network as INN (https://arxiv.org/abs/1605.08803)."""

    def configure_inn(self):
        """ Configures the INN with a RealNVP network and exponential subnets."""
        inn = Ff.SequenceINN(self.hparams.n_dims)
        # normalize inputs
        inn.append(Fm.ActNorm)
        # add coupling blocks
        for k in range(self.hparams.inn_depth):
            inn.append(
                Fm.RNVPCouplingBlock,
                subnet_constructor=partial(
                    exponential_subnet_constructor,
                    self.hparams.subnet_max_width,
                    self.hparams.subnet_depth,
                    self.hparams.subnet_growth_factor,
                )
            )
        return inn


class BaseRNVPIC(BaseTrainable):
    """ Base class for all models that use a RealNVP network together with internal coordinates."""
    needs_system = True

    def __init__(self, hparams, system=None, **kwargs):
        # we need the system to initialize the internal coordinates transform layer
        if system is None:
            raise ValueError("System must be provided for RNVPIC.")
        self.system = system
        super().__init__(hparams, **kwargs)

    def configure_inn(self):
        """ Configures the INN with a RealNVP network, and exponential subnets."""
        inn = Ff.SequenceINN(self.hparams.n_dims)
        # transform to normalised internal coordinates
        inn.append(ICTransform, system=self.system)
        # add coupling blocks
        for k in range(self.hparams.inn_depth):
            inn.append(
                Fm.RNVPCouplingBlock,
                subnet_constructor=partial(
                    exponential_subnet_constructor,
                    self.hparams.subnet_max_width,
                    self.hparams.subnet_depth,
                    self.hparams.subnet_growth_factor,
                )
            )
        return inn


class RNVPICfwkl(BaseRNVPIC):
    """ RealNVP model in internal coordinates with forward KL loss."""

    def forward_kl_loss(self, z, log_det_JF):
        """
        Computes the forward KL loss.

        :param z: Samples from the latent distribution.
        :param log_det_JF: the log determinant of the Jacobian of the INN.
        :return: forward KL loss.
        """
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx):
        z, log_det_JF = self.inn(batch)
        loss = self.forward_kl_loss(z, log_det_JF).mean()
        return dict(
            loss=loss,
        )


class BaseRNVPEnergy(BaseRNVP):
    """Base class for all models that use a RealNVP network together and require an energy function."""
    needs_energy_function = True
    needs_alignment = False

    def __init__(self, hparams, energy_function=None, **kwargs):
        if energy_function is None:
            raise ValueError("Energy function must be provided for RNVPEnergy.")
        super().__init__(hparams, **kwargs)
        # we implement the log probability of the true distribution instead of the energy function because it is more
        # general in its usage.
        self.pstar_log_prob = lambda x: - energy_function(x)
        pass


class RNVPfwkl(BaseRNVP):
    """ RealNVP model with forward KL loss."""
    hparams: BaseHParams

    def forward_kl_loss(self, z, log_det_JF):
        """
        Computes the forward KL loss.

        :param z: Samples from the latent distribution.
        :param log_det_JF: the log determinant of the Jacobian of the INN.
        :return: forward KL loss.
        """
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx):
        z, log_det_JF = self.inn(batch)
        loss = self.forward_kl_loss(z, log_det_JF).mean()
        return dict(
            loss=loss,
        )


class RNVPfff(BaseRNVP):
    """ RealNVP model with free-form flow (nll-surrogate) loss."""
    hparams: BaseHParams

    def compute_metrics(self, batch, batch_idx):
        print(batch)
        loss = fff_loss(
            batch,
            partial(self.inn, rev=False, jac=False),
            partial(self.inn, rev=True, jac=False),
            beta=10,
        )
        return dict(
            loss=loss,
        )



class RNVPpseudofwkl(BaseRNVPEnergy):
    """ RealNVP model with pseudo forward KL loss. """
    hparams: BaseHParams

    def pseudo_forward_kl_loss(self, hatx):
        """
        Computes the pseudo forward KL loss (https://arxiv.org/abs/2301.05475).
        :param hatx: Samples generated by the model (without a gradient).
        :return: pseudo forward KL loss.
        """
        # we calculate the normal forward KL loss
        z, log_det_JF = self.inn(hatx)
        log_hatp = self.q.log_prob(z) + log_det_JF
        # additionally, we calculate the log probability of the true distribution for the reweighting factor
        log_pstar = self.pstar_log_prob(hatx)
        # the reweighting factor is calculated without a gradient
        with torch.no_grad():
            # instead of the ratio p*/p_hat, we use the softmax here for numerical stability
            reweight = log_pstar.softmax(dim=0) / (log_hatp.softmax(dim=0) + 1e-9)
        return reweight * log_hatp, reweight

    def compute_metrics(self, batch, batch_idx):
        with torch.no_grad():
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            hatx = self.inn(z, rev=True)[0]
        loss, reweight = self.pseudo_forward_kl_loss(hatx)
        # currently, reweight and energy are also tracked to get a better understanding of the performance of this loss
        return dict(
            loss=loss.mean(),
            reweight=reweight.mean(),
            energy=self.pstar_log_prob(hatx).mean().cpu().detach()
        )


class RNVPrvkl(BaseRNVPEnergy):
    """ RealNVP model with reverse KL loss. """
    hparams: RvklHParams
    needs_alignment = True

    def __init__(self, hparams, energy_function=None, alignment_penalty=None, **kwargs):
        if alignment_penalty is None:
            raise ValueError("Alignment penalty must be provided for RNVPrvkl.")
        super().__init__(hparams, energy_function, **kwargs)
        # the alignment penalty is only used for molecules
        self.is_molecule = self.hparams.is_molecule
        self.alignment_penalty = alignment_penalty

    def reverse_kl_loss(self, z):
        """
        Computes the reverse KL loss.
        :param z: Samples from the latent distribution.
        :return: reverse KL loss.
        """
        hatx, log_det_JG = self.inn(z, rev=True)
        log_pstar = self.pstar_log_prob(hatx)
        # we only need alignment if the model is trained on molecules, because then the energy function is degenerated
        if self.is_molecule:
            alignment_penalty_loc, alignment_penalty_rot = self.alignment_penalty(hatx)
            alignment_penalty_loc, alignment_penalty_rot = alignment_penalty_loc.to(
                self.device) * self.hparams.lambda_alignment, alignment_penalty_rot.to(
                self.device) * self.hparams.lambda_alignment
        else:
            alignment_penalty_loc, alignment_penalty_rot = 0, 0

        # the reverse KL loss is extended by an alignment penalty. Substracting the log of p* is equivalent to adding
        # the energy. The alignment penalty, which adds a positive energy contribution, should therefore also be added
        return - log_pstar - log_det_JG + alignment_penalty_loc + alignment_penalty_rot

    def compute_metrics(self, batch, batch_idx) -> dict:
        # noinspection PyTypeChecker
        z = self.q.sample((self.hparams.batch_size,))
        loss = self.reverse_kl_loss(z)

        return dict(
            loss=loss.mean(),
        )


class RNVPvar(RNVPrvkl):
    """ RealNVP model with variance loss. """

    def var_loss(self, hatx):
        """
        Computes the variance loss (https://arxiv.org/abs/2301.05475).

        :param hatx: Samples generated by the model (without a gradient).
        :return: variance loss.
        """
        # calculate the normal forward KL loss
        z, log_det_JF = self.inn(hatx)
        log_hatp = self.q.log_prob(z) + log_det_JF
        # need the log probability of the true distribution for the log ratios
        log_pstar = self.pstar_log_prob(hatx)
        log_ratios = log_pstar - log_hatp

        # the mean of the log ratios acts as a mask and is calculated without a gradient
        # (https://arxiv.org/abs/2301.05475)
        with torch.no_grad():
            K = log_ratios.mean()

        # only molecules need alignment
        if self.is_molecule:
            alignment_penalty_loc, alignment_penalty_rot = self.alignment_penalty(hatx)
            alignment_penalty_loc, alignment_penalty_rot = alignment_penalty_loc.to(
                self.device) * self.hparams.lambda_alignment, alignment_penalty_rot.to(
                self.device) * self.hparams.lambda_alignment
        else:
            alignment_penalty_loc, alignment_penalty_rot = 0, 0
        return torch.nn.functional.relu(log_ratios - K).square() + alignment_penalty_loc + alignment_penalty_rot

    def compute_metrics(self, batch, batch_idx):
        with torch.no_grad():
            # noinspection PyTypeChecker
            z = self.q.sample((self.hparams.batch_size,))
            hatx = self.inn(z, rev=True)[0]

        loss = self.var_loss(hatx)
        return dict(
            loss=loss.mean(),
            energy=self.pstar_log_prob(hatx).mean().cpu().detach()
        )


class RNVPrvklLatent(RNVPrvkl):
    """ RealNVP model with reverse KL loss and an INN as latent distribution. """
    hparams: RvklLatentHParams

    def __init__(self, hparams, energy_function, alignment_penalty: AlignmentIC.penalty, **kwargs):
        super().__init__(hparams, energy_function, alignment_penalty, **kwargs)
        # if there is a latent network already trained, we can use it to initialize the latent distribution.
        # otherwise, we have to train it from scratch.
        if hparams.latent_network_params.model_checkpoint is None:
            hparams.latent_network_params.model_checkpoint = dict(dirpath="latent")
        else:
            hparams.latent_network_params.model_checkpoint["dirpath"] = "latent"
        model = RNVPfwkl(hparams.latent_network_params, **kwargs)
        self.q = TrainableDistribution(model)

    def fit(self, logger_kwargs: dict = None, trainer_kwargs: dict = None, fit_kwargs: dict = None) -> dict:
        # set latent name in logger
        if logger_kwargs is not None:
            latent_logger_kwargs = logger_kwargs.copy()
        else:
            latent_logger_kwargs = dict()
        latent_logger_kwargs["sub_dir"] = "latent"

        # make sure that version number is the same accross latent and main training
        if logger_kwargs.get("version", None) is None:
            version_number = self.find_next_version(logger_kwargs["save_dir"], logger_kwargs["name"])
            latent_logger_kwargs["version"] = f"version_{version_number}"
            logger_kwargs["version"] = f"version_{version_number}"
        self.q.model.fit(latent_logger_kwargs, trainer_kwargs, fit_kwargs)
        return super().fit(logger_kwargs, trainer_kwargs, fit_kwargs)

    @staticmethod
    def find_next_version(lightning_logs, param_name):
        """
        Finds the next version number for a given parameter name.
        :param lightning_logs: lightning log directory.
        :param param_name: name of the parameter.
        :return: the next version number.
        """
        dir_path = os.path.join(lightning_logs, param_name)
        dirs = os.listdir(dir_path)
        dirs_with_version = [f for f in dirs if f.startswith("version_")]
        if not dirs_with_version:
            return 0
        else:
            versions = [int(f.split("_")[1]) for f in dirs_with_version]
            return max(versions) + 1


class BaseRQS(BaseTrainable):
    """
    Base class for all models that use a Rational Quadratic Spline network as INN (https://arxiv.org/abs/1906.04032).
    """
    def configure_inn(self):
        """ Configures the INN with a Rational Quadratic Spline network and exponential subnets."""
        inn = Ff.SequenceINN(self.hparams.n_dims)
        # normalize inputs
        inn.append(Fm.ActNorm)
        # add coupling blocks
        for k in range(self.hparams.inn_depth):
            inn.append(
                Fm.RationalQuadraticSpline,
                subnet_constructor=partial(
                    exponential_subnet_constructor,
                    self.hparams.subnet_max_width,
                    self.hparams.subnet_depth,
                    self.hparams.subnet_growth_factor,
                )
            )
        return inn


class RQSfwkl(BaseRQS):
    """ Rational Quadratic Spline model with forward KL loss."""
    def forward_kl_loss(self, z, log_det_JF):
        """
        Computes the forward KL loss.
        :param z: Samples from the latent distribution.
        :param log_det_JF: the log determinant of the Jacobian of the INN.
        :return: forward KL loss.
        """
        log_likelihood = self.q.log_prob(z) + log_det_JF
        return - log_likelihood

    def compute_metrics(self, batch, batch_idx):
        z, log_det_JF = self.inn(batch)
        loss = self.forward_kl_loss(z, log_det_JF).mean()
        return dict(
            loss=loss,
        )
