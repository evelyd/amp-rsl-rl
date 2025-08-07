from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

import escnn
from escnn.gspaces import *
from escnn.nn import FieldType, EquivariantModule, GeometricTensor
import torch.nn.functional as F
from typing import List, Tuple, Any
import numpy as np

from hydra import compose, initialize

from morpho_symm.nn.EMLP import EMLP
from morpho_symm.utils.robot_utils import group_rep_from_gens
from amp_rsl_rl.dha_utils import isaaclab_joints_to_ms, ms_joints_to_isaaclab

class ExportedActorMoESymm(nn.Module):
            def __init__(self, experts, gate, softmax_fn, ms_obs_dim):
                super().__init__()
                self.experts = experts
                self.gate = gate
                self.softmax_fn = softmax_fn
                self.ms_obs_dim = ms_obs_dim

            def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
                expert_out_tensors = torch.stack([e(x_tensor) for e in self.experts], dim=-1)
                gate_logits = self.gate(x_tensor)
                weights = self.softmax_fn(gate_logits).unsqueeze(1)
                output = (expert_out_tensors * weights).sum(dim=-1)
                return output

class ActorMoESymm(nn.Module):
    """
    Mixture-of-Experts actor:  ⎡expert_1(x) … expert_K(x)⎤·softmax(gate(x))
    """

    def __init__(
        self,
        in_field_type: FieldType,
        gating_out_field_type: FieldType,
        out_field_type: FieldType,
        obs_dim: int,
        hidden_dims,
        num_experts: int = 4,
        gate_hidden_dims: list[int] | None = None,
        activation="elu",
    ):
        super().__init__()
        self.ms_obs_dim = in_field_type.size
        self.num_experts = num_experts
        self.in_field_type = in_field_type
        self.out_field_type = out_field_type

        # experts
        self.experts = nn.ModuleList(
            [SimpleEMLP(in_field_type, out_field_type, hidden_dims=hidden_dims, activation=activation) for _ in range(num_experts)]
        )

        # gating network
        self.gate = SimpleEMLP(in_field_type, gating_out_field_type,
            hidden_dims=gate_hidden_dims,
            activation=activation)

        self.softmax = SimpleEMLP.get_activation("softmax", gating_out_field_type)

    def forward(self, x: torch.GeometricTensor) -> torch.GeometricTensor:
        """
        Args:
            x: [batch, obs_dim]
        Returns:
            mean action: [batch, act_dim]
        """
        expert_out = torch.stack([e(x).tensor for e in self.experts], dim=-1)
        gate_logits = self.gate(x)  # [batch, K]
        weights = self.softmax(gate_logits).tensor.unsqueeze(1)  # [batch, 1, K]
        output = (expert_out * weights).sum(dim=-1)  # [batch, act_dim]
        return GeometricTensor(output, self.out_field_type)

    def export(self) -> nn.Module:
        exported_experts = nn.ModuleList([e.export() for e in self.experts])
        exported_gate = self.gate.export()
        softmax_fn = nn.Softmax(dim=-1)

        return ExportedActorMoESymm(exported_experts, exported_gate, softmax_fn, self.ms_obs_dim)


class ActorCriticMoESymm(nn.Module):
    """Actor-critic with Mixture-of-Experts policy."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        is_dae: bool,
        amp_joint_names: list[str],
        joint_order_for_morphosymm: list[str],
        G: escnn.group.groups.cyclicgroup.CyclicGroup,
        obs_state_ratio: int = 1,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        num_experts: int = 4,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                (
                    "ActorCriticMoESymm.__init__ ignored unexpected arguments: "
                    + str(list(kwargs.keys()))
                )
            )
        super().__init__()

        self.amp_joint_names = amp_joint_names
        self.joint_order_for_morphosymm = joint_order_for_morphosymm
        self.G = G


        # Set up symmetry information
        # We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
        gspace = escnn.gspaces.no_base_space(G)
        # Get the relevant group representations.
        rep_Rd = G.representations['R3']
        rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
        rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
        rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
        rep_xy.name = "base_xy"
        rep_euler_xyz = G.representations['euler_xyz']
        rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
        rep_euler_z.name = "euler_z"

        # Define the input and output FieldTypes using the representations of each geometric object.
        # Representation of x := [q, v] ∈ Q_js x TqQ_js      =>    ρ_X_js(g) := ρ_Q_js(g) ⊕ ρ_TqQ_js(g)  | g ∈ G
        base_transition = [rep_Rd, rep_euler_xyz, rep_Rd, rep_TqQJ, rep_TqQJ, rep_TqQJ, rep_xy, rep_euler_z]
        if is_dae:
            latent_transition = [rep_Rd, rep_euler_xyz, rep_Rd, rep_TqQJ, rep_TqQJ, rep_TqQJ, rep_xy, rep_euler_z] * obs_state_ratio

        in_field_type = FieldType(gspace, base_transition)
        # Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
        out_field_type = FieldType(gspace, [rep_QJ])

        if is_dae:
            critic_in_field_type = FieldType(gspace, latent_transition)
        else:
            critic_in_field_type = FieldType(gspace, base_transition)

        self.gspace = gspace
        self.in_field_type = in_field_type
        self.out_field_type = out_field_type
        self.critic_in_field_type = critic_in_field_type

        # one dimensional field type for critic
        critic_out_field_type = FieldType(gspace, [G.trivial_representation])
        gating_out_field_type = FieldType(gspace, [G.trivial_representation] * num_experts)

        # Actor (Mixture-of-Experts)
        self.actor = ActorMoESymm(
            in_field_type=in_field_type,
            gating_out_field_type=gating_out_field_type,
            out_field_type=out_field_type,
            obs_dim=num_actor_obs,
            hidden_dims=actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=actor_hidden_dims[:-1],  # last layer is output
            activation=activation,
        )

        # Critic
        self.critic = SimpleEMLP(critic_in_field_type, critic_out_field_type,
            hidden_dims=critic_hidden_dims,
            activation=activation, actor=False)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor (MoE) structure:\n{self.actor}")
        print(f"Critic MLP structure:\n{self.critic}")

    def reset(self, dones=None):  # noqa: D401
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        observations = self.in_field_type(observations)
        mean_ms = self.actor(observations).tensor

        mean = ms_joints_to_isaaclab(mean_ms, self.joint_order_for_morphosymm, self.amp_joint_names)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:  # "log"
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # deterministic (mean) action
        observations = self.in_field_type(observations)
        return self.actor(observations).tensor

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self.critic_in_field_type(critic_observations)
        return self.critic(critic_observations)

    # unchanged load_state_dict so checkpoints from the old class still load
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

class SimpleEMLP(EquivariantModule):
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 hidden_dims = [256, 256, 256],
                 bias: bool = True,
                 actor: bool = True,
                 activation: str = "ReLU"):
        super().__init__()
        self.out_type = out_type
        gspace = in_type.gspace
        group = gspace.fibergroup

        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(len(hidden_dims)):
            layer_out_type = FieldType(gspace, [group.regular_representation] * int((hidden_dims[n] / group.order())))

            self.net.add_module(f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                             escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            self.net.add_module(f"act_{n}", self.get_activation(activation, layer_out_type))

            layer_in_type = layer_out_type

        if actor:
            self.net.add_module(f"linear_{len(hidden_dims)}: in={layer_in_type.size}-out={out_type.size}",
                                escnn.nn.Linear(layer_in_type, out_type, bias=bias))
            self.extra_layer = None
        else:
            num_inv_features = len(layer_in_type.irreps)
            self.extra_layer = torch.nn.Linear(num_inv_features, out_type.size, bias=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x= self.net(x)
        if self.extra_layer:
            x = self.extra_layer(x.tensor)
        return x

    @staticmethod
    def get_activation(activation: str, hidden_type: FieldType) -> EquivariantModule:
        if activation.lower() == "relu":
            return escnn.nn.ReLU(hidden_type)
        elif activation.lower() == "elu":
            return escnn.nn.ELU(hidden_type)
        elif activation.lower() == "lrelu":
            return escnn.nn.LeakyReLU(hidden_type)
        elif activation.lower() == "softmax":
            return Softmax(hidden_type)
        else:
            raise NotImplementedError

    def evaluate_output_shape(self, input_shape):
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        sequential = nn.Sequential()
        for name, module in self.net.named_children():
            sequential.add_module(name, module.export())
        return sequential

class Softmax(EquivariantModule):

    def __init__(self, in_type: FieldType):
        r"""

        Module that implements a pointwise Softmax to every channel independently.
        The input representation is preserved by this operation and, therefore, it equals the output
        representation.

        Only representations supporting pointwise non-linearities are accepted as input field type.

        Args:
            in_type (FieldType):  the input field type
            alpha (float): the :math:`\alpha` value for the ELU formulation. Default: 1.0
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

        """

        assert isinstance(in_type.gspace, GSpace)

        super(Softmax, self).__init__()

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type

        # the representation in input is preserved
        self.out_type = in_type

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies softmax function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after elu has been applied

        """
        assert input.type == self.in_type
        return GeometricTensor(
            F.softmax(input.tensor, dim=-1),
            self.out_type, input.coords
        )

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:

        c = self.in_type.size

        x = torch.randn(3, c, 10, 10)

        x = GeometricTensor(x, self.in_type)

        errors = []

        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))

            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())

            assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

            errors.append((el, errs.mean()))

        return errors

    def extra_repr(self):
        return 'type={}'.format(
            self.in_type
        )

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.ELU` module and set to "eval" mode.

        """

        self.eval()

        return torch.nn.ELU(alpha=self.alpha, inplace=self._inplace)