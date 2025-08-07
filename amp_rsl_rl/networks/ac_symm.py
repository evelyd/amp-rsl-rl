# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import escnn
from escnn.nn import FieldType, EquivariantModule, GeometricTensor
from morpho_symm.utils.robot_utils import group_rep_from_gens

import torch
import torch.nn as nn
from torch.distributions import Normal

from amp_rsl_rl.dha_utils import ms_joints_to_isaaclab

class ActorCriticSymm(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        amp_joint_names: list[str],
                        joint_order_for_morphosymm: list[str],
                        G: escnn.group.groups.cyclicgroup.CyclicGroup,
                        is_dae: bool = False,
                        is_ideal: bool = False,
                        obs_state_ratio: int = 1,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        noise_std_type: str = "scalar",
                        **kwargs):

        super().__init__()

        self.amp_joint_names = amp_joint_names
        self.joint_order_for_morphosymm = joint_order_for_morphosymm
        self.G = G
        self.is_ideal = is_ideal

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
        # for push door task
        if self.is_ideal:
            base_transition = [rep_Rd, rep_euler_xyz, rep_Rd, rep_TqQJ, rep_TqQJ, rep_TqQJ, rep_xy, rep_euler_z]
            if is_dae:
                latent_transition = [rep_Rd, rep_euler_xyz, rep_Rd, rep_TqQJ, rep_TqQJ, rep_TqQJ, rep_xy, rep_euler_z] * obs_state_ratio
        else:
            base_transition = [rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz,
                               rep_euler_xyz, #10 * rep_euler_xyz,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,
                               rep_Rd,  # 10 * rep_Rd
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,  # 10 * rep_TqQJ
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,
                               rep_TqQJ,  # 10 * rep_TqQJ
                               rep_TqQJ,
                               rep_xy, rep_euler_z]
            if is_dae:
                latent_transition = [rep_euler_xyz, rep_Rd, rep_TqQJ, rep_TqQJ, rep_TqQJ, rep_xy, rep_euler_z] * obs_state_ratio

        in_field_type = FieldType(gspace, base_transition)
        # Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
        out_field_type = FieldType(gspace, [rep_QJ])

        if is_dae:
            critic_in_field_type = FieldType(gspace, base_transition + latent_transition)
        else:
            critic_in_field_type = FieldType(gspace, base_transition)
        print(f"critic in field type: {critic_in_field_type}")

        self.gspace = gspace
        self.in_field_type = in_field_type
        self.out_field_type = out_field_type
        self.critic_in_field_type = critic_in_field_type

        # one dimensional field type for critic
        critic_out_field_type = FieldType(gspace, [G.trivial_representation])

        # Construct the equivariant MLP

        self.actor = SimpleEMLP(in_field_type, out_field_type,
            hidden_dims = actor_hidden_dims,
            activation = activation,)

        self.critic = SimpleEMLP(critic_in_field_type, critic_out_field_type,
            hidden_dims = critic_hidden_dims,
            activation=activation,
            actor=False)   # Should set False！

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print("FieldType size:", self.in_field_type.size)

        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Actor #Params: ", params)
        model_parameters = filter(lambda p: p.requires_grad, self.critic.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Critic #Params: ", params)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
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

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        observations = self.in_field_type(observations)
        actions_mean = self.actor(observations).tensor
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self.critic_in_field_type(critic_observations)
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

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