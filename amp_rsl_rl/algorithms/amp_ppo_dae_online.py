# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import os

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

import escnn

# External modules providing the actor-critic model, storage utilities, and AMP components.
from rsl_rl.modules import ActorCritic

from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.utils import AMPLoader
from amp_rsl_rl.algorithms.amp_ppo import AMP_PPO
from rsl_rl.storage import RolloutStorage # for storing Koopman DAE data

from amp_rsl_rl.dha_utils import compute_ms_observations, compute_ms_observations_ideal, compute_ms_observations_dae, initialize_dae_model, isaaclab_joints_to_ms

from amp_rsl_rl.storage import RunningStdScaler, PrioritizedReplayBuffer

import amp_ergocub
from escnn.nn import GeometricTensor


class AMP_PPO_DAE_Online(AMP_PPO):
    """AMP PPO with DAE (Dynamics Autoencoder) critic information."""

    def __init__(
        self,
        koopman_cfg,
        task: str,
        dt: float,
        actor_critic: ActorCritic,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        amp_normalizer: Optional[Any],
        G: escnn.group.groups.cyclicgroup.CyclicGroup,
        amp_joint_names: list[str],
        joint_order_for_morphosymm: list[str],
        ms_critic_obs: int,
        dae_input_size: int,
        is_ideal: bool = False,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        amp_replay_buffer_size: int = 100000,
        use_smooth_ratio_clipping: bool = False,
        device: str = "cpu",
        replay_buffer_size: int = 100000,
    ) -> None:

        super().__init__(
            actor_critic,
            discriminator,
            amp_data,
            amp_normalizer,
            amp_joint_names,
            joint_order_for_morphosymm,
            is_ideal,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            amp_replay_buffer_size,
            use_smooth_ratio_clipping,
            device
        )

        self.koopman_transition = RolloutStorage.Transition()
        self.replay_buffer = PrioritizedReplayBuffer(dae_input_size, koopman_cfg["robot"]["action_dim"], koopman_cfg["model"]["beta_initial"], koopman_cfg["model"]["beta_annealing_steps"], replay_buffer_size) #default device will be cpu
        self.obs_action_normalizer = RunningStdScaler(dae_input_size, koopman_cfg["robot"]["action_dim"], device=self.device)
        self.state_dim = koopman_cfg["robot"]["state_dim"]

        self.amp_joint_names = amp_joint_names
        self.joint_order_for_morphosymm = joint_order_for_morphosymm

        # DAE specific initialization
        # Initialize DAE model
        self.task = task
        self.dae_model = initialize_dae_model(
            cfg=koopman_cfg,
            task=self.task,
            G=G,
            dt=dt,
            device=self.device
            )

        self.is_ideal = is_ideal

        # Set the DAE optimizer
        self.dae_optimizer = torch.optim.Adam(self.dae_model.parameters(), lr=koopman_cfg["robot"]["lr"])

    def act_koopman(self, koopman_obs, koopman_actions):
        if self.is_ideal:
            self.koopman_transition.observations = compute_ms_observations_ideal(koopman_obs, self.joint_order_for_morphosymm, self.amp_joint_names)
        else:
            self.koopman_transition.observations = compute_ms_observations_dae(koopman_obs, self.joint_order_for_morphosymm, self.amp_joint_names)
        self.koopman_transition.actions = isaaclab_joints_to_ms(koopman_actions, self.joint_order_for_morphosymm, self.amp_joint_names)

    def process_koopman_step(self, koopman_obs):
        device = self.replay_buffer.device
        if self.is_ideal:
            next_obs = compute_ms_observations_ideal(koopman_obs, self.joint_order_for_morphosymm, self.amp_joint_names)
        else:
            next_obs = compute_ms_observations_dae(koopman_obs, self.joint_order_for_morphosymm, self.amp_joint_names)
        self.replay_buffer.insert(self.koopman_transition.observations.to(device), self.koopman_transition.actions.to(device), next_obs.to(device)) # take only the most recent obs
        self.koopman_transition.clear()

    def get_critic_input(self, critic_obs):
        """Processes critic_obs through DAE to get augmented input for the critic."""
        # Make sure that the joint measurements are converted to morphosymm
        if self.is_ideal:
            dae_input = compute_ms_observations_ideal(critic_obs, self.joint_order_for_morphosymm, self.amp_joint_names)
        else:
            dae_input = compute_ms_observations_dae(critic_obs, self.joint_order_for_morphosymm, self.amp_joint_names)

        dae_input = dae_input.to(dtype=next(self.dae_model.parameters()).dtype)
        dae_input = dae_input.to(device=next(self.dae_model.parameters()).device)

        dae_input_normed = self.obs_action_normalizer.normalize_states(dae_input)

        # Wrap as GeometricTensor for E-DAE/EC-DAE
        if "edae" in self.task or "ecdae" in self.task:
            dae_input_normed = GeometricTensor(dae_input_normed, self.dae_model.obs_fn.in_type)
            latent = self.dae_model.obs_fn(dae_input_normed).tensor.detach()
        else:
            latent = self.dae_model.obs_fn(dae_input_normed).detach()

        if "Symm" in type(self.actor_critic).__name__:
            return torch.cat((compute_ms_observations(critic_obs, self.joint_order_for_morphosymm, self.amp_joint_names), latent), dim=-1)
        else:
            return torch.cat((critic_obs, latent), dim=-1)

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """
        Selects an action based on the current observation and critic observation.
        It also records the necessary data (actions, log probabilities, values, etc.) in a transition.

        Parameters
        ----------
        obs : torch.Tensor
            Observation used by the actor network.
        critic_obs : torch.Tensor
            Observation used by the critic network for value estimation.

        Returns
        -------
        torch.Tensor
            The selected actions.
        """
        # Override to use DAE-augmented critic input
        critic_input = self.get_critic_input(critic_obs)
        # If using a recurrent network, retrieve the hidden states.
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute actions and related statistics, ensuring we detach tensors to avoid gradient issues.
        if "Symm" in type(self.actor_critic).__name__:
            if self.is_ideal:
                    self.transition.actions = self.actor_critic.act(compute_ms_observations_ideal(obs, self.joint_order_for_morphosymm, self.amp_joint_names)).detach()
            else:
                self.transition.actions = self.actor_critic.act(compute_ms_observations(obs, self.joint_order_for_morphosymm, self.amp_joint_names)).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_input).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # Record the observations before taking an environment step.
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """
        Computes the discounted returns and advantages based on the critic's evaluation of the last observation.

        Parameters
        ----------
        last_critic_obs : torch.Tensor
            The critic observation after the last environment step.
        """
        critic_input = self.get_critic_input(last_critic_obs)
        last_values = self.actor_critic.evaluate(critic_input).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Performs a single update step for both the actor-critic (PPO) and the AMP discriminator.
        It iterates over mini-batches of data, computes surrogate, value, AMP and gradient penalty losses,
        performs adaptive learning rate scheduling (if enabled), and updates model parameters.

        Returns
        -------
        tuple
            A tuple containing mean losses and statistics:
            (mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss,
             mean_policy_pred, mean_expert_pred, mean_accuracy_policy, mean_accuracy_expert)
        """
        # Initialize mean loss and accuracy statistics.
        mean_value_loss: float = 0.0
        mean_surrogate_loss: float = 0.0
        mean_amp_loss: float = 0.0
        mean_grad_pen_loss: float = 0.0
        mean_policy_pred: float = 0.0
        mean_expert_pred: float = 0.0
        mean_accuracy_policy: float = 0.0
        mean_accuracy_expert: float = 0.0
        mean_accuracy_policy_elem: float = 0.0
        mean_accuracy_expert_elem: float = 0.0
        mean_kl_divergence: float = 0.0

        # Create data generators for mini-batch sampling.
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # Generator for policy-generated AMP transitions.
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size=self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
            allow_replacement=True,
        )

        # Generator for expert AMP data.
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )

        # Loop over mini-batches from the environment transitions and AMP data.
        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            # Unpack the mini-batch sample from the environment.
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample

            # Forward pass through the actor to get current policy outputs.
            #ideal is regardless of dae, emlp setting. if emlp, emlp+ecdae: use obs ms. if cdae, use raw obs
            if "Symm" in type(self.actor_critic).__name__:
                if self.is_ideal:
                    obs_batch_ms = compute_ms_observations_ideal(obs_batch, self.joint_order_for_morphosymm, self.amp_joint_names)
                else:
                    obs_batch_ms = compute_ms_observations(obs_batch, self.joint_order_for_morphosymm, self.amp_joint_names)
                self.actor_critic.act(
                obs_batch_ms, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            else:
                self.actor_critic.act(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            # Augment critic_obs_batch with DAE latent state
            critic_input_batch = self.get_critic_input(critic_obs_batch)
            value_batch = self.actor_critic.evaluate(
                critic_input_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive learning rate adjustment based on KL divergence if schedule is "adaptive".
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    mean_kl_divergence += kl_mean.item()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Compute the PPO surrogate loss.
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )

            min_ = 1.0 - self.clip_param
            max_ = 1.0 + self.clip_param
            # Smooth clipping for the ratio if enabled.
            if self.use_smooth_ratio_clipping:
                clipped_ratio = (
                    1
                    / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
                    * (max_ - min_)
                    + min_
                )
            else:
                clipped_ratio = torch.clamp(ratio, min_, max_)

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Compute the value function loss.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Combine surrogate loss, value loss and entropy regularization to form PPO loss.
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Process AMP loss by unpacking policy and expert AMP samples.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            # Normalize AMP observations if a normalizer is provided.
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize(policy_state)
                    policy_next_state = self.amp_normalizer.normalize(policy_next_state)
                    expert_state = self.amp_normalizer.normalize(expert_state)
                    expert_next_state = self.amp_normalizer.normalize(expert_next_state)

            # Concatenate policy and expert AMP observations for the discriminator input.
            B_pol = policy_state.size(0)
            discriminator_input = torch.cat(
                (
                    torch.cat([policy_state, policy_next_state], dim=-1),
                    torch.cat([expert_state, expert_next_state], dim=-1),
                ),
                dim=0,
            )
            discriminator_output = self.discriminator(discriminator_input)
            policy_d, expert_d = (
                discriminator_output[:B_pol],
                discriminator_output[B_pol:],
            )

            # Compute discriminator losses for expert and policy data.
            expert_loss = self.discriminator_expert_loss(expert_d)
            policy_loss = self.discriminator_policy_loss(policy_d)

            # AMP loss is the average of expert and policy losses.
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # Compute gradient penalty to stabilize discriminator training.
            grad_pen_loss = self.discriminator.compute_grad_pen(
                *sample_amp_expert, lambda_=10
            )

            # The final loss combines the PPO loss with AMP losses.
            loss = ppo_loss + (amp_loss + grad_pen_loss)

            # Backpropagation and optimizer step.
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update the normalizer with current policy and expert AMP observations.
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state)
                self.amp_normalizer.update(expert_state)

            # Compute probabilities from the discriminator logits.
            policy_d_prob = torch.sigmoid(policy_d)
            expert_d_prob = torch.sigmoid(expert_d)

            # Update running statistics.
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d_prob.mean().item()
            mean_expert_pred += expert_d_prob.mean().item()

            # Calculate the accuracy of the discriminator.
            mean_accuracy_policy += torch.sum(
                torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
            ).item()
            mean_accuracy_expert += torch.sum(
                torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
            ).item()

            # Record the total number of elements processed.
            mean_accuracy_expert_elem += expert_d_prob.numel()
            mean_accuracy_policy_elem += policy_d_prob.numel()

        # Average the statistics over all mini-batches.
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_accuracy_policy /= mean_accuracy_policy_elem
        mean_accuracy_expert /= mean_accuracy_expert_elem
        mean_kl_divergence /= num_updates

        # Clear the storage for the next update cycle.
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
            mean_kl_divergence,
        )