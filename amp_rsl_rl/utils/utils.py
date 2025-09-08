# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple, Union

import torch


class RunningMeanStd:
    """
    Calculates the running mean and standard deviation of a data stream.
    Based on the parallel algorithm for calculating variance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        epsilon (float): Small constant to initialize the count for numerical stability.
        shape (Tuple[int, ...]): Shape of the data (e.g., observation shape).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        """
        Updates the running statistics using a new batch of data.

        Args:
            arr (torch.Tensor): Batch of data (batch_size, *shape).
        """
        batch = arr.to(self.device, dtype=torch.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            batch.shape[0], dtype=torch.float32, device=self.device
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        """
        Updates statistics using precomputed batch mean, variance, and count.

        Args:
            batch_mean (torch.Tensor): Mean of the batch.
            batch_var (torch.Tensor): Variance of the batch.
            batch_count (torch.Tensor): Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)


class Normalizer(RunningMeanStd):
    """
    A normalizer that uses running statistics to normalize inputs, with optional clipping.

    Args:
        input_dim (Tuple[int, ...]): Shape of the input observations.
        epsilon (float): Small constant added to variance to avoid division by zero.
        clip_obs (float): Maximum absolute value to clip the normalized observations.
    """

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        shape = (input_dim,) if isinstance(input_dim, int) else tuple(input_dim)
        super().__init__(epsilon=epsilon, shape=shape, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input using running mean and std, and clips the result.

        Args:
            input (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized and clipped tensor.
        """
        x = input.to(self.device, dtype=torch.float32)
        std = (self.var + self.epsilon).sqrt()
        y = (x - self.mean) / std
        return torch.clamp(y, -self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def update_normalizer(self, rollouts, expert_loader) -> None:
        """
        Updates running statistics using samples from both policy and expert trajectories.

        Args:
            rollouts: Object with method `feed_forward_generator_amp(...)`.
            expert_loader: Dataloader or similar object providing expert batches.
        """
        policy_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size
        )
        expert_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size
        )

        for expert_batch, policy_batch in zip(expert_generator, policy_generator):
            batch = torch.cat((*expert_batch, *policy_batch), dim=0)
            self.update(batch)

def fill_replay_buffer(algorithm_instance, env_instance, obs_normalizer, critic_obs_normalizer, num_initial_steps=None):
    """
    Initializes the replay buffers within the algorithm by performing dummy rollouts,
    mimicking the regular training loop's data collection process.
    This version correctly handles rsl_rl's RolloutStorage by performing full rollouts
    and clearing storage periodically.

    Args:
        algorithm_instance: PPO or similar
        env_instance: env
        state_dim: Dim of a single state observation
        num_initial_steps: The total number of environment steps to take for pre-filling
    """

    # Determine num_initial_rollouts based on num_initial_steps
    if num_initial_steps is None:
        if hasattr(algorithm_instance, 'replay_buffer') and hasattr(algorithm_instance.replay_buffer.states, 'shape'):
            required_steps_for_koopman = algorithm_instance.replay_buffer.states.shape[0]
        else:
            required_steps_for_koopman = 10000 # Fallback if buffer info not available
            print("Warning: Could not determine Koopman buffer size. Defaulting to 10000 steps.")

        steps_per_full_rollout = env_instance.num_envs * algorithm_instance.storage.num_transitions_per_env
        if steps_per_full_rollout == 0: # Avoid division by zero if not configured
            steps_per_full_rollout = 1 # Dummy value, will lead to few rollouts
            print("Warning: steps_per_full_rollout is zero, likely due to num_envs or num_steps_per_env. Check config.")


        num_initial_rollouts = max(1, (required_steps_for_koopman + steps_per_full_rollout - 1) // steps_per_full_rollout)
        print(f"No num_initial_steps provided. Will perform {num_initial_rollouts} full rollouts to fill Koopman buffer.")
    else:
        # If num_initial_steps is provided, convert it to rollouts
        steps_per_full_rollout = env_instance.num_envs * algorithm_instance.num_steps_per_env
        if steps_per_full_rollout == 0:
            steps_per_full_rollout = 1
            print("Warning: steps_per_full_rollout is zero, likely due to num_envs or num_steps_per_env. Check config.")

        num_initial_rollouts = max(1, (num_initial_steps + steps_per_full_rollout - 1) // steps_per_full_rollout)
        print(f"num_initial_steps ({num_initial_steps}) will result in {num_initial_rollouts} full rollouts.")


    print(f"Initializing replay buffers by performing {num_initial_rollouts} full rollouts...")

    # Set algorithm's actor_critic to evaluation mode during initialization
    if hasattr(algorithm_instance.actor_critic, 'eval'):
        algorithm_instance.actor_critic.eval()

    # Reset environment to get initial observations for the very first rollout
    obs, extras = env_instance.get_observations()
    critic_obs = extras["observations"].get("critic", obs)
    obs, critic_obs = obs.to(algorithm_instance.device), critic_obs.to(algorithm_instance.device)

    # Ensure koopman_transition is initialized and clear if needed
    if not hasattr(algorithm_instance, 'koopman_transition'):
        raise AttributeError("Algorithm instance missing 'koopman_transition' attribute.")
    algorithm_instance.koopman_transition.clear()

    if hasattr(algorithm_instance, 'storage'):
        algorithm_instance.storage.observations[0].copy_(obs)
        if hasattr(algorithm_instance.storage, 'critic_observations') and extras["observations"]["critic"] is not None:
             algorithm_instance.storage.critic_observations[0].copy_(critic_obs)
    else:
        print("Warning: Algorithm instance does not have 'storage' attribute. Ensure your PPO handles initial observation internally.")

    for rollout_idx in range(num_initial_rollouts):
        for i in range(algorithm_instance.storage.num_transitions_per_env):
            actions = algorithm_instance.act(obs, critic_obs)
            algorithm_instance.act_koopman(obs, actions)
            obs, rewards, dones, infos = env_instance.step(actions)
            _, extras = env_instance.get_observations()
            obs = obs_normalizer(obs)
            if "critic" in infos["observations"]:
                critic_obs = critic_obs_normalizer(
                    infos["observations"]["critic"]
                )
            else:
                critic_obs = obs
            obs, critic_obs, rewards, dones = (
                obs.to(algorithm_instance.device),
                critic_obs.to(algorithm_instance.device),
                rewards.to(algorithm_instance.device),
                dones.to(algorithm_instance.device),
            )
            algorithm_instance.process_env_step(rewards, dones, infos)

            algorithm_instance.process_koopman_step(obs)

        algorithm_instance.compute_returns(critic_obs.clone().detach())
        algorithm_instance.storage.clear()

    print("Replay buffer initialization complete.")

    # Switch back to train mode
    if hasattr(algorithm_instance.actor_critic, 'train'):
        algorithm_instance.actor_critic.train()
