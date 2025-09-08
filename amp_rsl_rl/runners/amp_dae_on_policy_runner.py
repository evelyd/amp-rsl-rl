# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state

from amp_rsl_rl.utils import Normalizer
from amp_rsl_rl.utils import AMPLoader
from amp_rsl_rl.algorithms import AMP_PPO, AMP_PPO_DAE, AMP_PPO_DAE_Online
from amp_rsl_rl.networks import Discriminator, ActorCriticMoE, ActorCriticMoESymm, ActorCriticSymm
from amp_rsl_rl.utils import export_policy_as_onnx, fill_replay_buffer
from amp_rsl_rl.runners import AMPOnPolicyRunner
from amp_rsl_rl.dha_utils import compute_ms_observations_ideal, compute_ms_observations_dae, isaaclab_joints_to_ms

class AMPDAEOnPolicyRunner(AMPOnPolicyRunner):
    """
    This implementation of the AMP DAE on-policy runner is based on the AMP PPO algorithm. The only change from the regular AMP is that it uses the latent state from the DAE encoder as an extra critic obs.

    This runner works for EMLP, C-DAE, EC-DAE, and combinations of those. It won't work for the case without any of these, and for that it is necessary to use the AMPOnPolicyRunner directly.

    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", video_interval=None, video_length=None):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.task = train_cfg["experiment_name"]
        if "online" in self.task:
            self.koopman_cfg = train_cfg["koopman"]
        self.device = device
        self.env = env
        self.video_interval = video_interval
        self.video_length = video_length

        # Get the size of the observation space
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        # Initialize the PPO algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # AMP_PPO

        # Define the state representation
        # G is the symmetry group of the system
        from morpho_symm.utils.robot_utils import load_symmetric_system
        robot, G = load_symmetric_system(robot_name="ergocub")
        joint_order_for_morphosymm = robot.joint_space_names

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic

        # NOTE: to use this we need to configure the observations in the env coherently with amp observation. Tested with Manager Based envs in Isaaclab
        amp_joint_names = self.env.cfg.observations.amp.joint_pos.params['asset_cfg'].joint_names
        ms_joint_difference = len(joint_order_for_morphosymm) - len(amp_joint_names)
        is_ideal = False
        if num_obs == 90:
            if "online" in self.task:
                raise ValueError("The online DAE runner is not yet implemented for the ideal velocity task.")
            ms_critic_obs = num_critic_obs + 3 * ms_joint_difference
            dae_input_size = ms_critic_obs
            is_ideal = True
        elif num_obs == 357: # 357 = (3 + 3 + 14 + 14) * 10 + 14 + 2 + 1
            ms_critic_obs = num_critic_obs + (10 * 2 + 1) * ms_joint_difference
            dae_input_size = num_critic_obs - 9 * (3 + 3 + len(amp_joint_names) + len(amp_joint_names)) + 3 * ms_joint_difference

        if alg_class == AMP_PPO_DAE or alg_class == AMP_PPO_DAE_Online:
            is_dae = True
            # Scale the number of critic obs based on the DAE state to latent state ratio, accounting for the difference in joint space
            if "online" in self.task:
                obs_state_ratio = self.koopman_cfg['robot']["obs_state_ratio"]
            else:
                obs_state_ratio = self.cfg["obs_state_ratio"]

            if actor_critic_class == ActorCriticMoESymm or actor_critic_class == ActorCriticSymm:
                # if symm and dae, the critic obs is the morphosymm obs + the latent state
                ac_critic_obs = ms_critic_obs + dae_input_size * obs_state_ratio
            else:
                # if dae but not symm, the critic obs is the regular obs + the latent state
                ac_critic_obs = num_critic_obs + dae_input_size * obs_state_ratio
        else:
            obs_state_ratio = 1
            ac_critic_obs = ms_critic_obs
            is_dae = False

        if actor_critic_class == ActorCriticMoESymm or actor_critic_class == ActorCriticSymm:
            actor_critic: ActorCriticMoESymm | ActorCriticSymm = (
                actor_critic_class(
                    num_actor_obs=num_obs,
                    num_critic_obs=ac_critic_obs,
                    num_actions=self.env.num_actions,
                    is_dae=is_dae,
                    G=G,
                    obs_state_ratio=obs_state_ratio,
                    amp_joint_names=amp_joint_names,
                    joint_order_for_morphosymm=joint_order_for_morphosymm,
                    is_ideal=is_ideal,
                    **self.policy_cfg
                ).to(self.device)
            )
        else:
            actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
                actor_critic_class(
                    num_obs, ac_critic_obs, self.env.num_actions, **self.policy_cfg
                ).to(self.device)
            )

        delta_t = self.env.cfg.sim.dt * self.env.cfg.decimation

        # Initilize all the ingredients required for AMP (discriminator, dataset loader)
        num_amp_obs = extras["observations"]["amp"].shape[1]
        amp_data = AMPLoader(
            self.device,
            self.cfg["amp_data_path"],
            self.cfg["dataset_names"],
            self.cfg["dataset_weights"],
            delta_t,
            self.cfg["slow_down_factor"],
            amp_joint_names,
        )

        # self.env.unwrapped.scene["robot"].joint_names)

        # amp_data = AMPLoader(num_amp_obs, self.device)
        self.amp_normalizer = Normalizer(num_amp_obs, device=self.device)
        self.discriminator = Discriminator(
            num_amp_obs
            * 2,  # the discriminator takes in the concatenation of the current and next observation
            self.discriminator_cfg["hidden_dims"],
            self.discriminator_cfg["reward_scale"],
            device=self.device,
        ).to(self.device)

        # This removes from alg_cfg fields that are not in AMP_PPO but are introduced in rsl_rl 2.2.3 PPO
        # normalize_advantage_per_mini_batch=False,
        # rnd_cfg: dict | None = None,
        # symmetry_cfg: dict | None = None,
        # multi_gpu_cfg: dict | None = None,
        for key in list(self.alg_cfg.keys()):
            if key not in alg_class.__init__.__code__.co_varnames:
                self.alg_cfg.pop(key)
        input(f"Using algorithm: {alg_class.__name__}, Policy: {actor_critic.__class__.__name__}, Task: {self.task}")
        if alg_class == AMP_PPO_DAE_Online:
            self.alg: AMP_PPO_DAE_Online = alg_class(
                koopman_cfg=self.koopman_cfg,
                task=self.task,
                dt=delta_t,
                actor_critic=actor_critic,
                discriminator=self.discriminator,
                amp_data=amp_data,
                amp_normalizer=self.amp_normalizer,
                device=self.device,
                G=G,
                is_ideal=is_ideal,
                amp_joint_names=amp_joint_names,
                joint_order_for_morphosymm=joint_order_for_morphosymm,
                ms_critic_obs=ms_critic_obs,
                dae_input_size=dae_input_size,
                **self.alg_cfg,
            )
        elif alg_class == AMP_PPO_DAE:
            dae_model_path = self.cfg["model_path"]
            self.alg: AMP_PPO_DAE = alg_class(
                model_path=dae_model_path,
                actor_critic=actor_critic,
                discriminator=self.discriminator,
                amp_data=amp_data,
                amp_normalizer=self.amp_normalizer,
                device=self.device,
                G=G,
                is_ideal=is_ideal,
                amp_joint_names=amp_joint_names,
                joint_order_for_morphosymm=joint_order_for_morphosymm,
                **self.alg_cfg,
            )
        else:
            self.alg: AMP_PPO = alg_class(
            actor_critic=actor_critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            is_ideal=is_ideal,
            amp_joint_names=amp_joint_names,
            joint_order_for_morphosymm=joint_order_for_morphosymm,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=[num_critic_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Initialize the replay buffer
        if "online" in self.task:
            if hasattr(self.alg, 'replay_buffer'):
                fill_replay_buffer(self.alg, self.env, self.obs_normalizer, self.critic_obs_normalizer)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        input(f"alg class: {alg_class}, actor critic class: {actor_critic_class}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                import wandb

                # Update the run name with a sequence number. This function is useful to
                # replicate the same behaviour of rsl-rl-lib before v2.3.0
                def update_run_name_with_sequence(prefix: str) -> None:
                    # Retrieve the current wandb run details (project and entity)
                    project = wandb.run.project
                    entity = wandb.run.entity

                    # Use wandb's API to list all runs in your project
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project}")

                    max_num = 0
                    # Iterate through runs to extract the numeric suffix after the prefix.
                    for run in runs:
                        if run.name.startswith(prefix):
                            # Extract the numeric part from the run name.
                            numeric_suffix = run.name[
                                len(prefix) :
                            ]  # e.g., from "prefix564", get "564"
                            try:
                                run_num = int(numeric_suffix)
                                if run_num > max_num:
                                    max_num = run_num
                            except ValueError:
                                continue

                    # Increment to get the new run number
                    new_num = max_num + 1
                    new_run_name = f"{prefix}{new_num}"

                    # Update the wandb run's name
                    wandb.run.name = new_run_name
                    print("Updated run name to:", wandb.run.name)

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                update_run_name_with_sequence(prefix=self.cfg["wandb_project"])

                wandb.gym.monitor()
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )

                video_folder = os.path.join(self.log_dir, "videos", "train")
                logged_videos = []

            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        amp_obs = extras["observations"]["amp"]
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Set DAE to train mode also
        if "online" in self.task:
            self.alg.dae_model.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout

            mean_style_reward_log = 0
            mean_task_reward_log = 0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    if "online" in self.task:
                        # Collect data for DAE
                        if self.alg.is_ideal:
                            current_states_for_dae = compute_ms_observations_ideal(critic_obs, self.alg.joint_order_for_morphosymm, self.alg.amp_joint_names)
                        else:
                            current_states_for_dae = compute_ms_observations_dae(critic_obs, self.alg.joint_order_for_morphosymm, self.alg.amp_joint_names)
                        current_actions_for_dae = actions.clone()

                    self.alg.act_amp(amp_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    _, extras = self.env.get_observations()
                    next_amp_obs = extras["observations"]["amp"]
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(
                            infos["observations"]["critic"]
                        )
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    next_amp_obs = next_amp_obs.to(self.device)

                    # Process the AMP reward
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs, normalizer=self.amp_normalizer
                    )

                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # Combine the task and style rewards (TODO this can be a hyperparameters)
                    rewards = 0.5 * rewards + 0.5 * style_rewards

                    self.alg.process_env_step(rewards, dones, infos)
                    self.alg.process_amp_step(next_amp_obs)

                    if "online" in self.task:
                        # Get the next states for the DAE
                        if self.alg.is_ideal:
                            next_states_for_dae = compute_ms_observations_ideal(critic_obs, self.alg.joint_order_for_morphosymm, self.alg.amp_joint_names)
                        else:
                            next_states_for_dae = compute_ms_observations_dae(critic_obs, self.alg.joint_order_for_morphosymm, self.alg.amp_joint_names)

                        current_actions_for_dae = isaaclab_joints_to_ms(current_actions_for_dae, self.alg.joint_order_for_morphosymm, self.alg.amp_joint_names)

                        # Fill the PER buffer
                        self.alg.replay_buffer.insert(
                            current_states_for_dae.cpu(),
                            current_actions_for_dae.cpu(),
                            next_states_for_dae.cpu()
                        )

                    # The next observation becomes the current observation for the next step
                    amp_obs = torch.clone(next_amp_obs)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            if "online" in self.task:
                # Perform DAE training step
                mean_dae_loss = 0.0
                dae_train_time = 0.0
                mean_dae_obs_pred_loss = 0.0
                mean_dae_state_rec_loss = 0.0
                mean_dae_state_pred_loss = 0.0

                if len(self.alg.replay_buffer) >= self.koopman_cfg["model"]["mini_batch_size"]:
                    dae_training_start_time = time.time()

                    dae_num_mini_batches = self.koopman_cfg["model"]["num_mini_batches"]
                    dae_mini_batch_size = self.koopman_cfg["model"]["mini_batch_size"]

                    dae_losses_this_iter = []
                    dae_obs_pred_losses_this_iter = []
                    dae_state_rec_losses_this_iter = []
                    dae_state_pred_losses_this_iter = []

                    # Anneal beta for Importance Sampling weights
                    current_beta = self.alg.replay_buffer.beta_initial + (1.0 - self.alg.replay_buffer.beta_initial) * \
                                min(1.0, (it - self.current_learning_iteration) / self.alg.replay_buffer.beta_annealing_steps)


                    # Iterating over mini-batches for DAE training
                    for _ in range(dae_num_mini_batches):
                        # Sample from the Prioritized Replay Buffer
                        batch_states_raw, batch_actions_raw, batch_next_states_raw, batch_tree_indices, is_weights = \
                            self.alg.replay_buffer.sample(dae_mini_batch_size, current_beta)

                        # Transfer is_weights to cuda device
                        is_weights = is_weights.to(self.device)

                        sample_generator = self.alg.replay_buffer.preprocess_samples(
                            batch_states_raw, batch_actions_raw, batch_next_states_raw,
                            frames_per_step=self.koopman_cfg["robot"]["frames_per_state"],
                            prediction_horizon=self.koopman_cfg["robot"]["pred_horizon"]
                        )

                        all_state_observations = []
                        all_action_observations = []
                        all_next_state_observations = []
                        for sample in sample_generator:
                            all_state_observations.append(sample["state_observations"].unsqueeze(0))
                            all_action_observations.append(sample["action_observations"].unsqueeze(0))
                            all_next_state_observations.append(sample["next_state_observations"].unsqueeze(0))

                        # If preprocess_samples yielded no valid samples (e.g., traj too short), skip this mini-batch
                        if not all_state_observations:
                            print("Warning: No valid samples generated by preprocess_samples, skipping DAE mini-batch.")
                            continue

                        combined_state_observations = torch.cat(all_state_observations, dim=0).to(self.device)
                        combined_action_observations = torch.cat(all_action_observations, dim=0).to(self.device)
                        combined_next_state_observations = torch.cat(all_next_state_observations, dim=0).to(self.device)

                        # Normalize the observations and actions
                        normed_states, normed_actions = self.alg.obs_action_normalizer.normalize(
                            combined_state_observations, combined_action_observations
                        )
                        next_normed_states = self.alg.obs_action_normalizer.normalize(
                            combined_next_state_observations
                        )

                        # Move preprocessed and normalized batch to the correct device for the DAE model
                        batch = self.alg.replay_buffer.shape_states_actions(
                                normed_states, normed_actions, next_normed_states
                        )

                        batch_on_device = {k: v.to(self.device) for k, v in batch.items()}

                        # Forward pass through DAE
                        if hasattr(self.alg.dae_model, 'action_dim') and self.alg.dae_model.action_dim > 0:
                            outputs = self.alg.dae_model(**batch_on_device)
                        else:
                            outputs = self.alg.dae_model(**batch_on_device)

                        # Compute DAE losses
                        dae_loss_per_sample, dae_metrics = self.alg.dae_model.compute_loss_and_metrics(**outputs, **batch_on_device)

                        # Apply importance sampling weights to the loss
                        actual_batch_size_for_loss = dae_loss_per_sample.shape[0]
                        if is_weights.shape[0] != actual_batch_size_for_loss:
                            is_weights_aligned = is_weights[:actual_batch_size_for_loss]
                        else:
                            is_weights_aligned = is_weights

                        weighted_dae_loss = (dae_loss_per_sample * is_weights_aligned).mean()

                        # Backpropagate and update DAE weights
                        self.alg.dae_optimizer.zero_grad()
                        weighted_dae_loss.backward()
                        self.alg.dae_optimizer.step()

                        # Update priorities in the replay buffer
                        # Use the per-sample losses as errors
                        dae_errors_for_priority_update = dae_loss_per_sample.detach().cpu().numpy()

                        # Ensure that the batch_tree_indices also aligns with the number of samples that actually generated a loss.
                        if len(batch_tree_indices) != actual_batch_size_for_loss:
                            batch_tree_indices_aligned = batch_tree_indices[:actual_batch_size_for_loss]
                        else:
                            batch_tree_indices_aligned = batch_tree_indices

                        # Prioritize samples based on the overall DAE loss
                        self.alg.replay_buffer.update_priorities(
                            batch_tree_indices_aligned,
                            dae_errors_for_priority_update
                        )

                        dae_losses_this_iter.append(weighted_dae_loss.item())
                        dae_obs_pred_losses_this_iter.append(dae_metrics["obs_pred_loss"].item())
                        dae_state_rec_losses_this_iter.append(dae_metrics["state_rec_loss"].item())
                        dae_state_pred_losses_this_iter.append(dae_metrics["state_pred_loss"].item())

                    if dae_losses_this_iter:
                        mean_dae_loss = sum(dae_losses_this_iter) / len(dae_losses_this_iter)
                        mean_dae_obs_pred_loss = sum(dae_obs_pred_losses_this_iter) / len(dae_obs_pred_losses_this_iter)
                        mean_dae_state_rec_loss = sum(dae_state_rec_losses_this_iter) / len(dae_state_rec_losses_this_iter)
                        mean_dae_state_pred_loss = sum(dae_state_pred_losses_this_iter) / len(dae_state_pred_losses_this_iter)
                    else:
                        mean_dae_loss = 0.0 # No batches trained
                        mean_dae_obs_pred_loss = 0.0
                        mean_dae_state_rec_loss = 0.0
                        mean_dae_state_pred_loss = 0.0

                    dae_train_time = time.time() - dae_training_start_time

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env
            mean_total_reward_log = mean_style_reward_log + mean_task_reward_log

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_kl_divergence,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                locs = locals()
                if "online" in self.task:
                    locs["mean_dae_loss"] = mean_dae_loss
                    locs["mean_dae_obs_pred_loss"] = mean_dae_obs_pred_loss
                    locs["mean_dae_state_rec_loss"] = mean_dae_state_rec_loss
                    locs["mean_dae_state_pred_loss"] = mean_dae_state_pred_loss
                    locs["dae_train_time"] = dae_train_time
                self.log(locs)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)
                if "online" in self.task:
                    torch.save(self.alg.dae_model.state_dict(), os.path.join(self.log_dir, 'dae_model_{}.pt'.format(it)))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

            if self.logger_type == "wandb":
                # List all video files that end with .mp4
                video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

                # Log any new video files that haven't been logged yet
                for video_file in video_files:
                    video_path = os.path.join(video_folder, video_file)
                    if video_path not in logged_videos:
                        print(f"Found new video: {video_path}. Logging to wandb...")

                        # Create a wandb.Video object from the file
                        wandb_video = wandb.Video(video_path, fps=30, format="mp4")

                        # Log the video to W&B
                        wandb.log({"training_video": wandb_video}, step=self.current_learning_iteration)

                        # Add the video to our list of logged videos
                        logged_videos.append(video_path)

        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
            save_onnx=True,
        )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )

        # Adding logging due to AMP
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar(
            "Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"]
        )

        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar(
            "Loss/mean_kl_divergence", locs["mean_kl_divergence"], locs["it"]
        )
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if "online" in self.task:
                self.writer.add_scalar("DAE/loss", locs["mean_dae_loss"], locs["it"]) # or self.current_learning_iteration
                self.writer.add_scalar("DAE/obs_pred_loss", locs["mean_dae_obs_pred_loss"], locs["it"])
                self.writer.add_scalar("DAE/state_rec_loss", locs["mean_dae_state_rec_loss"], locs["it"])
                self.writer.add_scalar("DAE/state_pred_loss", locs["mean_dae_state_pred_loss"], locs["it"])
                self.writer.add_scalar("DAE/train_time", locs["dae_train_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"]
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # make the eta in H:M:S
        eta_seconds = (
            self.tot_time
            / (locs["it"] + 1)
            * (locs["num_learning_iterations"] - locs["it"])
        )

        # Convert seconds to H:M:S
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None, save_onnx=False):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = (
                self.critic_obs_normalizer.state_dict()
            )
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            # Save the model in ONNX format
            # extract the folder path
            onnx_folder = os.path.dirname(path)

            # extract the iteration number from the path. The path is expected to be in the format
            # model_{iteration}.pt
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"

            export_policy_as_onnx(
                self.alg.actor_critic,
                normalizer=self.obs_normalizer,
                path=onnx_folder,
                filename=onnx_model_name,
            )

            if self.logger_type in ["neptune", "wandb"]:
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(self, path, load_optimizer=True, weights_only=False):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]

        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(
                loaded_dict["critic_obs_norm_state_dict"]
            )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(
                self.obs_normalizer(x)
            )  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
