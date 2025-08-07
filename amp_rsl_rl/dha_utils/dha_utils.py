# This version is modified for your case:
# - 35-dimensional observation
# - DAE-aug model (no action used)
# - Not using IMU task

import torch
import os
import re
import numpy as np
import dha
from dha.utils.mysc import class_from_name
from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.ControlledDynamicsAutoEncoder import ControlledDAE
from dha.nn.ControlledEquivDynamicsAutoencoder import ControlledEquivDAE
from morpho_symm.utils.robot_utils import load_symmetric_system
from hydra import initialize, compose
import escnn
from escnn.nn import FieldType
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens

def isaaclab_joints_to_ms(joints, joint_order_for_morphosymm, amp_joint_names=None):
    """
    Converts a list of joint values in isaaclab to a list of morphosymm joint values, adding zeros for missing joints.
    """
    # Fill in zeros for all the joint-space obs for joints that are in ms but not isaaclab
    usd_name_to_vel_map = {name: pos for name, pos in zip(amp_joint_names, joints.T)} # type: ignore
    joints_ms = torch.stack([
        usd_name_to_vel_map.get(joint_name, torch.zeros(joints.shape[0], device=joints.device)) for joint_name in joint_order_for_morphosymm
    ], axis=1)
    return joints_ms

# def isaaclab_joints_to_ms(joints, joint_order_for_morphosymm, amp_joint_names=None):
#     """
#     Converts a list of joint values in isaaclab to a list of morphosymm joint values, adding zeros for missing joints.
#     Optimized for speed.
#     """
#     if amp_joint_names is None:
#         raise ValueError("amp_joint_names cannot be None for this optimized version.")

#     amp_name_to_idx = {name: i for i, name in enumerate(amp_joint_names)}
#     indices_in_isaaclab = torch.full(
#         (len(joint_order_for_morphosymm),),
#         -1, # A sentinel value indicating "not found"
#         dtype=torch.long,
#         device=joints.device
#     )

#     for i, ms_joint_name in enumerate(joint_order_for_morphosymm):
#         if ms_joint_name in amp_name_to_idx:
#             indices_in_isaaclab[i] = amp_name_to_idx[ms_joint_name]
#     joints_ms = torch.zeros(joints.shape[0], len(joint_order_for_morphosymm), dtype=joints.dtype, device=joints.device)
#     present_mask = indices_in_isaaclab != -1
#     valid_isaaclab_indices = indices_in_isaaclab[present_mask]
#     joints_ms[:, present_mask] = joints[:, valid_isaaclab_indices]

#     return joints_ms

def ms_joints_to_isaaclab(ms_joints, joint_order_for_morphosymm, amp_joint_names=None):
    """
    Converts a list of joint values in morphosymm to a list of isaaclab joint values, removing zeros used for missing joints in ms.
    """
    # Remove zeros for all the joint-space obs for joints that are in ms but not isaaclab
    ms_name_to_val_map = {name: val for name, val in zip(joint_order_for_morphosymm, ms_joints.T)}
    joints = torch.stack([ms_name_to_val_map[joint_name] for joint_name in amp_joint_names], axis=1)

    return joints

def compute_ms_observations_ideal(critic_obs, joint_order_for_morphosymm, amp_joint_names) -> torch.Tensor:
        # TODO this set of obs is only for the flat task
        joint_angles_ms = isaaclab_joints_to_ms(critic_obs[:, 9:35], joint_order_for_morphosymm, amp_joint_names)
        joint_vels_ms = isaaclab_joints_to_ms(critic_obs[:, 35:61], joint_order_for_morphosymm, amp_joint_names)
        past_action_ms = isaaclab_joints_to_ms(critic_obs[:, 61:87], joint_order_for_morphosymm, amp_joint_names)
        ms_critic_obs = torch.cat([critic_obs[:, :9], joint_angles_ms, joint_vels_ms, past_action_ms, critic_obs[:, 87:]], dim=-1)

        ms_critic_obs = ms_critic_obs.to(dtype=critic_obs.dtype)
        ms_critic_obs = ms_critic_obs.to(device=critic_obs.device)

        return ms_critic_obs

# def compute_ms_observations(self, critic_obs, joint_order_for_morphosymm, amp_joint_names) -> torch.Tensor:
#         # TODO this set of obs is only for the flat task
#         # TODO in this case I still want all the past obs, this obs is constructed with the past 10 base ang vels, projected gravity,
#         joint_angles_ms = isaaclab_joints_to_ms(critic_obs[:, 186:200], joint_order_for_morphosymm, amp_joint_names)
#         joint_vels_ms = isaaclab_joints_to_ms(critic_obs[:, 326:340], joint_order_for_morphosymm, amp_joint_names)
#         past_action_ms = isaaclab_joints_to_ms(critic_obs[:, 340:354], joint_order_for_morphosymm, amp_joint_names)
#         ms_critic_obs = torch.cat([critic_obs[:, :9], joint_angles_ms, joint_vels_ms, past_action_ms, critic_obs[:, 87:]], dim=-1)

#         ms_critic_obs = ms_critic_obs.to(dtype=critic_obs.dtype)
#         ms_critic_obs = ms_critic_obs.to(device=critic_obs.device)

#         return ms_critic_obs

def compute_ms_observations(critic_obs, joint_order_for_morphosymm, amp_joint_names) -> torch.Tensor:

        concatenated_joint_data_start_idx = 60
        ms_critic_obs_start = critic_obs[:, :concatenated_joint_data_start_idx]
        num_joints_amp = len(amp_joint_names) # Get this from amp_joint_names, not self.amp_joint_names if passed in
        num_joints_ms_order = len(joint_order_for_morphosymm) # Get this from the argument

        joint_angles_ms_list = []
        joint_vels_ms_list = []

        obs_set_size = 3 * num_joints_amp


        for i in range(10):
            current_set_start_idx = concatenated_joint_data_start_idx + (i * obs_set_size)

            current_joint_angles = critic_obs[:, current_set_start_idx : current_set_start_idx + num_joints_amp]
            joint_angles_ms_list.append(isaaclab_joints_to_ms(current_joint_angles, joint_order_for_morphosymm, amp_joint_names))

            current_joint_vels = critic_obs[:, current_set_start_idx + num_joints_amp : current_set_start_idx + (2 * num_joints_amp)]
            joint_vels_ms_list.append(isaaclab_joints_to_ms(current_joint_vels, joint_order_for_morphosymm, amp_joint_names))

        joint_angles_ms_all = torch.cat(joint_angles_ms_list, dim=-1)
        joint_vels_ms_all = torch.cat(joint_vels_ms_list, dim=-1)

        current_past_action = critic_obs[:, current_set_start_idx + (2 * num_joints_amp) : current_set_start_idx + (3 * num_joints_amp)]
        past_action_ms = isaaclab_joints_to_ms(current_past_action, joint_order_for_morphosymm, amp_joint_names)

        ms_critic_obs = torch.cat([
            ms_critic_obs_start,
            joint_angles_ms_all,
            joint_vels_ms_all,
            past_action_ms,
            critic_obs[:, concatenated_joint_data_start_idx + (10 * obs_set_size):]
        ], dim=-1)

        ms_critic_obs = ms_critic_obs.to(dtype=critic_obs.dtype)
        ms_critic_obs = ms_critic_obs.to(device=critic_obs.device)

        return ms_critic_obs

def extract_trained_model_info(state_dict, model_dir) -> (int, int, bool, int):
    """Extracts model information from a state_dict."""
    layers = 0
    hidden_units = 0
    obs_state_dim = 0
    has_bias = False

    for key in state_dict.keys():
        if ".obs_fn.net" in key:
            if "model.obs_fn.net.block_" in key and "weight" in key:
                layers += 1
            if "E-DAE" in model_dir or "EC-DAE" in model_dir:
                if "model.obs_fn.net.block_0.linear_0" in key and "matrix" in key:
                    state_dim = state_dict[key].shape[1]
            else:
                if "model.obs_fn.net.block_0" in key and "weight" in key:
                    state_dim = state_dict[key].shape[1]
            if "linear_0" in key and ("weight" in key or "matrix" in key):
                hidden_units = state_dict[key].shape[0]
            if 'bias' in key and not has_bias:
                has_bias = True
            if "head" in key and ("weight" in key or "matrix" in key):
                obs_state_dim = state_dict[key].shape[0]

    layers += 1  # Add one for the head layer

    return layers, hidden_units, has_bias, obs_state_dim, state_dim

def remove_state_dict_prefix(state_dict, prefix):
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}

def load_normalization_stats(model_path: str, device: torch.device):
    """
    From model_path get state_mean_var.npy file, which contains the PyTorch format of mean and std.
    """
    norm_path = os.path.join(model_path, "state_mean_var.npy")
    if not os.path.exists(norm_path):
        print(f"[Warning] Normalization file not found at: {norm_path}")
        # fallback to default
        return torch.zeros(35, device=device), torch.ones(35, device=device)

    norm_data = np.load(norm_path, allow_pickle=True).item()
    state_mean = torch.tensor(norm_data["state_mean"], device=device).float()
    state_var = torch.tensor(norm_data["state_var"], device=device).float()
    state_std = torch.sqrt(state_var)
    action_mean = torch.tensor(norm_data["action_mean"], device=device).float()
    action_var = torch.tensor(norm_data["action_var"], device=device).float()
    action_std = torch.sqrt(action_var)
    return state_mean, state_std, action_mean, action_std

def safe_standardize(x_normed: torch.Tensor | np.ndarray, mean: torch.Tensor | np.ndarray, std: torch.Tensor | np.ndarray):
    mask = std > 0
    if isinstance(x_normed, torch.Tensor):
        x_normed = x_normed.clone()
    if x_normed.ndim == 2:
        x_normed[:, mask] = (x_normed[:, mask] - mean[mask]) / std[mask]
    elif x_normed.ndim == 3:
        x_normed[:, :, mask] = (x_normed[:, :, mask] - mean[mask]) / std[mask]
    return x_normed

def get_trained_dae_model(model_dir, G):
    """
    Load the trained DAE model.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        torch.nn.Module: The trained model.
    """

    ckpt_path = os.path.join(model_dir, "best.ckpt")

    # Load the model from the checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Create the state representations
    gspace = escnn.gspaces.no_base_space(G)
    # Extract the representations from G.representations.items()
    rep_Rd = G.representations['R3']
    rep_TqQ_js = G.representations['TqQ_js']
    rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
    rep_xy.name = "base_xy"
    rep_euler_xyz = G.representations['euler_xyz']
    rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_euler_z.name = "euler_z"

    # Define the state and action type using the extracted representations
    state_reps = [rep_Rd, rep_euler_xyz, rep_Rd, rep_TqQ_js, rep_TqQ_js, rep_TqQ_js, rep_xy, rep_euler_z] # ['base_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'prev_action', 'velocity_commands_xy', 'velocity_commands_z']
    state_type = FieldType(gspace, representations=state_reps)
    state_type.size = sum(rep.size for rep in state_reps) + rep_Rd.size + 2 * rep_TqQ_js.size  # Count duplicates twice
    state_type = FieldType(gspace, representations=state_reps)
    action_reps = [rep_TqQ_js]  # ['actions']
    action_type = FieldType(gspace, representations=action_reps)
    action_type.size = sum(rep.size for rep in action_reps)

    num_layers, num_hidden_units, bias, obs_state_dim, state_dim = extract_trained_model_info(state_dict, model_dir)

    dt = 0.02
    orth_w_match = re.search(r"Orth_w:([\d\.]+)", model_dir)
    orth_w = float(orth_w_match.group(1)) if orth_w_match else 0.0
    obs_pred_w_match = re.search(r"Obs_w:([\d\.]+)", model_dir)
    obs_pred_w = float(obs_pred_w_match.group(1)) if obs_pred_w_match else 1.0
    group_avg_trick = True
    state_dependent_obs_dyn = False
    enforce_constant_fn = True
    act_match = re.search(r"Act:([\d\.]+)", model_dir)
    activation = obs_pred_w_match.group(1) if act_match else 'ELU'
    batch_norm = False

    if not "E-DAE" in model_dir and not "EC-DAE" in model_dir:
        activation = class_from_name("torch.nn", activation)

    obs_fn_params = {'num_layers': num_layers, 'num_hidden_units': num_hidden_units, 'activation': activation, 'bias': bias, 'batch_norm': batch_norm}

    initial_rng_state = torch.get_rng_state()

    if "E-DAE" in model_dir:
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
        )
    elif "EC-DAE" in model_dir:
        model = ControlledEquivDAE(
            state_rep=state_type.representation,
            action_rep=action_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
        )
    elif "C-DAE" in model_dir:
        model = ControlledDAE(
            state_dim=state_type.size,
            action_dim=action_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
        )
    else:
        corr_w = 0.0
        model = DAE(
            state_dim=state_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            corr_w=corr_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
        )

    torch.set_rng_state(initial_rng_state)
    model.load_state_dict(remove_state_dict_prefix(state_dict, "model."))

    return model

