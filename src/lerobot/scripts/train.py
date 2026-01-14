import gc
import torch

# from tests.envs.mujoco.test_mujoco_rendering import model

def get_cuda_tensor_ids():
    return {id(obj): obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda}

def print_new_cuda_tensors(prev_tensors):
    curr_tensors = get_cuda_tensor_ids()
    new_ids = set(curr_tensors.keys()) - set(prev_tensors.keys())
    print("New CUDA tensors added this step:")
    for tid in new_ids:
        t = curr_tensors[tid]
        print(f"ID: {tid}, Tensor: {t.shape}, dtype: {t.dtype}, size: {t.element_size() * t.nelement() / 1024**2:.2f} MB, requires_grad: {t.requires_grad}")

# Example usage in your training loop:
# prev_tensors = get_cuda_tensor_ids()
# ... training step ...
# print_new_cuda_tensors(prev_tensors)
import torch
import gc
# ...existing code...
# After buffer/model creation, add memory usage reporting
def report_gpu_memory(buffer=None):
    print("Total GPU memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Total GPU memory reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            print(f"Tensor: {obj.shape}, dtype: {obj.dtype}, size: {obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
    if buffer is not None:
        for name, tensor in buffer.__dict__.items():
            if torch.is_tensor(tensor) and tensor.is_cuda:
                print(f"Buffer field {name}: {tensor.shape}, {tensor.element_size() * tensor.nelement() / 1024**2:.2f} MB")

import gc
import torch

def print_live_cuda_tensors():
    print("Live CUDA tensors after step:")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"Tensor: {obj.shape}, dtype: {obj.dtype}, size: {obj.element_size() * obj.nelement() / 1024**2:.2f} MB, requires_grad: {obj.requires_grad}")
        except Exception:
            pass

# Example usage after buffer/model creation:
# report_gpu_memory(buffer)
#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
# import numpy as np


import torch

from termcolor import colored

from torch.amp import GradScaler

from torch.optim import Optimizer

import gymnasium as gym

from collections import namedtuple


from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.utils.buffer import RolloutBufferTorch


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # print("DEBUG: batch action size", batch["action"].shape)
        loss, output_dict, _, _, _ = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    # print("DEBUG: look at output_dict to potentially put into replay buffer", output_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def update_policy_ppo(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    penalty=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # print("DEBUG: batch action size", batch["action"].shape)
        loss_clip, loss_dict = ppo_clip_loss(policy, batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        # if penalty is not None:
        #     loss_clip += penalty * 0.1  # penalty coefficient
        print("DEBUG: PPO total loss", loss_clip)
    grad_scaler.scale(loss_clip).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    print("DEBUG: Optimizer", optimizer)
    grad_scaler.unscale_(optimizer)

    print("DEBUG:", grad_clip_norm)
    
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    # train_metrics.loss = loss_clip.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    train_metrics.ppo_loss = loss_dict["ppo_loss"]
    return train_metrics, loss_dict


def ppo_clip_loss(policy, batch, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    batch: dict with keys:
        - observations
        - actions
        - log_probs (old)
        - advantages
        - returns
    """
    # Get new log_probs, entropy, and values from the policy
    new_log_probs, entropy, new_values = policy.evaluate_actions(
        batch
    )
    device = batch["device"]
    # Calculate probability ratio
    ratios = torch.exp(new_log_probs - batch["log_probs"]).to(device)
    # Clipped surrogate objective
    surr1 = ratios * batch["advantages"].to(device)
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch["advantages"].to(device)
    policy_loss = -torch.min(surr1, surr2).mean().to(device)
    # Value function loss
    value_loss = value_coef * (new_values.squeeze(-1).to(device) - batch["returns"].to(device)).pow(2).mean()
    # Entropy bonus
    entropy_loss = -entropy_coef * entropy.mean()
    # Total loss
    loss = policy_loss + value_loss + entropy_loss
    print("DEBUG: PPO LOSS", loss)
    return loss, {"ppo_loss":loss.item() ,"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy_loss": entropy_loss.item()}

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    
    logging.info("Creating rollout buffer")
    rollout_buffer = RolloutBufferTorch(
        buffer_size=cfg.replay_capacity,
        obs_shape=dataset.features["observation.images.front"]["shape"],
        action_shape=dataset.features["action"]["shape"],
        device="cpu",
    )
    print("rollout_buffer", rollout_buffer.device)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        print("DEBUG:", cfg.eval.batch_size)
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    print("[DEBUG] Creating policy")
    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    print("[DEBUG] Policy created")
    
    print("[DEBUG] Creating optimizer and scheduler")
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    print("[DEBUG] Optimizer, scheduler, grad_scaler created")

    print("[DEBUG] Entering training loop setup")
    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        print("[DEBUG] Loading training state from checkpoint")
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    print("[DEBUG] Counting parameters")
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


    print("[DEBUG] Creating dataloader")
    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print("[DEBUG] Dataloader created")
    dl_iter = cycle(dataloader)
    print("[DEBUG] Dataloader iterator created")

    print("[DEBUG] Setting policy to train mode")

    policy.train()

    print("[DEBUG] Initializing metrics")
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "ppo_loss": AverageMeter("ppo_loss", ":.3f"),
    }

    print("[DEBUG] Initializing train tracker")
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # logging.info("Starting PPO layer warmup")

    # for name, param in policy.named_parameters():
    #     if "actor_head" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
            
    # head_params = []
    # for name, param in policy.named_parameters():
    #     if "actor_head" in name:
    #         head_params.append(param)
            
    # head_optimizer = torch.optim.AdamW(head_params, lr=3e-4, weight_decay=1e-2)
    
    # for warmup_step in range(1000):
    #         ppo_env = make_env(cfg.env)
    #         obs, info = ppo_env.reset()
    #         # obs = obs[:168, ...]

    #         ppo_env.render()
    #         ppo_step = 0
    #         timestamp = ppo_step / cfg.env.fps
    #         done = False
    #         norm_obs = obs/255.0
    #         # print("DEBUG:", norm_obs)
    #         batch = {
    #                 "observation.images.front": torch.tensor(norm_obs).unsqueeze(0).to(device).permute(0,3,1,2),
    #                 "action": torch.zeros((1, ppo_env.action_space.shape[0])).unsqueeze(0).to(device),
    #                 "observation.state": torch.zeros((1,ppo_env.action_space.shape[0])).unsqueeze(0).to(device),
    #                 "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
    #                 "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
    #                 "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
    #                 "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),

    #                 "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
    #                 "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #                 "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #                 "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #                 "task": ["Drive on the road."],
    #             }
    #         for k in batch:
    #             if k == "task":
    #                 continue
    #             if batch[k].isnan().any():
    #                 print("DEBUG: NaN in batch key", k)

    #         while not done:
    #             # # report_gpu_memory(rollout_buffer)
    #             # print_new_cuda_tensors(prev_tensors)
    #             # prev_tensors = get_cuda_tensor_ids()
    #             # # print_live_cuda_tensors()
    #             # print("------------------------------------------------DEBUG: PPO step------------------------------------------------")
    #             # print("DEBUG: PPO step start", ppo_step)
    #             # get action distributions for further calcs
    #             dists, value = policy.get_action_distributions(batch)
    #             # sample action
    #             raw_action = dists.rsample()
                
    #             penalty_0 = ((raw_action[..., 0] < -1).float() * (-1 - raw_action[..., 0]).pow(2) + (raw_action[..., 0] > 1).float() * (raw_action[..., 0] - 1).pow(2))
    #             penalty_rest = ((raw_action[..., 1:] < 0).float() * (0 - raw_action[..., 1:]).pow(2) + (raw_action[..., 1:] > 1).float() * (raw_action[..., 1:] - 1).pow(2))
    #             penalty = penalty_0.mean() + penalty_rest.mean()
                
    #             # map action to environment friendly range of [(-1,1),(0,1),(0,1)]
    #             action = torch.zeros(batch["action"].size())
    #             action[..., 0] = torch.tanh(raw_action[..., 0])
    #             action[..., 1:] = torch.sigmoid(raw_action[..., 1:])
    #             # calculate log_probs
    #             log_probs = dists.log_prob(raw_action)
    #             if (raw_action >= 100).any() or (raw_action <= -100).any(): 
    #                 print("DEBUG: raw_action", raw_action)
    #                 print("DEBUG: action", value)
    #             if log_probs.isnan().any():
    #                 print("DEBUG: NaN in log_probs")
    #                 print("DEBUG: log_probs", log_probs)
    #                 print("DEBUG: raw_action", raw_action)
    #                 print("DEBUG: dist", dists, value, action)
    #                 exit(1)
    #             # mappings for log_probs and their use
    #             tanh_jacobian = torch.log(1 - action[..., 0]**2 + 1e-6)
    #             sigmoid_jacobian = action[..., 1:].log() + (1- action[..., 1:]).log()
    #             tanh_jacobian = tanh_jacobian.to(device)
    #             sigmoid_jacobian = sigmoid_jacobian.to(device)
    #             log_probs[..., 0] -= tanh_jacobian
    #             log_probs[..., 1:] -= sigmoid_jacobian
    #             # sum to one log_prob
    #             log_prob = log_probs.sum(-1)  # maybe change for larger batches
    #             # TODO: Fix mask calculation
    #             # TODO: Calculate log_prob etc. for PPO
    #             # TODO: Store transition in replay buffer
    #             # TODO: Perform PPO update steps

    #             np_action = action.squeeze(0).squeeze(0).detach().numpy()
    #             obs, reward, terminated, truncated, info = ppo_env.step(np_action)
    #             # obs = obs[:168, ...]
    #             ppo_env.render()
    #             # action = action.unsqueeze(0)
    #             prev_action = batch["action"].unsqueeze(0)
    #             prev_obs = batch["observation.images.front"]
    #             ppo_step += 1
    #             timestamp = ppo_step / cfg.env.fps
    #             norm_obs = obs/255.0
    #             # batch = {
    #             #     "observation.images.front": torch.tensor(norm_obs).unsqueeze(0).to(device).permute(0,3,1,2),
    #             #     "action": action.detach().clone().to(device),
    #             #     "observation.state": prev_action.detach().clone().to(device),  # just a placeholder
    #             #     "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #             #     "task": ["Drive on the road."],
    #             # }
    #             # print("DEBUG: action and prev_action:", action, prev_action)
    #             batch = {
    #                 "observation.images.front": torch.tensor(norm_obs).unsqueeze(0).to(device).permute(0,3,1,2),
    #                 "action": action.detach().clone().to(device),
    #                 "observation.state": prev_action.detach().clone().to(device),  # just a placeholder
    #                 "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
    #                 "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
    #                 "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
    #                 "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
    #                 "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
    #                 "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #                 "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #                 "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
    #                 "task": ["Drive on the road."],
    #             }
    #             for k in batch:
    #                 if k == "task":
    #                     continue
    #                 if batch[k].isnan().any():
    #                     print("DEBUG: NaN in batch key", k)
    #             transition = {
    #                             "obs": prev_obs.permute(0,2,3,1).squeeze(0).detach().cpu(),
    #                             "action": action.squeeze(0).squeeze(0).squeeze(0).detach().cpu(),
    #                             "reward": reward,
    #                             "done": terminated,
    #                             "log_prob": log_prob.detach().cpu(),
    #                             "value": value.detach().cpu(),
    #                             "timestamp": timestamp,
    #                             "frame_index": ppo_step,
    #                             "episode_index": 0,
    #                             "index": ppo_step,
    #                             "task_index": 0,
    #                             "action_is_pad": torch.tensor(False).unsqueeze(0),
    #                             "state_is_pad": torch.tensor(False).unsqueeze(0),
    #                             "image_is_pad": torch.tensor(False).unsqueeze(0),
    #                             "task": "Drive on the road.",
    #                          }
    #             rollout_buffer.add(**transition)
    #             if rollout_buffer.ptr >= rollout_buffer.buffer_size:
    #                 print("DEBUG: Performing PPO update from rollout buffer")
    #                 # compute GAE advantages
    #                 with torch.no_grad():
    #                     next_value = policy.get_value(batch).squeeze(0)
    #                 rewards = rollout_buffer.rewards
    #                 values = rollout_buffer.values
    #                 print("DEBUG:", values)
    #                 dones = rollout_buffer.dones
    #                 advantages = compute_gae(rewards, values, dones, next_value)
    #                 # print("DEBUG: advantages mean", advantages.mean())
    #                 # print("DEBUG: advantages std", advantages.std())
    #                 returns = advantages + values
    #                 # Normalize advantages
    #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #                 # print("DEBUG: rewards mean", returns.mean())
    #                 # print("DEBUG: rewards std", returns.std())
    #                 # PPO update loop
    #                 ppo_batch_size = 32
    #                 ppo_epochs = 4
    #                 for epoch in range(ppo_epochs):
    #                     for start in range(0, rollout_buffer.buffer_size, ppo_batch_size):
    #                         end = start + ppo_batch_size
    #                         mbatch_ppo = rollout_buffer.get_ppo_batch(start, end)
    #                         for k in mbatch_ppo:
    #                             if isinstance(mbatch_ppo[k], torch.Tensor):
    #                                 mbatch_ppo[k] = mbatch_ppo[k].to(device)
    #                         for k in mbatch_ppo:
    #                             if k == "task":
    #                                 continue
    #                             if mbatch_ppo[k].isnan().any():
    #                                 print("DEBUG: NaN in mbatch_ppo key", k)
    #                         mbatch_smolvla = rollout_buffer.get_smolvla_batch(start, end)
    #                         for k in mbatch_smolvla:
    #                             if isinstance(mbatch_smolvla[k], torch.Tensor):
    #                                 mbatch_smolvla[k] = mbatch_smolvla[k].unsqueeze(0).to(device)
    #                         for k in mbatch_smolvla:
    #                             if k == "task":
    #                                 continue
    #                             if mbatch_smolvla[k].isnan().any():
    #                                 print("DEBUG: NaN in mbatch_smolvla key", k)
    #                         mbatch = {**mbatch_ppo, **mbatch_smolvla}
    #                         mbatch["advantages"] = advantages[start:end]
    #                         mbatch["returns"] = returns[start:end]
    #                         mbatch["device"] = device
    #                         print("DEBUG: PPO minibatch from", start, "to", end)
    #                         # mbatch.to(device)
    #                         train_tracker, output_dict = update_policy_ppo(
    #                             train_tracker,
    #                             policy,
    #                             mbatch,
    #                             head_optimizer,
    #                             0.1,
    #                             # cfg.optimizer.grad_clip_norm,
    #                             grad_scaler=grad_scaler,
    #                             lr_scheduler=lr_scheduler,
    #                             use_amp=cfg.policy.use_amp,
    #                             penalty=penalty,
    #                         )
    #                     logging.info(train_tracker)
    #                     if wandb_logger:
    #                         wandb_log_dict = train_tracker.to_dict()
    #                         if output_dict:
    #                             wandb_log_dict.update(output_dict)
    #                         wandb_logger.log_dict(wandb_log_dict, step)
    #                     train_tracker.reset_averages()
    #                 rollout_buffer.reset()

    #             # policy.forward(batch)  # to potentially update internal buffers
    #             done = terminated or truncated  
    #         print("DEBUG: PPO episode done", done)
    #         warmup_step += 1
    #         if warmup_step % 100 == 0:
    #             logging.info(f"--------------- PPO head warmup step {warmup_step} ----------------")
    # for param in policy.parameters():
    #     param.requires_grad = True
    
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    
    # logging.info("PPO layer warmup completed")

    print("[DEBUG] Starting training loop")
    logging.info("Start offline training on a fixed dataset")
    for name, param in policy.named_parameters():
            if "actor_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    for _ in range(step, cfg.steps):
        
        # prev_tensors = get_cuda_tensor_ids()
        # # ... your training step code ...
        # # Existing code for training step goes here
        # print_new_cuda_tensors(prev_tensors)
        # # print_live_cuda_tensors()
        # print("-----------------------------------------------------------------------------------------------------------"))
        is_DRL_step = cfg.DRL_freq > 0 and step > 0 and  step % cfg.DRL_freq == 0
        # is_DRL_step = cfg.DRL_freq > 0 and step % cfg.DRL_freq == 0
        print(step, is_DRL_step)
        if is_DRL_step and step > 0:
            for name, param in policy.named_parameters():
                if "actor_head" in name:
                    param.requires_grad = True
                    # print("DEBUG: Enabling gradient for", name)
                    
            head_params = []
            for name, param in policy.named_parameters():
                if "actor_head" in name:
                    head_params.append(param)
            print("DEBUG: Creating head optimizer for DRL step", head_params)
            head_optimizer = torch.optim.AdamW(head_params, lr=3e-4, weight_decay=1e-2)
            # # print_live_cuda_tensors()
            # prev_tensors = get_cuda_tensor_ids()
            # print_new_cuda_tensors(prev_tensors)
            # print("------------------------------------------------DEBUG: IN DRL STEP ------------------------------------------------")
            logging.info("Performing DRL step. Full PPO episode.")
            ppo_env = make_env(cfg.env)
            obs, info = ppo_env.reset()
            # obs = obs[:168, ...]

            ppo_env.render()
            ppo_step = 0
            timestamp = ppo_step / cfg.env.fps
            done = False
            norm_obs = obs/255.0
            # print("DEBUG:", norm_obs)
            batch = {
                    "observation.images.front": torch.tensor(norm_obs).to(device).permute(0,3,1,2),
                    "action": torch.zeros((1, ppo_env.action_space.shape[1])).to(device),
                    "observation.state": torch.zeros((1,ppo_env.action_space.shape[1])).to(device),
                    "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
                    "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
                    "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                    "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),

                    "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                    "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "task": ["Drive on the road."],
                }
            print("DEBUG: Initial PPO batch created")
            for k in batch:
                if k == "task":
                    continue
                if batch[k].isnan().any():
                    print("DEBUG: NaN in batch key", k)

            # while not done:
            while rollout_buffer.ptr < rollout_buffer.buffer_size:
                if done:
                    print("DEBUG: PPO episode done", done)
                    obs, info = ppo_env.reset()
                    # obs = obs[:168, ...]
                    ppo_env.render()
                    ppo_step = 0
                    timestamp = ppo_step / cfg.env.fps
                    done = False
                    norm_obs = obs/255.0
                    batch = {
                    "observation.images.front": torch.tensor(norm_obs).to(device).permute(0,3,1,2),
                    "action": torch.zeros((1, ppo_env.action_space.shape[1])).to(device),
                    "observation.state": torch.zeros((1,ppo_env.action_space.shape[1])).to(device),
                    "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
                    "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
                    "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                    "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),

                    "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                    "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "task": ["Drive on the road."],
                }
                # # report_gpu_memory(rollout_buffer)
                # print_new_cuda_tensors(prev_tensors)
                # prev_tensors = get_cuda_tensor_ids()
                # # print_live_cuda_tensors()
                print("------------------------------------------------DEBUG: PPO step------------------------------------------------")
                # print("DEBUG: PPO step start", ppo_step)
                # get action distributions for further calcs
                dists, value = policy.get_action_distributions(batch)
                # sample action
                raw_action = dists.rsample()
                # map action to environment friendly range of [(-1,1),(0,1),(0,1)]
                action = torch.zeros(batch["action"].size())
                print(action.size())
                action[..., 0] = torch.tanh(raw_action[..., 0])
                action[..., 1:] = torch.sigmoid(raw_action[..., 1:])
                # calculate log_probs
                log_probs = dists.log_prob(raw_action)
                if (raw_action >= 100).any() or (raw_action <= -100).any(): 
                    print("DEBUG: raw_action", raw_action)
                    print("DEBUG: action", value)
                if log_probs.isnan().any():
                    print("DEBUG: NaN in log_probs")
                    print("DEBUG: log_probs", log_probs)
                    print("DEBUG: raw_action", raw_action)
                    print("DEBUG: dist", dists, value, action)
                    exit(1)
                # mappings for log_probs and their use
                tanh_jacobian = torch.log(1 - action[..., 0]**2 + 1e-6)
                sigmoid_jacobian = action[..., 1:].log() + (1- action[..., 1:]).log()
                tanh_jacobian = tanh_jacobian.to(device)
                sigmoid_jacobian = sigmoid_jacobian.to(device)
                log_probs[..., 0] -= tanh_jacobian
                log_probs[..., 1:] -= sigmoid_jacobian
                # sum to one log_prob
                log_prob = log_probs.sum(-1)  # maybe change for larger batches
                # TODO: Fix mask calculation
                # TODO: Calculate log_prob etc. for PPO
                # TODO: Store transition in replay buffer
                # TODO: Perform PPO update steps
                np_action = action.detach().numpy()
                print("DEBUG: Taking action in PPO env:", np_action)
                obs, reward, terminated, truncated, info = ppo_env.step(np_action)
                # obs = obs[:168, ...]
                ppo_env.render()
                # action = action.unsqueeze(0)
                prev_action = batch["action"].unsqueeze(0)
                prev_obs = batch["observation.images.front"]
                ppo_step += 1
                timestamp = ppo_step / cfg.env.fps
                norm_obs = obs/255.0
                # batch = {
                #     "observation.images.front": torch.tensor(norm_obs).unsqueeze(0).to(device).permute(0,3,1,2),
                #     "action": action.detach().clone().to(device),
                #     "observation.state": prev_action.detach().clone().to(device),  # just a placeholder
                #     "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
                #     "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
                #     "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                #     "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
                #     "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                #     "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                #     "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                #     "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                #     "task": ["Drive on the road."],
                # }
                batch = {
                    "observation.images.front": torch.tensor(norm_obs).to(device).permute(0,3,1,2),
                    "action": action.detach().clone().to(device),
                    "observation.state": prev_action.detach().clone().to(device),  # just a placeholder
                    "timestamp": torch.tensor(timestamp).unsqueeze(0).unsqueeze(0).to(device),
                    "frame_index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
                    "episode_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                    "index": torch.tensor(ppo_step).unsqueeze(0).unsqueeze(0).to(device),
                    "task_index": torch.tensor(0).unsqueeze(0).unsqueeze(0).to(device),
                    "action_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "observation.state_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "observation.images.front_is_pad": torch.tensor(False).unsqueeze(0).unsqueeze(0).to(device),
                    "task": ["Drive on the road."],
                }
                for k in batch:
                    if k == "task":
                        continue
                    if batch[k].isnan().any():
                        print("DEBUG: NaN in batch key", k)
                        
                # TODO: Put the scalars into tensors in the transition
                transition = {
                                "obs": prev_obs.permute(0,2,3,1).squeeze(0).detach().cpu(),
                                "action": action.squeeze(0).squeeze(0).squeeze(0).detach().cpu(),
                                "reward": reward,
                                "done": terminated,
                                "log_prob": log_prob.detach().cpu(),
                                "value": value.detach().cpu(),
                                "timestamp": timestamp,
                                "frame_index": ppo_step,
                                "episode_index": 0,
                                "index": ppo_step,
                                "task_index": 0,
                                "action_is_pad": torch.tensor(False).unsqueeze(0),
                                "state_is_pad": torch.tensor(False).unsqueeze(0),
                                "image_is_pad": torch.tensor(False).unsqueeze(0),
                                "task": "Drive on the road.",
                             }
                rollout_buffer.add(**transition)
                print("DEBUG: Transition added to rollout buffer. Current ptr:", rollout_buffer.ptr)
            if rollout_buffer.ptr >= rollout_buffer.buffer_size:
                    print("DEBUG: Performing PPO update from rollout buffer")
                    # compute GAE advantages
                    with torch.no_grad():
                        next_value = policy.get_value(batch).squeeze(0)
                    rewards = rollout_buffer.rewards
                    values = rollout_buffer.values
                    print("DEBUG:", values)
                    dones = rollout_buffer.dones
                    advantages = compute_gae(rewards, values, dones, next_value)
                    # print("DEBUG: advantages mean", advantages.mean())
                    # print("DEBUG: advantages std", advantages.std())
                    returns = advantages + values
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # print("DEBUG: rewards mean", returns.mean())
                    # print("DEBUG: rewards std", returns.std())
                    # PPO update loop
                    ppo_batch_size = 32
                    ppo_epochs = 4
                    for epoch in range(ppo_epochs):
                        for start in range(0, rollout_buffer.buffer_size, ppo_batch_size):
                            end = start + ppo_batch_size
                            mbatch_ppo = rollout_buffer.get_ppo_batch(start, end)
                            for k in mbatch_ppo:
                                if isinstance(mbatch_ppo[k], torch.Tensor):
                                    mbatch_ppo[k] = mbatch_ppo[k].to(device)
                            for k in mbatch_ppo:
                                if k == "task":
                                    continue
                                if mbatch_ppo[k].isnan().any():
                                    print("DEBUG: NaN in mbatch_ppo key", k)
                            mbatch_smolvla = rollout_buffer.get_smolvla_batch(start, end)
                            for k in mbatch_smolvla:
                                if isinstance(mbatch_smolvla[k], torch.Tensor):
                                    mbatch_smolvla[k] = mbatch_smolvla[k].unsqueeze(0).to(device)
                            for k in mbatch_smolvla:
                                if k == "task":
                                    continue
                                if mbatch_smolvla[k].isnan().any():
                                    print("DEBUG: NaN in mbatch_smolvla key", k)
                            mbatch = {**mbatch_ppo, **mbatch_smolvla}
                            mbatch["advantages"] = advantages[start:end]
                            mbatch["returns"] = returns[start:end]
                            mbatch["device"] = device
                            print("DEBUG: PPO minibatch from", start, "to", end)
                            # mbatch.to(device)
                            train_tracker, output_dict = update_policy_ppo(
                                train_tracker,
                                policy,
                                mbatch,
                                head_optimizer,
                                0.1,
                                # cfg.optimizer.grad_clip_norm,
                                grad_scaler=grad_scaler,
                                lr_scheduler=lr_scheduler,
                                use_amp=cfg.policy.use_amp,
                            )
                        logging.info(train_tracker)
                        if wandb_logger:
                            wandb_log_dict = train_tracker.to_dict()
                            if output_dict:
                                wandb_log_dict.update(output_dict)
                            wandb_logger.log_dict(wandb_log_dict, step)
                        train_tracker.reset_averages()
                    rollout_buffer.reset()

                # policy.forward(batch)  # to potentially update internal buffers
            done = terminated or truncated  
            print("DEBUG: PPO episode done", done)
            step += 1
            for name, param in policy.named_parameters():
                if "actor_head" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            continue  # After one DRL/PPO step, break to return to SFT (offline) training does not work right now

        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
        
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

        
            
            
            episode_rewards = []
            logging.info(f"DRL episode reward: {sum(episode_rewards):.2f}")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    rewards: [T]
    values: [T]
    dones: [T]
    next_value: scalar
    Returns: advantages [T]
    """
    T = len(rewards)
    dones_f = dones.float()
    advantages = torch.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        # delta = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
        if t == T - 1:
            next_v = next_value
        else:
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * (1 - dones_f[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones_f[t]) * gae
        advantages[t] = gae
    return advantages
    

def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
