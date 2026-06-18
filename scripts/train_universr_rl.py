import os
os.environ["WANDB_MODE"] = "offline"
import datetime
import contextlib
import random
import time
import json
import hashlib
import math
from concurrent import futures
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
import wandb
from tqdm import tqdm
from absl import app, flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from ml_collections import config_flags

import sys
sys.path.append("external/UniverSR")

# UniverSR Imports
from universr.models.unet import ConvNeXtUNetCond
from universr.flow.path import OriginalCFMPath
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT
from universr.utils.utils import read_file_list

# Flow GRPO Imports
import flow_grpo.rewards
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm, dynamic_ncols=True)
logger = get_logger(__name__)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

# =========================================================================
# 1. Dataset, Sampler and Collator
# =========================================================================

class AudioSRDataset(Dataset):
    def __init__(self, file_list, mode="train", num_samples=32768, target_sr=48000):
        self.file_paths = read_file_list(file_list)
        self.mode = mode
        self.num_samples = num_samples
        self.target_sr = target_sr

    def __len__(self):
        return len(self.file_paths)

    def _ensure(self, wav, L):
        if wav.shape[-1] < L:
            wav = torch.nn.functional.pad(wav, (0, 4000))
            reps = (L + wav.shape[-1] - 1) // wav.shape[-1]
            wav = wav.repeat(1, reps)[..., :L]
        elif wav.shape[-1] > L:
            wav = wav[..., :L]
        return wav

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y, sr = torchaudio.load(path)
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)

        gain = np.random.uniform(-1, -6) if self.mode == 'train' else -3
        peak = y.abs().max().clamp(min=1e-8)
        target_peak = 10 ** (gain / 20.0)
        y = y * (target_peak / peak)

        if sr != self.target_sr:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.target_sr)

        if self.mode == "train":
            if y.shape[-1] <= self.num_samples:
                y = self._ensure(y, self.num_samples)
            else:
                s = np.random.randint(0, y.shape[-1] - self.num_samples)
                y = y[..., s:s+self.num_samples]
        else:
            y = y[..., :48000*5] # Val 5 seconds

        return {
            'hr': y,
            'filename': os.path.basename(path),
            'path': path
        }

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.k = k                   
        self.num_replicas = num_replicas
        self.rank = rank              
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0
        self.m = self.total_samples // self.k  
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch

class AudioWaveformCollator:
    def __init__(self, target_sr=48000, sampling_rates_probs={8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1}):
        self.target_sr = target_sr
        self.sampling_rates = list(sampling_rates_probs.keys())
        self.probs = list(sampling_rates_probs.values())

    def _apply_lpf(self, hr_wave, low_sr_khz):
        original_len = hr_wave.shape[-1]
        target_sr_hz = low_sr_khz * 1000
        lr_wave_resampled = torchaudio.functional.resample(hr_wave, orig_freq=self.target_sr, new_freq=target_sr_hz)
        lr_wave_upsampled = torchaudio.functional.resample(lr_wave_resampled, orig_freq=target_sr_hz, new_freq=self.target_sr)
        return lr_wave_upsampled[..., :original_len]

    def __call__(self, batch):
        low_sr_khz = random.choices(self.sampling_rates, self.probs, k=1)[0]
        hr_waves = torch.stack([item['hr'].squeeze(0) for item in batch])
        lr_waves = self._apply_lpf(hr_waves, low_sr_khz)
        
        return {
            'hr': hr_waves.unsqueeze(1),       # [B, 1, T]
            'lr_wave': lr_waves.unsqueeze(1),  # [B, 1, T]
            'low_sr': [low_sr_khz] * len(batch),
            'filename': [item['filename'] for item in batch],
            'path': [item['path'] for item in batch],
        }

# =========================================================================
# 2. SDE Log-Prob Step
# =========================================================================

def sde_step_with_logprob_audio(xt, v, dt, noise_level=0.1, generator=None):
    """
    Euler SDE step: x_{t+dt} = x_t + v * dt + noise_level * sqrt(dt) * epsilon
    Returns the next sample and its log probability under this normal distribution.
    """
    mean = xt + v * dt
    variance = (noise_level ** 2) * dt
    std = torch.sqrt(torch.as_tensor(variance, device=xt.device))
    
    noise = torch.randn(xt.shape, dtype=xt.dtype, device=xt.device, generator=generator)
    nxt = mean + std * noise
    
    log_prob = -((nxt.detach() - mean) ** 2) / (2 * variance) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return nxt, log_prob

# =========================================================================
# 3. Main Loop
# =========================================================================

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name + "_" + unique_id if config.run_name else unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(project="flow_grpo_audio", name=config.run_name, config=config.to_dict())
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(config.logdir, config.run_name, "tb_logs"))

    logger.info(f"\\n{config}")
    set_seed(config.seed, device_specific=True)

    # 1. Models and Transforms
    model = ConvNeXtUNetCond(**config.universr.model)
    model.to(accelerator.device)
    
    transform = AmplitudeCompressedComplexSTFT(**config.universr.transform)
    transform.to(accelerator.device)

    path_obj = OriginalCFMPath(**config.universr.path.get("init_args", {}))

    # EMA
    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    ema = EMAModuleWrapper(trainable_parameters, decay=0.999, update_step_interval=8, device=accelerator.device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # 2. Reward fn
    reward_fn = flow_grpo.rewards.audio_multi_score(accelerator.device, config.reward_fn)

    # 3. Dataloaders
    train_dataset = AudioSRDataset(config.dataset.train_file_list, 'train', config.dataset.num_samples, config.dataset.sr)
    
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed
    )

    collator = AudioWaveformCollator(target_sr=config.dataset.sr)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.dataset.get("num_workers", 4),
        collate_fn=collator,
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    executor = futures.ThreadPoolExecutor(max_workers=config.train.get("max_workers", 8))

    logger.info("***** Running GRPO Training *****")
    
    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        model.eval()
        samples = []
        
        # Sampling Phase
        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling", disable=not accelerator.is_local_main_process):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            batch = next(train_iter)
            
            lr_audio = batch['lr_wave'].to(accelerator.device)
            hr_audio = batch['hr'].to(accelerator.device)
            sr_khz = batch['low_sr'][0]
            
            # STFT Preprocess
            Y_hr = transform(hr_audio)
            Y_hr = torch.view_as_real(Y_hr.squeeze(1)).permute(0, 3, 1, 2)[:, :, :-1, :] # [B, 2, F-1, T]
            
            Y_lr = transform(lr_audio)
            Y_lr = torch.view_as_real(Y_lr.squeeze(1)).permute(0, 3, 1, 2)[:, :, :-1, :] # [B, 2, F-1, T]
            
            lr_bin_count = accelerator.unwrap_model(model).sr_to_lr_bins[sr_khz]
            hf_start_bin = accelerator.unwrap_model(model).total_freq_bins - accelerator.unwrap_model(model).hr_freq_bins
            
            Y_lr_cond = Y_lr[:, :, :lr_bin_count, :]
            Y_hr_target = Y_hr[:, :, hf_start_bin:, :]
            
            x0 = path_obj.sample_source(Y_hr_target).to(accelerator.device)
            
            # Simulate SDE
            ts = torch.linspace(0, 1, config.sample.num_steps + 1, device=accelerator.device)
            
            xt = x0
            traj_latents = []
            traj_next_latents = []
            traj_log_probs = []
            
            with torch.no_grad():
                for t_idx in range(len(ts) - 1):
                    t_val = ts[t_idx].unsqueeze(0).repeat(xt.shape[0])
                    dt = ts[t_idx+1] - ts[t_idx]
                    
                    v_pred = model(xt, t_val, Y_lr_cond, sr_values=torch.tensor([sr_khz]*xt.shape[0], device=accelerator.device))
                    
                    nxt, log_prob = sde_step_with_logprob_audio(xt, v_pred, dt, noise_level=config.sample.noise_level)
                    
                    traj_latents.append(xt.clone())
                    traj_next_latents.append(nxt.clone())
                    traj_log_probs.append(log_prob)
                    xt = nxt
            
            # Stack trajectories
            latents = torch.stack(traj_latents, dim=1) # [B, num_steps, C, H, W]
            next_latents = torch.stack(traj_next_latents, dim=1)
            log_probs = torch.stack(traj_log_probs, dim=1) # [B, num_steps]
            
            # Postprocess to waveform
            slice_start = max(0, lr_bin_count - hf_start_bin)
            x1_spec = xt[:, :, slice_start:, :]
            full_spec = torch.cat([Y_lr_cond, x1_spec], dim=2)
            
            full_spec = torch.nn.functional.pad(full_spec, [0, 0, 0, 1], value=0)
            full_spec = full_spec.permute(0, 2, 3, 1).contiguous()
            full_spec = torch.view_as_complex(full_spec)
            sr_waveform = transform.invert(full_spec)
            
            # Calculate Rewards
            rewards_future = executor.submit(reward_fn, sr_waveform, hr_audio, sample_rate=sr_khz*1000)
            time.sleep(0)
            
            timesteps = ts[:-1].unsqueeze(0).repeat(latents.shape[0], 1) # [B, num_steps]
            
            samples.append({
                "latents": latents,
                "next_latents": next_latents,
                "log_probs": log_probs,
                "timesteps": timesteps,
                "Y_lr_cond": Y_lr_cond,
                "sr_khz": torch.tensor([sr_khz]*latents.shape[0], device=accelerator.device),
                "rewards_future": rewards_future
            })

        for sample in tqdm(samples, desc="Waiting for rewards", disable=not accelerator.is_local_main_process):
            rewards, _ = sample["rewards_future"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            del sample["rewards_future"]

        # Collate samples
        samples = {k: torch.cat([s[k] for s in samples], dim=0) if not isinstance(samples[0][k], dict) else {sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]} for k in samples[0].keys()}
        
        # Calculate Advantages (group-wise)
        avg_rewards = samples["rewards"]["avg"] # [B]
        group_size = config.sample.num_image_per_prompt
        # reshape to [num_unique_samples, k] to compute mean/std per group
        reshaped_rewards = avg_rewards.view(-1, group_size)
        mean_rewards = reshaped_rewards.mean(dim=1, keepdim=True)
        std_rewards = reshaped_rewards.std(dim=1, keepdim=True)
        advantages = (reshaped_rewards - mean_rewards) / (std_rewards + 1e-4)
        advantages = advantages.view(-1) # back to [B]
        
        samples["advantages"] = advantages
        samples["rewards"]["avg"] = avg_rewards.unsqueeze(1).repeat(1, config.sample.num_steps)
        
        gathered_rewards = {key: accelerator.gather(value).cpu().numpy() for key, value in samples["rewards"].items()}
        
        if accelerator.is_main_process:
            wandb.log({f"reward_{key}": value.mean() for key, value in gathered_rewards.items()}, step=global_step)
            for key, value in gathered_rewards.items():
                tb_writer.add_scalar(f"rewards/{key}", float(value.mean()), global_step)
            reward_str = ", ".join([f"{key}: {value.mean():.4f}" for key, value in gathered_rewards.items()])
            logger.info(f"Epoch {epoch} | Rewards -> {reward_str}")
            
        # Optimization Phase
        model.train()
        for j in tqdm(range(config.sample.num_steps), desc="Optimization", disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                xt = samples["latents"][:, j].requires_grad_(True)
                t_val = samples["timesteps"][:, j]
                Y_lr_cond = samples["Y_lr_cond"]
                sr_khz_tensor = samples["sr_khz"]
                
                v_pred = model(xt, t_val, Y_lr_cond, sr_values=sr_khz_tensor)
                
                dt = (1.0 / config.sample.num_steps)
                _, new_log_prob = sde_step_with_logprob_audio(xt, v_pred, dt, noise_level=config.sample.noise_level)
                
                old_log_prob = samples["log_probs"][:, j]
                adv = samples["advantages"]
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - config.train.clip_eps, 1.0 + config.train.clip_eps) * adv
                
                loss = -torch.min(surr1, surr2).mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        accelerator.free_memory()

        if accelerator.sync_gradients:
            ema.update(trainable_parameters)
            global_step += 1
            if global_step % 100 == 0 and accelerator.is_main_process:
                # save
                save_dir = os.path.join(config.logdir, config.run_name, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        epoch += 1

if __name__ == "__main__":
    app.run(main)
