import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector.vector_env import VectorEnv

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv

from src.args import Args


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.control_mode = args.control_mode
        
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        
        self.act_dim = env.single_action_space.shape[0]
        
        # Store action bounds for normalization
        self.register_buffer('action_low', torch.tensor(env.single_action_space.low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(env.single_action_space.high, dtype=torch.float32))
        
        # Check if we need to normalize (action space is not already [-1, 1])
        self.needs_normalization = not (
            (env.single_action_space.high == 1).all() and 
            (env.single_action_space.low == -1).all()
        )
        
        if self.needs_normalization:
            print(f"Action space bounds: [{self.action_low.min().item():.2f}, {self.action_high.max().item():.2f}]")
            print(f"Will normalize actions to [-1, 1] for training")
        else:
            print("Action space already normalized to [-1, 1]")
        
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",
        )

    def normalize_action(self, action):
        """Normalize action from [low, high] to [-1, 1]"""
        if not self.needs_normalization:
            return action
        # Ensure action_low and action_high are on the same device as action
        action_low = self.action_low.to(action.device)
        action_high = self.action_high.to(action.device)
        return 2.0 * (action - action_low) / (action_high - action_low) - 1.0
    
    def denormalize_action(self, action):
        """Denormalize action from [-1, 1] to [low, high]"""
        if not self.needs_normalization:
            return action
        # Ensure action_low and action_high are on the same device as action
        action_low = self.action_low.to(action.device)
        action_high = self.action_high.to(action.device)
        return (action + 1.0) / 2.0 * (action_high - action_low) + action_low

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # Normalize actions to [-1, 1] for training
        action_seq = self.normalize_action(action_seq)

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=action_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action_seq.device
        ).long()

        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # Denormalize from [-1, 1] to actual action space
        action_seq = self.denormalize_action(noisy_action_seq)

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return action_seq[:, start:end]  # (B, act_horizon, act_dim)
