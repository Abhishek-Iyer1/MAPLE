ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import gc
import h5py
import json
import os
import random
import time
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    worker_init_fn,
)
from diffusion_policy.utils import load_demo_dataset
from diffusion_policy.wandb_logger import create_logger

from src.agent import Agent
from src.args import Args
from src.training.datasets import SmallDemoDataset_DiffusionPolicy
from src.training.training_utils import reorder_keys
from src.envs.pick_cube_side import PickCubeSideViewEnv  # noqa: F401 — registers PickCube-SideView-v1


class Trainer:
    def __init__(self, args: Args):
        self.args = args

    def train(self):
        self.setup()
        self._run_training_loop()

    def setup(self):
        self._resolve_run_name()
        self._validate_demo_config()
        self._set_seeds()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.args.cuda else "cpu"
        )
        self.envs = self._setup_envs()
        self.logger = self._setup_logger()
        self.dataset, self.train_dataloader = self._setup_dataset()
        (
            self.agent,
            self.ema_agent,
            self.optimizer,
            self.lr_scheduler,
            self.ema,
        ) = self._setup_agent()
        self.best_eval_metrics = defaultdict(float)
        self.timings = defaultdict(float)

    def _resolve_run_name(self):
        args = self.args
        if args.exp_name is None:
            args.exp_name = os.path.basename(__file__)[: -len(".py")]
            self.run_name = (
                f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
            )
        else:
            self.run_name = args.exp_name

    def _validate_demo_config(self):
        args = self.args
        if args.demo_path.endswith(".h5"):
            json_file = args.demo_path[:-2] + "json"
            with open(json_file, "r") as f:
                demo_info = json.load(f)
                if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                    control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
                elif "control_mode" in demo_info["episodes"][0]:
                    control_mode = demo_info["episodes"][0]["control_mode"]
                else:
                    raise Exception("Control mode not found in json")
                assert control_mode == args.control_mode, (
                    f"Control mode mismatched. Dataset has control mode {control_mode}, "
                    f"but args has control mode {args.control_mode}"
                )
        assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
        assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    def _set_seeds(self):
        args = self.args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

    def _setup_envs(self):
        args = self.args
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
        )
        assert args.max_episode_steps is not None, (
            "max_episode_steps must be specified as imitation learning algorithms "
            "task solve speed is dependent on the data you train on"
        )
        env_kwargs["max_episode_steps"] = args.max_episode_steps
        other_kwargs = dict(obs_horizon=args.obs_horizon)

        wrappers = [FlattenRGBDObservationWrapper]

        envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{self.run_name}/videos" if args.capture_video else None,
            wrappers=wrappers,
        )
        return envs

    def _setup_logger(self):
        args = self.args
        if args.track:
            import wandb

            config = vars(args)
            env_kwargs = dict(
                control_mode=args.control_mode,
                reward_mode="sparse",
                obs_mode=args.obs_mode,
                render_mode="rgb_array",
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                env_horizon=args.max_episode_steps,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=config,
                name=self.run_name,
                save_code=True,
                group="DiffusionPolicy-Baselines",
                tags=[
                    "diffusion_policy",
                    "baseline-1cam",
                    "panda",
                    "pick-cube-side-view",
                    "rgbd",
                ],
            )
        return create_logger(args)

    def _setup_dataset(self):
        args = self.args
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs=dict(shader_pack="default"),
        )
        env_kwargs["max_episode_steps"] = args.max_episode_steps

        tmp_env = gym.make(args.env_id, **env_kwargs)
        orignal_obs_space = tmp_env.observation_space
        include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
        include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
        tmp_env.close()

        obs_process_fn = partial(
            convert_obs,
            concat_fn=partial(np.concatenate, axis=-1),
            transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
            state_obs_extractor=build_state_obs_extractor(args.env_id),
            depth=include_depth,
        )

        trajectories = load_demo_dataset(args.demo_path, num_traj=1, concat=False)
        print(
            "trajectories['observations'][0] keys:",
            trajectories["observations"][0].keys(),
        )
        print(
            "sensor_data keys:",
            trajectories["observations"][0].get("sensor_data", {}).keys(),
        )

        obs_traj_dict = trajectories["observations"][0]
        print("BEFORE reorder_keys:", obs_traj_dict.keys())

        _obs_traj_dict = reorder_keys(obs_traj_dict, orignal_obs_space)
        print("AFTER reorder_keys:", _obs_traj_dict.keys())

        _obs_traj_dict = obs_process_fn(_obs_traj_dict)
        print("AFTER obs_process_fn:", _obs_traj_dict.keys())

        dataset = SmallDemoDataset_DiffusionPolicy(
            data_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=orignal_obs_space,
            include_rgb=include_rgb,
            include_depth=include_depth,
            num_traj=args.num_demos,
            device=self.device,
            control_mode=args.control_mode,
            args=args,
        )

        sampler = RandomSampler(dataset, replacement=False)
        batch_sampler = BatchSampler(
            sampler, batch_size=args.batch_size, drop_last=True
        )
        batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_dataload_workers,
            worker_init_fn=lambda worker_id: worker_init_fn(
                worker_id, base_seed=args.seed
            ),
            persistent_workers=(args.num_dataload_workers > 0),
        )

        with h5py.File(args.demo_path, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj")]
            demo_cameras = list(f[traj_keys[0]]["obs/sensor_data"].keys())
        print(f"Demo has cameras: {demo_cameras}")

        return dataset, train_dataloader

    def _setup_agent(self):
        args = self.args
        agent = Agent(self.envs, args).to(self.device)
        optimizer = optim.AdamW(
            params=agent.parameters(),
            lr=args.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
        )
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=args.total_iters,
        )
        ema = EMAModel(parameters=agent.parameters(), power=0.75)
        ema_agent = Agent(self.envs, args).to(self.device)
        return agent, ema_agent, optimizer, lr_scheduler, ema

    def save_ckpt(self, tag):
        os.makedirs(f"runs/{self.run_name}/checkpoints", exist_ok=True)
        self.ema.copy_to(self.ema_agent.parameters())
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "ema_agent": self.ema_agent.state_dict(),
            },
            f"runs/{self.run_name}/checkpoints/{tag}.pt",
        )

    def log_metrics(self, iteration, total_loss):
        if iteration % self.args.log_freq == 0:
            metrics = {
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/loss": total_loss.item(),
            }
            for k, v in self.timings.items():
                metrics[f"time/{k}"] = v
            self.logger.log(metrics, step=iteration)

    def evaluate_and_save_best(self, iteration):
        if iteration % self.args.eval_freq == 0:
            last_tick = time.time()
            self.ema.copy_to(self.ema_agent.parameters())
            eval_metrics = evaluate(
                self.args.num_eval_episodes,
                self.ema_agent,
                self.envs,
                self.device,
                self.args.sim_backend,
            )
            self.timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")

            log_dict = {}
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                log_dict[f"eval/{k}"] = eval_metrics[k]
                print(f"{k}: {eval_metrics[k]:.4f}")

            self.logger.log(log_dict, step=iteration)

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > self.best_eval_metrics[k]:
                    self.best_eval_metrics[k] = eval_metrics[k]
                    self.save_ckpt(f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

    def _run_training_loop(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.agent.train()
        pbar = tqdm(total=self.args.total_iters)
        last_tick = time.time()
        total_loss = None
        for iteration, data_batch in enumerate(self.train_dataloader):
            self.timings["data_loading"] += time.time() - last_tick

            obs_batch = {
                k: v.to(self.device) for k, v in data_batch["observations"].items()
            }
            action_batch = data_batch["actions"].to(self.device)

            last_tick = time.time()
            total_loss = self.agent.compute_loss(
                obs_seq=obs_batch,
                action_seq=action_batch,
            )
            self.timings["forward"] += time.time() - last_tick

            last_tick = time.time()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.timings["backward"] += time.time() - last_tick

            last_tick = time.time()
            self.ema.step(self.agent.parameters())
            self.timings["ema"] += time.time() - last_tick

            self.evaluate_and_save_best(iteration)
            self.log_metrics(iteration, total_loss)

            if (
                self.args.save_freq is not None
                and iteration % self.args.save_freq == 0
            ):
                self.save_ckpt(str(iteration))
            pbar.update(1)
            pbar.set_postfix({"loss": total_loss.item()})
            last_tick = time.time()

        if total_loss is not None:
            self.evaluate_and_save_best(self.args.total_iters)
            self.log_metrics(self.args.total_iters, total_loss)

        self.envs.close()
