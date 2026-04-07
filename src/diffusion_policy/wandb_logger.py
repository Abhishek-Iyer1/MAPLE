# src/diffusion_policy/wandb_logger.py
"""Comprehensive wandb logger for diffusion policy training."""

import os
import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, fields
from collections import defaultdict
import json


class WandbLogger:
    """
    Comprehensive logger for diffusion policy training.
    
    Features:
    - Automatic config logging
    - Train/val loss tracking with running averages
    - Evaluation metrics logging
    - Video logging
    - Model checkpointing to wandb
    - Summary statistics
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        log_dir: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            entity: WandB entity (username or team)
            run_name: Name for this run
            config: Configuration dictionary to log
            enabled: Whether logging is enabled
            log_dir: Directory for local logs
            tags: List of tags for the run
            notes: Notes for the run
        """
        self.enabled = enabled
        self._step = 0
        self._epoch = 0
        
        # Running averages for losses
        self._train_losses = []
        self._val_losses = []
        self._loss_components = defaultdict(list)
        
        # Best metrics tracking
        self._best_metrics = {
            'best_train_loss': float('inf'),
            'best_val_loss': float('inf'),
            'best_eval_success_rate': 0.0,
            'best_eval_success_once': 0.0,
        }
        
        if not enabled:
            self.run = None
            return
        
        # Initialize wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            dir=log_dir,
            tags=tags,
            notes=notes,
            reinit=True,
        )
        
        # Define custom charts
        self._define_custom_charts()
    
    def _define_custom_charts(self):
        """Define custom wandb charts for better visualization."""
        if not self.enabled:
            return
        
        # Define a combined loss chart
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("eval/success_rate", summary="max")
        wandb.define_metric("eval/success_once", summary="max")
    
    @classmethod
    def from_args(cls, args, extra_config: Optional[Dict] = None) -> 'WandbLogger':
        """
        Create logger from training arguments dataclass.
        
        Args:
            args: Training arguments dataclass (e.g., TrainArgs)
            extra_config: Additional config to merge
        """
        # Convert dataclass to dict
        if hasattr(args, '__dataclass_fields__'):
            config = asdict(args)
        else:
            config = vars(args) if hasattr(args, '__dict__') else {}
        
        # Merge extra config
        if extra_config:
            config.update(extra_config)
        
        # Extract relevant fields
        project = getattr(args, 'wandb_project_name', 'diffusion-policy')
        entity = getattr(args, 'wandb_entity', None)
        enabled = getattr(args, 'track', False)
        
        # Create run name from key parameters
        env_id = getattr(args, 'env_id', 'unknown')
        seed = getattr(args, 'seed', 0)
        run_name = f"{env_id}_seed{seed}"
        
        # Generate tags
        tags = [
            env_id,
            getattr(args, 'control_mode', 'unknown'),
            getattr(args, 'obs_mode', 'unknown'),
        ]
        
        # Generate notes
        notes = f"Training diffusion policy on {env_id}"
        if hasattr(args, 'num_demos'):
            notes += f" with {args.num_demos} demos"
        
        return cls(
            project=project,
            entity=entity,
            run_name=run_name,
            config=config,
            enabled=enabled,
            tags=tags,
            notes=notes,
        )
    
    def log_config(self, config: Dict[str, Any]):
        """Log additional configuration after initialization."""
        if not self.enabled or self.run is None:
            return
        
        wandb.config.update(config, allow_val_change=True)
    
    def log_model_summary(
        self,
        model: torch.nn.Module,
        input_shapes: Optional[Dict[str, tuple]] = None
    ):
        """Log model architecture summary."""
        if not self.enabled:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
            'model/total_params_M': total_params / 1e6,
        }
        
        if input_shapes:
            summary['model/input_shapes'] = str(input_shapes)
        
        wandb.config.update(summary)
        self.log(summary, step=0)
    
    def log_dataset_info(
        self,
        num_trajectories: int,
        num_transitions: int,
        obs_keys: List[str],
        action_dim: int,
        state_dim: Optional[int] = None,
    ):
        """Log dataset information."""
        if not self.enabled:
            return
        
        info = {
            'dataset/num_trajectories': num_trajectories,
            'dataset/num_transitions': num_transitions,
            'dataset/obs_keys': obs_keys,
            'dataset/action_dim': action_dim,
        }
        
        if state_dim is not None:
            info['dataset/state_dim'] = state_dim
        
        wandb.config.update(info)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log arbitrary metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Global step (uses internal counter if None)
        """
        if not self.enabled or self.run is None:
            return
        
        if step is None:
            step = self._step
        
        # Convert tensors to Python scalars
        processed = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            processed[k] = v
        
        wandb.log(processed, step=step)
    
    def log_train_loss(
        self,
        loss: float,
        step: int,
        loss_components: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """
        Log training loss with optional components.
        
        Args:
            loss: Total training loss
            step: Global step
            loss_components: Optional breakdown of loss (e.g., mse, kl, etc.)
            lr: Current learning rate
        """
        self._step = step
        self._train_losses.append(loss)
        
        # Track best
        if loss < self._best_metrics['best_train_loss']:
            self._best_metrics['best_train_loss'] = loss
        
        metrics = {
            'train/loss': loss,
            'train/loss_avg_100': np.mean(self._train_losses[-100:]),
        }
        
        if loss_components:
            for name, value in loss_components.items():
                metrics[f'train/loss_{name}'] = value
                self._loss_components[name].append(value)
        
        if lr is not None:
            metrics['train/learning_rate'] = lr
        
        self.log(metrics, step=step)
    
    def log_val_loss(self, loss: float, step: int):
        """Log validation loss."""
        self._step = step
        self._val_losses.append(loss)
        
        # Track best
        if loss < self._best_metrics['best_val_loss']:
            self._best_metrics['best_val_loss'] = loss
        
        metrics = {
            'val/loss': loss,
            'val/loss_avg': np.mean(self._val_losses[-10:]) if self._val_losses else loss,
        }
        
        self.log(metrics, step=step)
    
    def log_evaluation(
        self,
        step: int,
        success_rate: float,
        success_once: float,
        episode_rewards: Optional[List[float]] = None,
        episode_lengths: Optional[List[int]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log evaluation metrics.
        
        Args:
            step: Global step
            success_rate: Success rate at end of episode
            success_once: Success at any point during episode
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            extra_metrics: Additional evaluation metrics
        """
        self._step = step
        
        # Track best
        if success_rate > self._best_metrics['best_eval_success_rate']:
            self._best_metrics['best_eval_success_rate'] = success_rate
        if success_once > self._best_metrics['best_eval_success_once']:
            self._best_metrics['best_eval_success_once'] = success_once
        
        metrics = {
            'eval/success_rate': success_rate,
            'eval/success_once': success_once,
            'eval/best_success_rate': self._best_metrics['best_eval_success_rate'],
            'eval/best_success_once': self._best_metrics['best_eval_success_once'],
        }
        
        if episode_rewards:
            metrics.update({
                'eval/reward_mean': np.mean(episode_rewards),
                'eval/reward_std': np.std(episode_rewards),
                'eval/reward_min': np.min(episode_rewards),
                'eval/reward_max': np.max(episode_rewards),
            })
        
        if episode_lengths:
            metrics.update({
                'eval/episode_length_mean': np.mean(episode_lengths),
                'eval/episode_length_std': np.std(episode_lengths),
            })
        
        if extra_metrics:
            for name, value in extra_metrics.items():
                metrics[f'eval/{name}'] = value
        
        self.log(metrics, step=step)
        
        return metrics
    
    def log_video(
        self,
        video_path: str,
        step: int,
        tag: str = "eval/video",
        fps: int = 30,
    ):
        """Log a video file."""
        if not self.enabled or self.run is None:
            return
        
        if os.path.exists(video_path):
            wandb.log({tag: wandb.Video(video_path, fps=fps, format="mp4")}, step=step)
    
    def log_videos(
        self,
        video_paths: List[str],
        step: int,
        tag_prefix: str = "eval/video",
    ):
        """Log multiple videos."""
        if not self.enabled or self.run is None:
            return
        
        videos = {}
        for i, path in enumerate(video_paths):
            if os.path.exists(path):
                videos[f"{tag_prefix}_{i}"] = wandb.Video(path, fps=30, format="mp4")
        
        if videos:
            wandb.log(videos, step=step)
    
    def log_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        step: int,
        tag: str = "image",
        caption: Optional[str] = None,
    ):
        """Log an image."""
        if not self.enabled or self.run is None:
            return
        
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        wandb.log({tag: wandb.Image(image, caption=caption)}, step=step)
    
    def log_histogram(
        self,
        values: Union[np.ndarray, torch.Tensor, List],
        step: int,
        tag: str = "histogram",
    ):
        """Log a histogram of values."""
        if not self.enabled or self.run is None:
            return
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, list):
            values = np.array(values)
        
        wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        metadata: Optional[Dict] = None,
    ):
        """Log model checkpoint as artifact."""
        if not self.enabled or self.run is None:
            return
        
        artifact = wandb.Artifact(
            name=f"checkpoint-{step}",
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def log_best_checkpoint(self, checkpoint_path: str, metric_name: str, metric_value: float):
        """Log best model checkpoint."""
        if not self.enabled or self.run is None:
            return
        
        artifact = wandb.Artifact(
            name=f"best-{metric_name}",
            type="model",
            metadata={
                metric_name: metric_value,
                'step': self._step,
            },
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def set_step(self, step: int):
        """Set the current global step."""
        self._step = step
    
    def set_epoch(self, epoch: int):
        """Set the current epoch."""
        self._epoch = epoch
        if self.enabled:
            self.log({'train/epoch': epoch}, step=self._step)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get dictionary of best metrics seen during training."""
        return self._best_metrics.copy()
    
    def log_summary(self):
        """Log final summary statistics."""
        if not self.enabled or self.run is None:
            return
        
        # Log best metrics to summary
        for name, value in self._best_metrics.items():
            wandb.run.summary[name] = value
        
        # Log final averages
        if self._train_losses:
            wandb.run.summary['final_train_loss_avg'] = np.mean(self._train_losses[-100:])
        if self._val_losses:
            wandb.run.summary['final_val_loss_avg'] = np.mean(self._val_losses[-10:])
    
    def finish(self):
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return
        
        self.log_summary()
        wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


# Convenience function for backwards compatibility
def create_logger(args) -> WandbLogger:
    """Create and initialize wandb logger from args."""
    return WandbLogger.from_args(args)