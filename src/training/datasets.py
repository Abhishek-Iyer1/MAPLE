import h5py
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from src.args import Args
from src.training.training_utils import reorder_keys

class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj, control_mode, args:Args):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.device = device
        self.control_mode = control_mode
        
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process observations
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.from_numpy(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).float()
            
            obs_traj_dict_list.append(_obs_traj_dict)
        
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        
        # Pre-process actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.from_numpy(
                trajectories["actions"][i]
            ).float()
        
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

        # Padding setup based on control mode
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            print("Detected a delta controller type, padding with zero action.")
            self.pad_action_arm = torch.zeros((trajectories["actions"][0].shape[1] - 1,))
            self.pad_with_last = False
        elif "pd_joint_pos" in control_mode:
            print("Detected pd_joint_pos controller, padding with last action (hold position).")
            self.pad_action_arm = None
            self.pad_with_last = True
        else:
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        
        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        
        # Pre-compute slices
        self.slices = []
        total_transitions = 0
        num_traj = len(trajectories["actions"])
        
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories
        
        # Save which trajectories were used
        self.traj_keys_used = [f"traj_{i}" for i in range(num_traj)]

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start) : start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            if self.pad_with_last:
                # For pd_joint_pos: repeat last action (hold position)
                pad_action = act_seq[-1]
                act_seq = torch.cat([act_seq, pad_action.unsqueeze(0).repeat(end - L, 1)], dim=0)
            else:
                # For delta: pad with zero movement but keep gripper
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)

        assert obs_seq["state"].shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)

class LazyDemoDataset(Dataset):
    """Load trajectories from disk on-demand."""
    
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, args: Args, num_traj=None):
        self.data_path = data_path
        self.obs_process_fn = obs_process_fn
        self.obs_space = obs_space
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.device = device
        
        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        
        # Only read metadata, not actual data
        with h5py.File(data_path, 'r') as f:
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith('traj')],
                key=lambda x: int(x.split('_')[1])
            )
            if num_traj:
                traj_keys = traj_keys[:num_traj]
            
            # Store trajectory lengths and find camera names
            self.traj_lengths = []
            self.camera_names = []
            
            for i, k in enumerate(traj_keys):
                L = f[k]['actions'].shape[0]
                self.traj_lengths.append(L)
                
                # Get camera names from first trajectory
                if i == 0:
                    sensor_data = f[k]['obs/sensor_data']
                    self.camera_names = list(sensor_data.keys())
                    print(f"Found cameras: {self.camera_names}")
        
        self.traj_keys = traj_keys
        self.num_traj = len(traj_keys)
        
        # Pre-compute slices
        self.slices = []
        obs_horizon, pred_horizon = self.obs_horizon, self.pred_horizon
        
        for traj_idx, L in enumerate(self.traj_lengths):
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]
        
        # Padding for delta control
        if "delta_pos" in args.control_mode or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            with h5py.File(data_path, 'r') as f:
                act_dim = f[traj_keys[0]]['actions'].shape[1]
            self.pad_action_arm = torch.zeros((act_dim - 1,))
        else:
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        
        print(f"LazyDataset: {self.num_traj} trajectories, {len(self.slices)} slices")
        
        # Cache for recently accessed trajectories
        self._cache = {}
        self._cache_size = 10
    
    def _load_trajectory(self, traj_idx):
        """Load a single trajectory from disk in the format expected by obs_process_fn."""
        if traj_idx in self._cache:
            return self._cache[traj_idx]
        
        with h5py.File(self.data_path, 'r') as f:
            traj = f[self.traj_keys[traj_idx]]
            
            # Build observation dict in the same structure as env observations
            # This matches what convert_obs expects: obs['sensor_data'][camera_name]['rgb'/'depth']
            sensor_data = {}
            for cam_name in self.camera_names:
                sensor_data[cam_name] = {}
                if self.include_rgb:
                    sensor_data[cam_name]['rgb'] = traj[f'obs/sensor_data/{cam_name}/rgb'][:]
                if self.include_depth:
                    sensor_data[cam_name]['depth'] = traj[f'obs/sensor_data/{cam_name}/depth'][:]
            
            # Build agent observation
            agent_obs = {
                'qpos': traj['obs/agent/qpos'][:],
                'qvel': traj['obs/agent/qvel'][:],
            }
            
            # Build extra observation
            extra_obs = {}

            for key in traj["obs"]["extra"].keys():
                extra_obs[key] = traj['obs/extra'][key][:]
            
            # Build full observation dict matching env structure
            obs_dict = {
                'sensor_data': sensor_data,
                'agent': agent_obs,
                'extra': extra_obs,
            }
            
            # Load actions
            actions = traj['actions'][:]
        
        # Process observations using the same function as training
        processed_obs = self.obs_process_fn(obs_dict)
        
        # Convert to tensors (CPU)
        processed = {
            'observations': {
                'state': torch.from_numpy(processed_obs['state']).float(),
            },
            'actions': torch.from_numpy(actions).float(),
        }
        if self.include_rgb:
            processed['observations']['rgb'] = torch.from_numpy(processed_obs['rgb'])
        if self.include_depth:
            processed['observations']['depth'] = torch.from_numpy(
                processed_obs['depth'].astype(np.float32)
            ).to(torch.float16)
        
        # Update cache (LRU-style)
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[traj_idx] = processed
        
        return processed
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        traj_data = self._load_trajectory(traj_idx)
        
        obs_traj = traj_data['observations']
        actions = traj_data['actions']
        L = actions.shape[0]
        
        # Extract observation sequence
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start) : start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # Extract action sequence
        act_seq = actions[max(0, start) : end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.unsqueeze(0).repeat(end - L, 1)], dim=0)
        
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }
    
    def __len__(self):
        return len(self.slices)