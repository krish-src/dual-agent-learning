"""
CARLA Reinforcement Learning Training Script
=============================================

Trains a driving agent in CARLA using SAC (default) or PPO.
Runs headless by default for maximum training speed.

Usage:
    python train.py                     # Train SAC headless (fastest)
    python train.py --algo ppo          # Train with PPO instead
    python train.py --rendering         # Enable CARLA server-side rendering
    python train.py --load model.zip    # Resume from checkpoint

    # Monitor in browser:
    tensorboard --logdir=./tensorboard_logs

Algorithm comparison:
    SAC (default) — Off-policy, replay buffer, entropy-regularized.
        Best for continuous control (steering/accel).
        ~5-10x more sample efficient than PPO thanks to replay buffer.
        Replay buffer acts like learning from many episodes at once.
    PPO — On-policy, rollout-based. Simpler but needs more samples.
        Only learns from the most recent rollout, then discards it.

Parallelism note:
    A single CARLA server = one shared physics world, so you cannot run
    truly parallel episodes. SAC's replay buffer (30K transitions)
    compensates by learning from ALL past experience every update.
    For true parallelism, run multiple CARLA servers on different ports.
"""

import argparse
import glob
import os

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from carla_rl_env import CarlaRLEnv


# ==============================================================================
# -- Custom CNN + State Feature Extractor --------------------------------------
# ==============================================================================

class CarlaCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Dict observation space.

    Architecture:
        Camera (84x84x3) ──┐
                            ├→ CNN (4-ch input, 3 conv layers) → 256-dim feature
        SegCam (84x84x1) ──┘
        State (8,)          → MLP (1 layer)                    → 64-dim feature
        Concatenated → 320-dim feature vector → Policy & Value heads

    The CNN processes a 4-channel image: RGB (3ch) + semantic segmentation (1ch).
    The semantic seg channel gives the agent pixel-perfect labels for vehicles,
    pedestrians, road, etc. — much easier than learning to parse raw RGB.

    The MLP processes the state vector (speed, lane pos, heading, junction,
    radar distance, radar speed, traffic light state, speed limit).
    """

    def __init__(self, observation_space: gymnasium.spaces.Dict):
        # Output feature dimension
        features_dim = 320
        super().__init__(observation_space, features_dim=features_dim)

        # -- CNN for camera image + semantic segmentation (4 channels total)
        # SB3 auto-transposes images to CHW format via VecTransposeImage
        # Camera: (3, 84, 84) — but we concatenate seg as 4th channel
        cam_shape = observation_space['camera'].shape
        # cam_shape is CHW after SB3 transpose: (3, 84, 84)
        n_input_channels = cam_shape[0] + 1  # RGB (3) + seg (1) = 4

        # Standard Nature-CNN architecture (proven in Atari, works great for driving)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        with torch.no_grad():
            sample_img = torch.zeros(1, n_input_channels, cam_shape[1], cam_shape[2])
            cnn_output_size = self.cnn(sample_img).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_output_size, 256),
            nn.ReLU(),
        )

        # -- MLP for state vector (8 values: speed, lane_dist, heading, junction,
        #    radar_dist, radar_speed, traffic_light_state, speed_limit_norm)
        state_size = observation_space['state'].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Camera: SB3 already transposes to CHW and normalizes to [0,1] float
        camera = observations['camera']

        # Semantic segmentation: (batch, H, W) uint8 → normalize and add channel dim
        seg = observations['seg_camera'].float() / 12.0  # normalize to [0, 1]
        if seg.dim() == 3:
            seg = seg.unsqueeze(1)  # (batch, 1, H, W)

        # Concatenate RGB + seg → 4-channel input
        combined = torch.cat([camera, seg], dim=1)
        cnn_features = self.cnn_linear(self.cnn(combined))

        # State vector
        state = observations['state'].float()
        state_features = self.state_mlp(state)

        # Concatenate: 256 + 64 = 320
        return torch.cat([cnn_features, state_features], dim=1)


# ==============================================================================
# -- Training Callback ---------------------------------------------------------
# ==============================================================================

class TrainingLogCallback(BaseCallback):
    """Logs episode stats and prints progress to terminal."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        # Check for completed episodes
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)

                if self.verbose:
                    n = len(self._episode_rewards)
                    avg_r = np.mean(self._episode_rewards[-20:])
                    avg_l = np.mean(self._episode_lengths[-20:])
                    print(f'  Episode {n:4d} | '
                          f'Reward: {ep_reward:+8.1f} | '
                          f'Length: {ep_length:5d} | '
                          f'Avg(20): {avg_r:+8.1f} | '
                          f'AvgLen: {avg_l:6.0f}')
        return True


# ==============================================================================
# -- Main Training Function ----------------------------------------------------
# ==============================================================================

def train(args):
    algo_name = args.algo.upper()
    print('=' * 60)
    print(f'  CARLA RL Training - {algo_name}')
    print('=' * 60)
    print()

    # -- Create environment (headless by default for speed)
    no_rendering = not args.rendering
    env = CarlaRLEnv(
        host=args.host,
        port=args.port,
        num_npc_vehicles=args.npc_vehicles,
        num_npc_walkers=args.npc_walkers,
        no_rendering=no_rendering,
    )

    print(f'Environment created')
    print(f'  NPC traffic: {args.npc_vehicles} vehicles, {args.npc_walkers} walkers')
    print(f'  CARLA no_rendering_mode: {no_rendering}')
    print(f'  Observation: camera {env.observation_space["camera"].shape} + '
          f'seg {env.observation_space["seg_camera"].shape} + '
          f'state {env.observation_space["state"].shape}')
    print(f'  Action: {env.action_space.shape} '
          f'[steer, accel]')
    print()

    # -- Auto-find latest checkpoint if --resume is set
    if args.resume and not args.load:
        pattern = f'./checkpoints/carla_{args.algo}_*_steps.zip'
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        if checkpoints:
            args.load = checkpoints[-1]
            print(f'Resuming from latest checkpoint: {args.load}')
        else:
            # Try the final model as fallback
            final = f'./carla_{args.algo}_final.zip'
            if os.path.exists(final):
                args.load = final
                print(f'Resuming from final model: {args.load}')
            else:
                print(f'No checkpoints found, starting fresh')
    print()

    # -- Policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CarlaCNNExtractor,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU,
    )

    # -- Create or load model
    if args.algo == 'sac':
        # SAC-specific policy kwargs (SAC uses 'qf' instead of 'vf')
        policy_kwargs_sac = dict(
            features_extractor_class=CarlaCNNExtractor,
            net_arch=dict(pi=[128, 64], qf=[128, 64]),
            activation_fn=nn.ReLU,
        )

        if args.load:
            print(f'Loading model from {args.load}')
            model = SAC.load(args.load, env=env, device=args.device,
                             tensorboard_log='./tensorboard_logs')
        else:
            model = SAC(
                policy='MultiInputPolicy',
                env=env,
                learning_rate=3e-4,
                buffer_size=30_000,       # Replay buffer — sized to fit in 8GB Docker RAM
                learning_starts=500,      # Start learning after 500 transitions (~8 episodes)
                batch_size=256,           # Larger batches for off-policy stability
                tau=0.005,                # Soft target update coefficient
                gamma=0.99,               # Discount factor
                train_freq=4,             # Collect 4 steps then update (less overhead)
                gradient_steps=4,         # 4 gradient steps per update (amortize cost)
                ent_coef='auto',          # Auto-tune entropy (SAC's key feature)
                policy_kwargs=policy_kwargs_sac,
                tensorboard_log='./tensorboard_logs',
                verbose=0,
                device=args.device,
            )
    else:
        # PPO
        if args.load:
            print(f'Loading model from {args.load}')
            model = PPO.load(args.load, env=env, device=args.device,
                             tensorboard_log='./tensorboard_logs')
        else:
            model = PPO(
                policy='MultiInputPolicy',
                env=env,
                learning_rate=3e-4,
                n_steps=1024,             # Steps per rollout before update
                batch_size=64,            # Minibatch size for SGD
                n_epochs=10,              # Epochs per PPO update
                gamma=0.99,               # Discount factor
                gae_lambda=0.95,          # GAE lambda
                clip_range=0.2,           # PPO clip range
                ent_coef=0.01,            # Entropy bonus (exploration)
                vf_coef=0.5,              # Value function loss weight
                max_grad_norm=0.5,        # Gradient clipping
                policy_kwargs=policy_kwargs,
                tensorboard_log='./tensorboard_logs',
                verbose=0,
                device=args.device,
            )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f'Model: {algo_name} with {total_params:,} parameters')
    print(f'Device: {args.device}')
    print(f'Training for {args.timesteps:,} timesteps')
    if args.algo == 'sac':
        print(f'Replay buffer: {model.buffer_size:,} transitions '
              f'(acts like learning from many episodes at once)')
    print()

    # -- Callbacks
    callbacks = [TrainingLogCallback(verbose=1)]

    # Save checkpoint every N steps
    os.makedirs('./checkpoints', exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix=f'carla_{args.algo}',
    ))

    # -- Train
    print('Starting training...')
    print('-' * 60)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    finally:
        # Save final model
        save_path = f'./carla_{args.algo}_final.zip'
        model.save(save_path)
        print(f'\nModel saved to {save_path}')
        env.close()


# ==============================================================================
# -- Entry Point ---------------------------------------------------------------
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent in CARLA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm notes:
  sac   Soft Actor-Critic (default). Off-policy with replay buffer.
        Best for continuous control. ~5-10x more sample efficient.
        The replay buffer (100K transitions) acts like learning from
        many episodes at once — no need for parallel environments.

  ppo   Proximal Policy Optimization. On-policy, rollout-based.
        Simpler but less sample efficient (discards experience after each update).

Parallelism:
  A single CARLA server = one physics world = one episode at a time.
  SAC compensates with its replay buffer. For true parallelism, run
  multiple CARLA servers on different ports (e.g. 2000, 2002, 2004).
        """)
    parser.add_argument('--algo', type=str, default='sac',
                        choices=['sac', 'ppo'],
                        help='RL algorithm: sac (default) or ppo')
    parser.add_argument('--host', default='127.0.0.1',
                        help='CARLA server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--rendering', action='store_true',
                        help='Enable CARLA server-side rendering (slower, for debugging)')
    parser.add_argument('--resume', action='store_true',
                        help='Auto-load latest checkpoint and continue training')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to specific model checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda (default: auto)')
    parser.add_argument('--npc-vehicles', type=int, default=15,
                        help='Number of NPC traffic cars (default: 15)')
    parser.add_argument('--npc-walkers', type=int, default=5,
                        help='Number of NPC pedestrians (default: 5)')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
