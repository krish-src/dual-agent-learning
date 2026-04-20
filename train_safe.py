"""
CARLA Reinforcement Learning Training Script (IMPROVED)
========================================================

Trains a driving agent in CARLA using SAC (default) or PPO with safe model saving.

IMPROVEMENTS:
- Atomic model saving (prevents corruption)
- Model validation after save
- Checkpoint integrity verification
- Graceful shutdown handling
- Safe final model save

Usage:
    python train_safe.py                     # Train SAC headless (fastest)
    python train_safe.py --algo ppo          # Train with PPO instead
    python train_safe.py --rendering         # Enable CARLA server-side rendering
    python train_safe.py --load model.zip    # Resume from checkpoint
"""

import argparse
import glob
import os
import signal
import sys

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
from model_utils import safe_model_save, validate_model_file


# ==============================================================================
# -- Custom CNN + State Feature Extractor --------------------------------------
# ==============================================================================

class CarlaCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Dict observation space.
    (Same as original training script)
    """

    def __init__(self, observation_space: gymnasium.spaces.Dict):
        features_dim = 320
        super().__init__(observation_space, features_dim=features_dim)

        cam_shape = observation_space['camera'].shape
        n_input_channels = cam_shape[0] + 1  # RGB (3) + seg (1) = 4

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_img = torch.zeros(1, n_input_channels, cam_shape[1], cam_shape[2])
            cnn_output_size = self.cnn(sample_img).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_output_size, 256),
            nn.ReLU(),
        )

        state_size = observation_space['state'].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        camera = observations['camera']
        seg = observations['seg_camera'].float() / 12.0
        if seg.dim() == 3:
            seg = seg.unsqueeze(1)

        combined = torch.cat([camera, seg], dim=1)
        cnn_features = self.cnn_linear(self.cnn(combined))

        state = observations['state'].float()
        state_features = self.state_mlp(state)

        return torch.cat([cnn_features, state_features], dim=1)


# ==============================================================================
# -- Safe Checkpoint Callback --------------------------------------------------
# ==============================================================================

class SafeCheckpointCallback(CheckpointCallback):
    """
    Extended checkpoint callback with validation.
    
    Validates each checkpoint after saving to ensure integrity.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_count = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Standard checkpoint save
            checkpoint_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            
            # Save using safe method
            print(f"\n[Checkpoint] Saving at step {self.num_timesteps}...")
            if safe_model_save(self.model, checkpoint_path, backup=False):
                self.save_count += 1
                print(f"[Checkpoint] ✓ Saved successfully ({self.save_count} total)")
            else:
                print(f"[Checkpoint] ✗ Save failed!")
        
        return True


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

# Global variables for graceful shutdown
MODEL_INSTANCE = None
ENV_INSTANCE = None
FINAL_SAVE_PATH = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully by saving model before exit."""
    print("\n" + "="*80)
    print("Training interrupted by user (Ctrl+C)")
    print("="*80)
    
    if MODEL_INSTANCE is not None and FINAL_SAVE_PATH is not None:
        print("\nAttempting to save model before exit...")
        if safe_model_save(MODEL_INSTANCE, FINAL_SAVE_PATH, backup=True):
            print("✓ Model saved successfully")
        else:
            print("✗ Failed to save model")
            
            # Try to save to emergency backup
            emergency_path = FINAL_SAVE_PATH.replace('.zip', '_emergency.zip')
            print(f"\nTrying emergency save to: {emergency_path}")
            try:
                MODEL_INSTANCE.save(emergency_path)
                print("✓ Emergency save completed")
            except Exception as e:
                print(f"✗ Emergency save failed: {e}")
    
    if ENV_INSTANCE is not None:
        print("\nClosing environment...")
        try:
            ENV_INSTANCE.close()
            print("✓ Environment closed")
        except:
            pass
    
    print("\nExiting...")
    sys.exit(0)


def train(args):
    global MODEL_INSTANCE, ENV_INSTANCE, FINAL_SAVE_PATH
    
    algo_name = args.algo.upper()
    print('=' * 60)
    print(f'  CARLA RL Training - {algo_name} (SAFE MODE)')
    print('=' * 60)
    print()

    # Create environment
    no_rendering = not args.rendering
    env = CarlaRLEnv(
        host=args.host,
        port=args.port,
        num_npc_vehicles=args.npc_vehicles,
        num_npc_walkers=args.npc_walkers,
        no_rendering=no_rendering,
    )
    ENV_INSTANCE = env  # Store for signal handler

    print(f'Environment created')
    print(f'  NPC traffic: {args.npc_vehicles} vehicles, {args.npc_walkers} walkers')
    print(f'  CARLA no_rendering_mode: {no_rendering}')
    print(f'  Observation: camera {env.observation_space["camera"].shape} + '
          f'seg {env.observation_space["seg_camera"].shape} + '
          f'state {env.observation_space["state"].shape}')
    print(f'  Action: {env.action_space.shape} [steer, accel]')
    print()

    # Auto-find latest checkpoint if --resume is set
    if args.resume and not args.load:
        pattern = f'./checkpoints/carla_{args.algo}_*_steps.zip'
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        # Validate checkpoints and find latest valid one
        valid_checkpoints = [cp for cp in checkpoints if validate_model_file(cp, verbose=False)]
        
        if valid_checkpoints:
            args.load = valid_checkpoints[-1]
            print(f'Resuming from latest valid checkpoint: {args.load}')
        else:
            # Try the final model as fallback
            final = f'./carla_{args.algo}_final.zip'
            if os.path.exists(final) and validate_model_file(final, verbose=False):
                args.load = final
                print(f'Resuming from final model: {args.load}')
            else:
                print(f'No valid checkpoints found, starting fresh')
    
    if args.load and not validate_model_file(args.load, verbose=True):
        print(f"\n❌ ERROR: Cannot load from invalid model file: {args.load}")
        print("Starting fresh training instead...\n")
        args.load = None
    
    print()

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CarlaCNNExtractor,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU,
    )

    # Create or load model
    if args.algo == 'sac':
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
                buffer_size=30_000,
                learning_starts=500,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=4,
                gradient_steps=4,
                ent_coef='auto',
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
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log='./tensorboard_logs',
                verbose=0,
                device=args.device,
            )

    MODEL_INSTANCE = model  # Store for signal handler
    FINAL_SAVE_PATH = f'./carla_{args.algo}_final.zip'

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f'Model: {algo_name} with {total_params:,} parameters')
    print(f'Device: {args.device}')
    print(f'Training for {args.timesteps:,} timesteps')
    if args.algo == 'sac':
        print(f'Replay buffer: {model.buffer_size:,} transitions')
    print()

    # Callbacks
    callbacks = [TrainingLogCallback(verbose=1)]

    # Safe checkpoint callback
    os.makedirs('./checkpoints', exist_ok=True)
    callbacks.append(SafeCheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix=f'carla_{args.algo}',
    ))

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Train
    print('Starting training...')
    print('Safe mode enabled: Model will be validated after each save')
    print('Press Ctrl+C to interrupt and save current progress')
    print('-' * 60)
    
    training_successful = False
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        training_successful = True
        
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        
    except Exception as e:
        print(f'\n❌ Training failed with error: {e}')
        import traceback
        traceback.print_exc()
        
    finally:
        # Save final model using safe method
        print('\n' + '='*80)
        print('Saving final model...')
        print('='*80)
        
        if safe_model_save(model, FINAL_SAVE_PATH, backup=True):
            print(f'\n✓ Final model saved successfully: {FINAL_SAVE_PATH}')
            
            # Verify one more time
            if validate_model_file(FINAL_SAVE_PATH, verbose=True):
                print('\n✓ Model verified and ready for testing')
                print(f'\nTo test this model, run:')
                print(f'  python test_with_explainability.py --load {FINAL_SAVE_PATH}')
            else:
                print('\n⚠ WARNING: Final validation failed!')
                print('Model may be corrupted. Check checkpoint files instead.')
        else:
            print(f'\n❌ Failed to save final model!')
            print('Attempting emergency save...')
            
            emergency_path = FINAL_SAVE_PATH.replace('.zip', '_emergency.zip')
            try:
                model.save(emergency_path)
                print(f'✓ Emergency save completed: {emergency_path}')
                print('⚠ WARNING: Emergency save not validated, may be corrupted')
            except Exception as e:
                print(f'❌ Emergency save also failed: {e}')
        
        # Close environment
        print('\nClosing environment...')
        env.close()
        print('✓ Environment closed')
        
        if training_successful:
            print('\n' + '='*80)
            print('Training completed successfully!')
            print('='*80)


# ==============================================================================
# -- Entry Point ---------------------------------------------------------------
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent in CARLA (Safe Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SAFE MODE Features:
  - Atomic model saving (prevents corruption)
  - Checkpoint validation
  - Graceful shutdown (Ctrl+C saves model)
  - Emergency backup on failure
        """)
    
    parser.add_argument('--algo', type=str, default='sac',
                        choices=['sac', 'ppo'],
                        help='RL algorithm: sac (default) or ppo')
    parser.add_argument('--host', default='127.0.0.1',
                        help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total training timesteps')
    parser.add_argument('--rendering', action='store_true',
                        help='Enable CARLA rendering')
    parser.add_argument('--resume', action='store_true',
                        help='Auto-load latest valid checkpoint')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to specific checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    parser.add_argument('--npc-vehicles', type=int, default=15,
                        help='Number of NPC vehicles')
    parser.add_argument('--npc-walkers', type=int, default=5,
                        help='Number of NPC pedestrians')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
