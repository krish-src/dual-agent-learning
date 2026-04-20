"""
Configuration file for CARLA RL Explainability System
======================================================

This file contains all configurable parameters for the system.
Import and modify as needed for your experiments.
"""

# ==============================================================================
# Training Configuration
# ==============================================================================

TRAINING_CONFIG = {
    # Algorithm
    'algorithm': 'sac',  # 'sac' or 'ppo'
    'timesteps': 500_000,
    'device': 'auto',  # 'auto', 'cpu', or 'cuda'
    
    # Environment
    'carla_host': '127.0.0.1',
    'carla_port': 2000,
    'npc_vehicles': 15,
    'npc_walkers': 5,
    'no_rendering': True,  # Headless for speed
    
    # Checkpointing
    'checkpoint_freq': 10_000,
    'checkpoint_dir': './checkpoints/',
    'resume_from_latest': False,
    'load_checkpoint': None,  # Path to specific checkpoint
}

# ==============================================================================
# SAC Hyperparameters
# ==============================================================================

SAC_CONFIG = {
    'learning_rate': 3e-4,
    'buffer_size': 30_000,
    'learning_starts': 500,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 4,
    'gradient_steps': 4,
    'ent_coef': 'auto',
}

# ==============================================================================
# PPO Hyperparameters
# ==============================================================================

PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 1024,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

# ==============================================================================
# Network Architecture
# ==============================================================================

NETWORK_CONFIG = {
    # Feature extractor
    'cnn_channels': [32, 64, 64],
    'cnn_kernels': [8, 4, 3],
    'cnn_strides': [4, 2, 1],
    'cnn_output_dim': 256,
    'state_mlp_dim': 64,
    'features_dim': 320,  # cnn_output_dim + state_mlp_dim
    
    # Policy and value networks
    'policy_layers': [128, 64],
    'value_layers': [128, 64],
    'activation': 'relu',
}

# ==============================================================================
# Explainability Configuration
# ==============================================================================

EXPLAINABILITY_CONFIG = {
    # Generation frequency
    'explain_every_n_steps': 10,  # Generate explanation every N steps
    
    # Methods
    'compute_state_importance': True,
    'compute_visual_attention': True,
    'compute_confidence': True,
    'generate_text': True,
    
    # Importance thresholds
    'importance_threshold': 0.3,  # Only mention factors above this
    'num_top_factors': 3,  # Number of factors to mention
    
    # Uncertainty thresholds
    'high_uncertainty_threshold': 1.0,
    'low_uncertainty_threshold': 0.3,
    
    # History
    'history_length': 100,  # Number of explanations to keep
}

# ==============================================================================
# UI Configuration
# ==============================================================================

UI_CONFIG = {
    # Refresh rate
    'default_refresh_rate': 1.0,  # seconds
    'min_refresh_rate': 0.5,
    'max_refresh_rate': 5.0,
    
    # Display options
    'show_camera': True,
    'show_segmentation': False,
    'show_attention_heatmap': True,
    'show_state_importance': True,
    'show_confidence_metrics': True,
    
    # Chart styling
    'attention_colorscale': 'Hot',
    'importance_colorscale': 'Viridis',
    'chart_height': 400,
}

# ==============================================================================
# State Variable Names and Descriptions
# ==============================================================================

STATE_VARIABLES = {
    'speed': {
        'name': 'Speed',
        'unit': 'km/h',
        'description': 'Current vehicle speed',
        'range': [0, 100],
    },
    'lane_offset': {
        'name': 'Lane Offset',
        'unit': 'm',
        'description': 'Distance from lane center',
        'range': [-2, 2],
    },
    'heading_error': {
        'name': 'Heading Error',
        'unit': 'rad',
        'description': 'Angle difference from road direction',
        'range': [-3.14, 3.14],
    },
    'junction_flag': {
        'name': 'Junction',
        'unit': 'bool',
        'description': 'Whether approaching intersection',
        'range': [0, 1],
    },
    'radar_distance': {
        'name': 'Radar Distance',
        'unit': 'm',
        'description': 'Distance to nearest vehicle ahead',
        'range': [0, 100],
    },
    'radar_speed': {
        'name': 'Radar Speed',
        'unit': 'km/h',
        'description': 'Speed of nearest vehicle ahead',
        'range': [0, 100],
    },
    'traffic_light_state': {
        'name': 'Traffic Light',
        'unit': 'enum',
        'description': 'Traffic light state (-1=red, 0=yellow, 1=green)',
        'range': [-1, 1],
    },
    'speed_limit_norm': {
        'name': 'Speed Limit',
        'unit': 'normalized',
        'description': 'Normalized speed limit',
        'range': [0, 1],
    },
}

# ==============================================================================
# Explanation Templates
# ==============================================================================

EXPLANATION_TEMPLATES = {
    # Traffic light
    'red_light': "red traffic light detected",
    'yellow_light': "yellow traffic light (preparing to stop)",
    'green_light': "green traffic light (safe to proceed)",
    
    # Vehicle ahead
    'close_vehicle': "vehicle ahead at {distance:.1f}m",
    'following_vehicle': "following vehicle at {distance:.1f}m",
    
    # Lane position
    'lane_left': "correcting lane position (left offset)",
    'lane_right': "correcting lane position (right offset)",
    'lane_centered': "maintaining center of lane",
    
    # Heading
    'heading_correction': "adjusting heading to follow road",
    'sharp_turn': "navigating sharp turn",
    
    # Speed
    'low_speed': "low speed requires acceleration",
    'high_speed': "high speed requires braking",
    'speed_limit': "approaching speed limit",
    
    # Junction
    'approaching_junction': "approaching intersection",
    'in_junction': "navigating through intersection",
    
    # Confidence
    'high_confidence': "High confidence in decision",
    'low_confidence': "Moderate confidence in decision",
    'high_uncertainty': "High uncertainty in decision (±{uncertainty:.1f})",
}

# ==============================================================================
# Logging Configuration
# ==============================================================================

LOGGING_CONFIG = {
    'tensorboard_log_dir': './tensorboard_logs',
    'console_log_level': 'INFO',
    'file_log_level': 'DEBUG',
    'log_file': './training.log',
    
    # What to log
    'log_episodes': True,
    'log_checkpoints': True,
    'log_explanations': False,  # Can generate large logs
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_config(algorithm: str) -> dict:
    """Get complete configuration for specified algorithm."""
    config = TRAINING_CONFIG.copy()
    config.update(NETWORK_CONFIG)
    config.update(EXPLAINABILITY_CONFIG)
    config.update(UI_CONFIG)
    config.update(LOGGING_CONFIG)
    
    if algorithm.lower() == 'sac':
        config.update(SAC_CONFIG)
    elif algorithm.lower() == 'ppo':
        config.update(PPO_CONFIG)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return config


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    # Check required keys
    required_keys = ['algorithm', 'timesteps', 'device']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check value ranges
    if config['timesteps'] <= 0:
        raise ValueError("timesteps must be positive")
    
    if config['explain_every_n_steps'] <= 0:
        raise ValueError("explain_every_n_steps must be positive")
    
    if not 0 < config['default_refresh_rate'] <= 10:
        raise ValueError("default_refresh_rate must be in (0, 10]")
    
    return True


def print_config(config: dict):
    """Pretty-print configuration."""
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    for key, value in sorted(config.items()):
        print(f"  {key:30s} = {value}")
    print("=" * 60)


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == '__main__':
    # Get SAC configuration
    config = get_config('sac')
    
    # Validate
    validate_config(config)
    
    # Print
    print_config(config)
    
    # Modify for your experiment
    config['timesteps'] = 1_000_000
    config['explain_every_n_steps'] = 5
    config['npc_vehicles'] = 30
    
    print("\nModified configuration:")
    print_config(config)
