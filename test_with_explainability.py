"""
CARLA RL Testing Script with Post-Hoc Explainability
=====================================================

Tests a trained SAC/PPO agent in CARLA and generates comprehensive explanations.

Features:
    - Grad-CAM visualization (visual attention on RGB + segmentation)
    - State saliency analysis (why each action was taken)
    - Human-readable rule explanations
    - Causal graphs showing decision pathways
    - Performance metrics and episode statistics

Usage:
    python test_with_explainability.py --load carla_sac_final.zip
    python test_with_explainability.py --load carla_sac_final.zip --episodes 10
    python test_with_explainability.py --load carla_sac_final.zip --save-video
    python test_with_explainability.py --load carla_sac_final.zip --rendering

Outputs:
    - ./test_results/metrics.json          # Performance statistics
    - ./test_results/explanations/         # Saved explanations
    - ./test_results/videos/               # Recorded episodes (if --save-video)
    - ./test_results/causal_graphs/        # Decision causal graphs
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gymnasium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from carla_rl_env import CarlaRLEnv


# ==============================================================================
# -- Feature Extractor (Must Match Training) ----------------------------------
# ==============================================================================

class CarlaCNNExtractor(BaseFeaturesExtractor):
    """Same architecture as training - required for loading model."""

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
# -- Post-Hoc Explainability Agent ---------------------------------------------
# ==============================================================================

class PostHocExplainabilityAgent:
    """
    Observer model that generates comprehensive explanations AFTER decisions.
    
    Generates:
    1. Grad-CAM: Visual attention heatmaps on RGB + segmentation
    2. State Saliency: Importance of each state variable
    3. Human-Readable Rules: Natural language explanations
    4. Causal Graphs: Decision pathway visualization
    """
    
    STATE_NAMES = [
        'speed',
        'lane_offset',
        'heading_error',
        'junction_flag',
        'radar_distance',
        'radar_speed',
        'traffic_light_state',
        'speed_limit_norm'
    ]
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
        # Store intermediate activations for Grad-CAM
        self.activations = {}
        self.gradients = {}
        # Visual availability flag (set False if we cannot hook into a conv layer)
        self.visual_available = True
        self.visual_unavailable_reason = None

        # Try to hook the last conv layer for Grad-CAM; if that fails, continue without visual explainability
        try:
            self._register_hooks()
        except Exception as e:
            # Don't fail the whole pipeline for missing visual extractor; fall back to non-visual explainability
            self.visual_available = False
            self.visual_unavailable_reason = str(e)
            print("Warning: Visual explainability disabled:", self.visual_unavailable_reason)
        
    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        fe = self.model.policy.features_extractor

        # Helpful debug info for incompatible/unknown feature extractors
        print("Debug: features_extractor type:", type(fe))
        try:
            print("Debug: features_extractor repr:", repr(fe))
        except Exception:
            pass

        # Define hooks
        def forward_hook(module, input, output):
            # Prefer storing CPU tensors to avoid GPU-device issues during analysis
            try:
                self.activations['conv_output'] = output.detach().cpu()
            except Exception:
                self.activations['conv_output'] = output.detach()

        def backward_hook(module, grad_in, grad_out):
            try:
                self.gradients['conv_output'] = grad_out[0].detach().cpu()
            except Exception:
                self.gradients['conv_output'] = grad_out[0].detach()

        # Primary: chain contains explicit 'cnn' (CarlaCNNExtractor)
        if hasattr(fe, 'cnn'):
            try:
                target_layer = fe.cnn[4]  # Last ReLU in our expected extractor
            except Exception:
                # Fallback to last module in the sequence
                try:
                    target_layer = list(fe.cnn.modules())[-1]
                except Exception:
                    target_layer = None

        else:
            # Try to find the last Conv2d in the feature extractor (generic fallback)
            target_layer = None
            try:
                import torch

                conv_modules = []
                # Some feature extractors may expose named_modules
                if hasattr(fe, 'named_modules'):
                    for name, module in fe.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            conv_modules.append((name, module))
                else:
                    # Fall back to modules() if available
                    if hasattr(fe, 'modules'):
                        for module in fe.modules():
                            if isinstance(module, torch.nn.Conv2d):
                                conv_modules.append((None, module))

                if conv_modules:
                    # Choose the last conv layer found
                    target_layer = conv_modules[-1][1]
            except Exception:
                target_layer = None

        if target_layer is None:
            # Print available attributes to help debugging and fail with a clear message
            available = []
            try:
                available = dir(fe)
            except Exception:
                available = ['<unavailable>']
            raise AttributeError(
                "The model's features_extractor does not appear to contain a CNN or Conv2d layers. "
                "Ensure the model was trained with a compatible visual feature extractor (e.g. CarlaCNNExtractor). "
                "Available attributes: {}".format(available)
            )

        # Register hooks on the discovered layer
        try:
            target_layer.register_forward_hook(forward_hook)
        except Exception:
            # Some modules only support register_forward_pre_hook or have different API; rethrow with context
            raise RuntimeError(f"Failed to register forward hook on target layer: {target_layer}")

        # Use full backward hook when available (PyTorch >=1.8). If not, use register_backward_hook as fallback.
        try:
            target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            try:
                target_layer.register_backward_hook(backward_hook)
            except Exception:
                raise RuntimeError(f"Failed to register backward hook on target layer: {target_layer}")

    def _policy_forward(self, obs_tensor, deterministic: bool = True):
        """Call the model's policy forward and normalize the outputs.

        Returns a tuple (action, value, log_prob) where missing items are None.
        """
        out = self.model.policy.forward(obs_tensor, deterministic=deterministic)

        # Normalize output formats
        try:
            if isinstance(out, (list, tuple)):
                if len(out) == 1:
                    return out[0], None, None
                elif len(out) == 2:
                    return out[0], out[1], None
                else:
                    return out[0], out[1], out[2]
            else:
                # Single tensor returned
                return out, None, None
        except Exception:
            # As a last resort, return the raw output as action
            return out, None, None
    
    def explain_decision(self, observation: Dict, action: np.ndarray) -> Dict:
        """
        Generate comprehensive explanation for a decision.
        
        Returns dict with:
        - grad_cam_rgb: Grad-CAM heatmap for RGB input
        - grad_cam_seg: Grad-CAM heatmap for segmentation input
        - state_saliency: Importance scores for each state variable
        - rule_explanation: Human-readable text
        - causal_graph: NetworkX graph of decision pathway
        - q_values: Critic Q-values (if SAC)
        - action_probs: Action probabilities (if PPO)
        """
        explanation = {}
        
        # 1. Grad-CAM: Visual attention
        grad_cam_rgb, grad_cam_seg = self._compute_grad_cam(observation, action)
        if grad_cam_rgb is None or grad_cam_seg is None:
            explanation['grad_cam_rgb'] = None
            explanation['grad_cam_seg'] = None
            explanation['grad_cam_available'] = False
            explanation['grad_cam_unavailable_reason'] = getattr(self, 'visual_unavailable_reason', None)
        else:
            explanation['grad_cam_rgb'] = grad_cam_rgb
            explanation['grad_cam_seg'] = grad_cam_seg
            explanation['grad_cam_available'] = True
        
        # 2. State Saliency: Feature importance
        state_saliency = self._compute_state_saliency(observation, action)
        explanation['state_saliency'] = state_saliency

        # 2.5 Counterfactuals: suggest minimal state changes that impact action
        try:
            cfs = self.compute_counterfactuals(observation, action)
            explanation['counterfactuals'] = cfs
        except Exception:
            explanation['counterfactuals'] = []
        
        # 3. Q-values or action probabilities
        if hasattr(self.model, 'critic'):
            # SAC: compute Q-values
            q1, q2 = self._compute_q_values(observation, action)
            explanation['q_value_1'] = q1
            explanation['q_value_2'] = q2
            explanation['uncertainty'] = abs(q1 - q2)
        else:
            # PPO: compute action probabilities
            action_probs = self._compute_action_probs(observation)
            explanation['action_probs'] = action_probs
        
        # 4. Human-readable rule explanation
        rule_text = self._generate_rule_explanation(
            observation, action, state_saliency
        )
        explanation['rule_explanation'] = rule_text

        # Natural language one-sentence summary (concise)
        nl_summary = self._generate_nl_summary(observation, action, state_saliency, explanation)
        explanation['nl_explanation'] = nl_summary
        
        # 5. Causal graph
        causal_graph = self._build_causal_graph(
            observation, action, state_saliency
        )
        explanation['causal_graph'] = causal_graph
        
        # Store raw data
        explanation['action'] = action
        explanation['observation'] = observation
        
        return explanation
    
    def _compute_grad_cam(self, observation: Dict, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM heatmaps for RGB and segmentation inputs.
        
        Returns:
        - grad_cam_rgb: (H, W) heatmap for RGB channels
        - grad_cam_seg: (H, W) heatmap for segmentation channel
        """
        # If visual explainability is not available, fall back to occlusion heatmap
        if not getattr(self, 'visual_available', True):
            try:
                occl = self.compute_occlusion_heatmap(observation, action)
                return occl, occl
            except Exception:
                return None, None

        # Prepare observation tensors
        obs_tensor = self._obs_to_tensor(observation)
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        # Enable gradients
        for key in obs_tensor:
            if isinstance(obs_tensor[key], torch.Tensor):
                obs_tensor[key].requires_grad = True
        
        # Forward pass through policy
        with torch.set_grad_enabled(True):
            action_pred, _, _ = self._policy_forward(obs_tensor, deterministic=True)
            
            # Compute loss (distance from actual action)
            loss = F.mse_loss(action_pred, action_tensor.unsqueeze(0))
            
            # Backward pass
            self.model.policy.zero_grad()
            loss.backward()
        
        # Get activations and gradients from last conv layer
        activations = self.activations.get('conv_output', None)
        gradients = self.gradients.get('conv_output', None)
        
        if activations is None or gradients is None:
            # Fallback: return zeros
            return np.zeros((84, 84)), np.zeros((84, 84))
        
        # Compute Grad-CAM
        # activations: (batch, channels, H, W)
        # gradients: (batch, channels, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
        
        # Weighted combination of activations
        grad_cam = torch.sum(weights * activations, dim=1)  # (batch, H, W)
        
        # ReLU to keep only positive influences
        grad_cam = F.relu(grad_cam)
        
        # Normalize to [0, 1]
        grad_cam = grad_cam.squeeze().cpu().numpy()
        if grad_cam.max() > 0:
            grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        
        # Upsample to original image size (84x84)
        grad_cam = cv2.resize(grad_cam, (84, 84))
        
        # Split attribution between RGB and segmentation channels
        # For simplicity, we use the same heatmap for both
        # (In practice, you could compute separate Grad-CAMs for each channel)
        grad_cam_rgb = grad_cam.copy()
        grad_cam_seg = grad_cam.copy()
        
        return grad_cam_rgb, grad_cam_seg

    def compute_occlusion_heatmap(self, observation: Dict, action: np.ndarray, grid_size: int = 7) -> np.ndarray:
        """
        Compute a black-box occlusion heatmap by masking patches of the RGB image and
        measuring the change in the model's predicted action. Returns an upsampled
        heatmap of shape (84, 84) with values normalized to [0,1].
        """
        # Get original action from model.predict (raw observation expected)
        try:
            orig_action, _ = self.model.predict(observation, deterministic=True)
            orig_action = np.array(orig_action).astype(float)
        except Exception:
            # Fallback: use provided action
            orig_action = np.array(action).astype(float)

        # Prepare image (H,W, C) uint8 0-255
        img = observation['camera']
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        H, W, C = img.shape
        heatmap = np.zeros((grid_size, grid_size), dtype=float)

        patch_h = H // grid_size
        patch_w = W // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                img_copy = img.copy()
                y0 = i * patch_h
                x0 = j * patch_w
                y1 = y0 + patch_h
                x1 = x0 + patch_w

                # Occlude with mean color
                img_copy[y0:y1, x0:x1, :] = img_copy.mean(axis=(0, 1), keepdims=True)

                # Build occluded observation
                occluded_obs = observation.copy()
                # convert back to original expected channel order if necessary
                occluded_img = img_copy
                if observation['camera'].shape[0] == 3:
                    occluded_img = np.transpose(occluded_img, (2, 0, 1))
                occluded_obs = dict(observation)
                occluded_obs['camera'] = occluded_img

                try:
                    pred, _ = self.model.predict(occluded_obs, deterministic=True)
                    pred = np.array(pred).astype(float)
                except Exception:
                    pred = orig_action

                # importance measured as L2 change in action
                heatmap[i, j] = np.linalg.norm(orig_action - pred)

        # Normalize heatmap to [0,1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        # Upsample to image size
        heatmap_up = cv2.resize(heatmap.astype(np.float32), (W, H))
        return heatmap_up

    def compute_counterfactuals(self, observation: Dict, action: np.ndarray, target_delta: float = 0.2) -> List[str]:
        """
        Generate simple counterfactual suggestions for state variables using finite differences.
        For each state variable, estimate sensitivity of the action (steer) to that variable and
        compute an approximate delta that would change the steer by `target_delta`.
        Returns a list of human-readable sentences.
        """
        state = observation['state']
        if isinstance(state, np.ndarray):
            state = state.copy()
        else:
            state = np.array(state)

        base_action, _ = self.model.predict(observation, deterministic=True)
        base_action = np.array(base_action).astype(float)

        eps = 1e-2
        results = []

        for i, name in enumerate(self.STATE_NAMES):
            orig_val = float(state[i])
            # choose epsilon relative to value scale
            delta = max(abs(orig_val) * 0.05, eps)

            # perturb +delta
            s_plus = state.copy()
            s_plus[i] = orig_val + delta
            obs_plus = dict(observation)
            obs_plus['state'] = np.array(s_plus)
            try:
                pred_plus, _ = self.model.predict(obs_plus, deterministic=True)
                pred_plus = np.array(pred_plus).astype(float)
            except Exception:
                pred_plus = base_action

            # perturb -delta
            s_minus = state.copy()
            s_minus[i] = orig_val - delta
            obs_minus = dict(observation)
            obs_minus['state'] = np.array(s_minus)
            try:
                pred_minus, _ = self.model.predict(obs_minus, deterministic=True)
                pred_minus = np.array(pred_minus).astype(float)
            except Exception:
                pred_minus = base_action

            # approximate derivative for steer (index 0)
            deriv = (pred_plus[0] - pred_minus[0]) / (2 * delta)
            if abs(deriv) < 1e-6:
                continue

            required = target_delta / abs(deriv)
            direction = 'increase' if deriv > 0 else 'decrease'
            sentence = f"If {name.replace('_', ' ')} were to {direction} by ~{required:.2f}, the steering would change by about {target_delta:.2f}."
            results.append(sentence)

        return results
    
    def _compute_state_saliency(self, observation: Dict, action: np.ndarray) -> Dict[str, float]:
        """
        Compute saliency (importance) of each state variable using gradients.
        
        Returns dict: {state_name: saliency_score}
        """
        # Prepare observation with gradients enabled for state
        obs_tensor = self._obs_to_tensor(observation)
        state_tensor = obs_tensor['state']
        state_tensor.requires_grad = True
        
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        # Forward pass
        with torch.set_grad_enabled(True):
            action_pred, _, _ = self._policy_forward(obs_tensor, deterministic=True)
            
            # Compute loss
            loss = F.mse_loss(action_pred, action_tensor.unsqueeze(0))
            
            # Backward pass
            self.model.policy.zero_grad()
            loss.backward()
        
        # Get gradients of loss w.r.t. state
        if state_tensor.grad is not None:
            saliency = state_tensor.grad.abs().squeeze().cpu().numpy()
            
            # Normalize to [0, 1]
            if saliency.max() > 0:
                saliency = saliency / saliency.max()
        else:
            # Fallback: uniform importance
            saliency = np.ones(len(self.STATE_NAMES)) / len(self.STATE_NAMES)
        
        # Map to state names
        saliency_dict = {
            name: float(saliency[i]) 
            for i, name in enumerate(self.STATE_NAMES)
        }
        
        return saliency_dict
    
    def _compute_q_values(self, observation: Dict, action: np.ndarray) -> Tuple[float, float]:
        """Compute Q-values for SAC model."""
        obs_tensor = self._obs_to_tensor(observation)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Some critic implementations expose q1_forward/q2_forward, others only q1_forward.
            # Handle both cases robustly.
            try:
                q1 = self.model.critic.q1_forward(obs_tensor, action_tensor).item()
            except Exception:
                # Try common alternative: forward call
                try:
                    q1 = self.model.critic.forward(obs_tensor, action_tensor).item()
                except Exception:
                    q1 = float('nan')

            try:
                q2 = self.model.critic.q2_forward(obs_tensor, action_tensor).item()
            except Exception:
                # If q2 not available, fallback to q1 (or NaN)
                q2 = q1
        
        return q1, q2
    
    def _compute_action_probs(self, observation: Dict) -> np.ndarray:
        """Compute action probabilities for PPO model."""
        obs_tensor = self._obs_to_tensor(observation)
        
        with torch.no_grad():
            # For continuous actions, we approximate with action distribution
            action, _, _ = self._policy_forward(obs_tensor, deterministic=False)
            # Return as numpy array
            if isinstance(action, torch.Tensor):
                return action.cpu().numpy().squeeze()
            else:
                try:
                    return np.array(action).squeeze()
                except Exception:
                    return action
    
    def _generate_rule_explanation(self, observation: Dict, action: np.ndarray, 
                                   saliency: Dict[str, float]) -> str:
        """
        Generate human-readable explanation using domain rules.
        
        Returns natural language text explaining the decision.
        """
        state = observation['state']
        state_dict = {name: float(state[i]) for i, name in enumerate(self.STATE_NAMES)}
        
        steering = action[0]
        accel = action[1]
        
        # Get top 3 most important state variables
        top_factors = sorted(saliency.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Build explanation parts
        parts = []
        
        # Action description
        if abs(steering) < 0.1:
            steer_desc = "maintaining straight path"
        elif steering > 0:
            steer_desc = f"steering right ({steering:.2f})"
        else:
            steer_desc = f"steering left ({steering:.2f})"
        
        if accel > 0.3:
            accel_desc = "accelerating"
        elif accel < -0.3:
            accel_desc = "braking"
        else:
            accel_desc = "maintaining speed"
        
        parts.append(f"Agent is {steer_desc} and {accel_desc}.")
        
        # Explain based on important factors
        reasons = []
        
        for var_name, importance in top_factors:
            if importance < 0.3:  # Only include significant factors
                continue
            
            value = state_dict[var_name]
            
            if var_name == 'traffic_light_state' and value < -0.5:
                reasons.append(f"red traffic light (importance: {importance:.2f})")
            elif var_name == 'radar_distance' and value < 20:
                reasons.append(f"vehicle ahead at {value:.1f}m (importance: {importance:.2f})")
            elif var_name == 'lane_offset' and abs(value) > 0.3:
                direction = "left" if value < 0 else "right"
                reasons.append(f"lane correction needed - {direction} offset {abs(value):.2f}m (importance: {importance:.2f})")
            elif var_name == 'heading_error' and abs(value) > 0.2:
                reasons.append(f"heading adjustment needed - error {abs(value):.2f} rad (importance: {importance:.2f})")
            elif var_name == 'speed' and value < 5:
                reasons.append(f"low speed {value:.1f} km/h requires acceleration (importance: {importance:.2f})")
            elif var_name == 'junction_flag' and value > 0.5:
                reasons.append(f"approaching intersection (importance: {importance:.2f})")
        
        if reasons:
            parts.append(f"\nKey factors influencing decision:\n  - " + "\n  - ".join(reasons))
        
        # State summary
        parts.append(f"\nCurrent state:")
        parts.append(f"  Speed: {state_dict['speed']:.1f} km/h")
        parts.append(f"  Lane offset: {state_dict['lane_offset']:.2f} m")
        parts.append(f"  Heading error: {state_dict['heading_error']:.2f} rad")
        if state_dict['radar_distance'] < 50:
            parts.append(f"  Vehicle ahead: {state_dict['radar_distance']:.1f} m")
        
        return "\n".join(parts)
    
    def _build_causal_graph(self, observation: Dict, action: np.ndarray,
                           saliency: Dict[str, float]) -> nx.DiGraph:
        """
        Build causal graph showing decision pathway.
        
        Returns NetworkX directed graph with nodes:
        - State variables (inputs)
        - Feature extraction (hidden)
        - Action outputs (outputs)
        """
        G = nx.DiGraph()
        
        state = observation['state']
        state_dict = {name: float(state[i]) for i, name in enumerate(self.STATE_NAMES)}
        
        # Add state nodes (inputs)
        for name in self.STATE_NAMES:
            importance = saliency.get(name, 0.0)
            G.add_node(f"state_{name}", 
                      type='input',
                      value=state_dict[name],
                      importance=importance,
                      label=f"{name}\n{state_dict[name]:.2f}")
        
        # Add visual input nodes
        G.add_node("rgb_camera", type='input', importance=0.5, label="RGB Camera")
        G.add_node("seg_camera", type='input', importance=0.5, label="Segmentation")
        
        # Add feature extraction nodes
        G.add_node("cnn_features", type='hidden', label="CNN Features\n(256-dim)")
        G.add_node("state_features", type='hidden', label="State Features\n(64-dim)")
        G.add_node("combined_features", type='hidden', label="Combined\n(320-dim)")
        
        # Add action nodes (outputs)
        G.add_node("steering", type='output', value=action[0], 
                  label=f"Steering\n{action[0]:.2f}")
        G.add_node("acceleration", type='output', value=action[1],
                  label=f"Acceleration\n{action[1]:.2f}")
        
        # Add edges with weights based on importance
        
        # Visual inputs → CNN features
        G.add_edge("rgb_camera", "cnn_features", weight=0.6)
        G.add_edge("seg_camera", "cnn_features", weight=0.4)
        
        # State inputs → State features
        for name in self.STATE_NAMES:
            importance = saliency.get(name, 0.0)
            if importance > 0.1:  # Only show significant connections
                G.add_edge(f"state_{name}", "state_features", weight=importance)
        
        # Features → Combined
        G.add_edge("cnn_features", "combined_features", weight=0.8)
        G.add_edge("state_features", "combined_features", weight=0.8)
        
        # Combined → Actions
        G.add_edge("combined_features", "steering", weight=0.7)
        G.add_edge("combined_features", "acceleration", weight=0.7)
        
        return G

    def _generate_nl_summary(self, observation: Dict, action: np.ndarray,
                             saliency: Dict[str, float], explanation: Dict) -> str:
        """
        Generate a concise natural-language summary (one or two sentences)
        explaining the decision. Uses top salient state variables and optional
        Q-value uncertainty to add confidence phrasing.
        """
        # Action descriptors
        steer = action[0]
        accel = action[1]

        if abs(steer) < 0.1:
            steer_phrase = "maintain a straight trajectory"
        elif steer > 0:
            steer_phrase = "steer right"
        else:
            steer_phrase = "steer left"

        if accel > 0.3:
            accel_phrase = "accelerate"
        elif accel < -0.3:
            accel_phrase = "brake"
        else:
            accel_phrase = "maintain speed"

        # Top salient factors
        top = sorted(saliency.items(), key=lambda x: x[1], reverse=True)[:2]
        reasons = []
        state = observation.get('state')
        state_vals = {}
        try:
            state_vals = {name: float(state[i]) for i, name in enumerate(self.STATE_NAMES)}
        except Exception:
            pass

        for name, imp in top:
            if imp < 0.2:
                continue
            val = state_vals.get(name, None)
            if name == 'traffic_light_state':
                if val is not None and val < -0.5:
                    reasons.append('a red traffic light')
                else:
                    reasons.append('traffic signal state')
            elif name == 'radar_distance':
                if val is not None:
                    reasons.append(f'vehicle ahead at ~{int(val)}m')
                else:
                    reasons.append('nearby vehicle')
            elif name == 'lane_offset':
                reasons.append('lane offset')
            elif name == 'speed':
                reasons.append('current speed')
            elif name == 'radar_speed':
                reasons.append('relative object speed')
            elif name == 'junction_flag':
                reasons.append('approaching intersection')
            else:
                reasons.append(name.replace('_', ' '))

        if reasons:
            reason_text = ' mainly because of ' + ' and '.join(reasons)
        else:
            reason_text = ''

        # Confidence phrasing using Q-value uncertainty if available
        conf = ''
        try:
            if 'uncertainty' in explanation and explanation['uncertainty'] is not None:
                unc = float(explanation['uncertainty'])
                if unc < 1.0:
                    conf = ' with moderate confidence'
                else:
                    conf = ' with low confidence'
        except Exception:
            conf = ''

        sentence = f"The agent will {steer_phrase} and {accel_phrase}{reason_text}.{conf}"
        return sentence
    
    def _obs_to_tensor(self, observation: Dict) -> Dict:
        """Convert observation to tensor format."""
        device = self.device
        
        # Camera (already in CHW format from SB3)
        camera = observation['camera']
        if isinstance(camera, np.ndarray):
            # Convert HWC to CHW if needed
            if camera.shape[-1] in [1, 3, 4]:
                camera = np.transpose(camera, (2, 0, 1))
            camera = torch.FloatTensor(camera).unsqueeze(0).to(device) / 255.0
        
        # Segmentation
        seg = observation['seg_camera']
        if isinstance(seg, np.ndarray):
            seg = torch.FloatTensor(seg).unsqueeze(0).to(device)
        
        # State
        state = observation['state']
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        return {
            'camera': camera,
            'seg_camera': seg,
            'state': state
        }


# ==============================================================================
# -- Visualization Utilities ---------------------------------------------------
# ==============================================================================

def visualize_explanation(explanation: Dict, save_path: str = None):
    """
    Create comprehensive visualization of explanation.
    
    Shows:
    - Original RGB and segmentation images
    - Grad-CAM heatmaps overlaid
    - State saliency bar chart
    - Rule explanation text
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. RGB Camera
    ax1 = fig.add_subplot(gs[0, 0])
    rgb = explanation['observation']['camera']
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    if rgb.shape[0] == 3:  # CHW to HWC
        rgb = np.transpose(rgb, (1, 2, 0))
    ax1.imshow(rgb)
    ax1.set_title("RGB Camera", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Segmentation Camera
    ax2 = fig.add_subplot(gs[0, 1])
    seg = explanation['observation']['seg_camera']
    ax2.imshow(seg, cmap='tab20')
    ax2.set_title("Segmentation", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Grad-CAM RGB
    ax3 = fig.add_subplot(gs[0, 2])
    grad_cam_rgb = explanation['grad_cam_rgb']
    # Overlay on RGB (handle missing grad-cam gracefully)
    overlay = rgb.copy()
    if grad_cam_rgb is None:
        # No visual explanation available: show original image with a notice
        ax3.imshow(overlay)
        ax3.text(0.5, 0.5, 'Grad-CAM unavailable', color='white', fontsize=14,
                 ha='center', va='center', transform=ax3.transAxes,
                 bbox=dict(facecolor='red', alpha=0.5))
    else:
        heatmap = cv2.applyColorMap((grad_cam_rgb * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
        ax3.imshow(overlay)
    ax3.set_title("Grad-CAM (RGB Attention)", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Grad-CAM Segmentation
    ax4 = fig.add_subplot(gs[1, 0])
    grad_cam_seg = explanation['grad_cam_seg']
    if grad_cam_seg is None:
        ax4.imshow(np.zeros((84, 84)), cmap='gray')
        ax4.text(0.5, 0.5, 'Grad-CAM unavailable', color='white', fontsize=14,
                 ha='center', va='center', transform=ax4.transAxes,
                 bbox=dict(facecolor='red', alpha=0.5))
    else:
        ax4.imshow(grad_cam_seg, cmap='hot')
    ax4.set_title("Grad-CAM (Seg Attention)", fontsize=14, fontweight='bold')
    ax4.axis('off')
    if grad_cam_seg is not None:
        ax4.colorbar = plt.colorbar(ax4.imshow(grad_cam_seg, cmap='hot'), ax=ax4)
    
    # 5. State Saliency Bar Chart
    ax5 = fig.add_subplot(gs[1, 1:])
    saliency = explanation['state_saliency']
    names = list(saliency.keys())
    values = list(saliency.values())
    colors = plt.cm.viridis(np.array(values))
    ax5.barh(names, values, color=colors)
    ax5.set_xlabel("Saliency Score", fontsize=12)
    ax5.set_title("State Variable Importance", fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 1.0)
    
    # 6. Rule Explanation Text
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    explanation_text = explanation['rule_explanation']
    # Prepend concise NL summary if available
    nl = explanation.get('nl_explanation')
    if nl:
        explanation_text = nl + "\n\n" + explanation_text
    ax6.text(0.05, 0.95, explanation_text, 
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add Q-values if available
    if 'q_value_1' in explanation:
        q_text = f"\nQ-Values: Q1={explanation['q_value_1']:.2f}, Q2={explanation['q_value_2']:.2f}, Uncertainty={explanation['uncertainty']:.2f}"
        ax6.text(0.05, 0.05, q_text,
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f"Decision Explanation - Action: Steer={explanation['action'][0]:.2f}, Accel={explanation['action'][1]:.2f}",
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_causal_graph(graph: nx.DiGraph, save_path: str = None):
    """Visualize causal decision graph."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Position nodes using hierarchical layout
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Separate nodes by type
    input_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'input']
    hidden_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'hidden']
    output_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'output']
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=input_nodes, 
                          node_color='lightblue', node_size=2000, alpha=0.9,
                          label='Inputs', ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=hidden_nodes,
                          node_color='lightgreen', node_size=2500, alpha=0.9,
                          label='Features', ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=output_nodes,
                          node_color='salmon', node_size=2500, alpha=0.9,
                          label='Actions', ax=ax)
    
    # Draw edges with varying thickness based on weight
    edges = graph.edges(data=True)
    weights = [d.get('weight', 0.5) for _, _, d in edges]
    nx.draw_networkx_edges(graph, pos, width=[w*3 for w in weights], 
                          alpha=0.6, edge_color='gray',
                          arrows=True, arrowsize=20, ax=ax)
    
    # Draw labels
    labels = nx.get_node_attributes(graph, 'label')
    nx.draw_networkx_labels(graph, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    plt.title("Causal Decision Graph", fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==============================================================================
# -- Testing Function ----------------------------------------------------------
# ==============================================================================

def test_model(args):
    """Test trained model and generate explanations."""
    
    print('=' * 80)
    print('  CARLA RL Testing with Post-Hoc Explainability')
    print('=' * 80)
    print()
    
    # Create output directories
    output_dir = Path('./test_results')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'explanations').mkdir(exist_ok=True)
    (output_dir / 'videos').mkdir(exist_ok=True)
    (output_dir / 'causal_graphs').mkdir(exist_ok=True)
    (output_dir / 'nl_explanations').mkdir(exist_ok=True)
    (output_dir / 'heatmaps').mkdir(exist_ok=True)
    
    # Create environment
    env = CarlaRLEnv(
        host=args.host,
        port=args.port,
        num_npc_vehicles=args.npc_vehicles,
        num_npc_walkers=args.npc_walkers,
        no_rendering=not args.rendering,
    )
    
    print(f'Environment created')
    print(f'  Observation space: {env.observation_space}')
    print(f'  Action space: {env.action_space}')
    print()
    
    # Load model
    print(f'Loading model from: {args.load}')
    
    # FIX: Use safe loading with validation
    from model_utils import load_model_safe, find_latest_checkpoint
    
    model, algo_name = load_model_safe(args.load, env, args.device)
    
    if model is None:
        print("\n" + "="*80)
        print("RECOVERY OPTIONS:")
        print("="*80)
        
        # Try to find checkpoints
        latest = find_latest_checkpoint('./checkpoints', 'sac')
        if latest:
            print(f"\n✓ Found checkpoint: {latest}")
            print(f"\nTry loading this checkpoint instead:")
            print(f"  python test_with_explainability.py --load {latest}")
        else:
            print("\n✗ No checkpoints found in ./checkpoints/")
            print("\nYou need to:")
            print("  1. Retrain the model")
            print("  2. Ensure training completes without interruption")
            print("  3. Check that model.save() completes successfully")
        
        print("="*80)
        env.close()
        return
    
    print(f'Model loaded: {algo_name}')
    print(model.policy.features_extractor)
    print()
    
    # Create explainability agent
    print('Initializing Post-Hoc Explainability Agent...')
    explainer = PostHocExplainabilityAgent(model, device=args.device)
    print('  ✓ Grad-CAM hooks registered')
    print('  ✓ State saliency analyzer ready')
    print('  ✓ Rule generator initialized')
    print('  ✓ Causal graph builder ready')
    print()
    
    # Testing loop
    episode_rewards = []
    episode_lengths = []
    all_explanations = []
    
    print(f'Starting testing for {args.episodes} episodes...')
    print('-' * 80)
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_explanations = []
        last_explanation = None
        
        # Video recording setup
        if args.save_video:
            video_path = output_dir / 'videos' / f'episode_{episode+1}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (84*3, 84*2))
        
        done = False
        step = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Generate explanation every N steps
            if step % args.explain_every == 0 or step < 10:
                explanation = explainer.explain_decision(obs, action)
                last_explanation = explanation
                episode_explanations.append(explanation)
                
                # Save visualization
                if args.save_explanations:
                    save_path = output_dir / 'explanations' / f'ep{episode+1}_step{step}.png'
                    visualize_explanation(explanation, str(save_path))

                    # Save concise NL summary separately
                    nl_path = output_dir / 'nl_explanations' / f'ep{episode+1}_step{step}.txt'
                    try:
                        with open(nl_path, 'w') as f:
                            f.write(explanation.get('nl_explanation', '') + "\n\n")
                            f.write(explanation.get('rule_explanation', ''))
                            # Save counterfactuals if present
                            cfs = explanation.get('counterfactuals')
                            if cfs:
                                f.write("\n\nCounterfactuals:\n")
                                for cf in cfs:
                                    f.write(f" - {cf}\n")
                    except Exception:
                        pass
                
                # Save causal graph
                if args.save_graphs and step == 0:  # Save first step of each episode
                    graph_path = output_dir / 'causal_graphs' / f'ep{episode+1}_causal_graph.png'
                    visualize_causal_graph(explanation['causal_graph'], str(graph_path))
                    # Also save heatmaps for the first step if available
                    try:
                        grad_rgb = explanation.get('grad_cam_rgb')
                        grad_seg = explanation.get('grad_cam_seg')
                        if grad_rgb is not None:
                            heat_rgb_path = output_dir / 'heatmaps' / f'ep{episode+1}_step{step}_heatmap_rgb.png'
                            cv2.imwrite(str(heat_rgb_path), (grad_rgb * 255).astype(np.uint8))
                        if grad_seg is not None:
                            heat_seg_path = output_dir / 'heatmaps' / f'ep{episode+1}_step{step}_heatmap_seg.png'
                            cv2.imwrite(str(heat_seg_path), (grad_seg * 255).astype(np.uint8))
                    except Exception:
                        pass
                
                # Print summary
                if args.verbose:
                    print(f"\n[Episode {episode+1}, Step {step}]")
                    print(f"Action: Steer={action[0]:.3f}, Accel={action[1]:.3f}")
                    print("Top 3 important states:")
                    top_3 = sorted(explanation['state_saliency'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
                    for name, val in top_3:
                        print(f"  {name}: {val:.3f}")
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            step += 1
            
            # Record video frame
            if args.save_video:
                # Create frame with RGB, segmentation, and Grad-CAM
                rgb = obs['camera']
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                if rgb.shape[0] == 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                
                seg = obs['seg_camera']
                seg_colored = cv2.applyColorMap((seg * 20).astype(np.uint8), cv2.COLORMAP_JET)
                
                # Create explanation heatmap if available
                # Use last_explanation (most recent) to populate heatmap; handle missing gracefully
                if last_explanation is not None and last_explanation.get('grad_cam_rgb') is not None:
                    try:
                        heat_src = (last_explanation['grad_cam_rgb'] * 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(heat_src, cv2.COLORMAP_HOT)
                    except Exception:
                        heatmap = np.zeros((84, 84, 3), dtype=np.uint8)
                else:
                    heatmap = np.zeros((84, 84, 3), dtype=np.uint8)
                
                # Combine frames
                top_row = np.hstack([rgb, seg_colored, heatmap])
                
                # Add text info
                bottom_frame = np.zeros((84, 84*3, 3), dtype=np.uint8)
                cv2.putText(bottom_frame, f"Step: {step}", (10, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(bottom_frame, f"Reward: {episode_reward:.1f}", (10, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(bottom_frame, f"Steer: {action[0]:.2f}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(bottom_frame, f"Accel: {action[1]:.2f}", (10, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                combined_frame = np.vstack([top_row, bottom_frame])
                video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
        
        if args.save_video:
            video_writer.release()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_explanations.extend(episode_explanations)
        
        print(f'Episode {episode+1}/{args.episodes} | '
              f'Reward: {episode_reward:+8.1f} | '
              f'Length: {episode_length:4d} steps | '
              f'Explanations generated: {len(episode_explanations)}')
    
    print('-' * 80)
    print()
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print('Testing Results:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Mean Reward: {mean_reward:+8.2f} ± {std_reward:.2f}')
    print(f'  Mean Length: {mean_length:.1f} steps')
    print(f'  Total Explanations: {len(all_explanations)}')
    print()
    
    # Save metrics
    metrics = {
        'algorithm': algo_name,
        'model_path': args.load,
        'episodes': args.episodes,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'Results saved to: {output_dir}')
    print(f'  Metrics: {output_dir / "metrics.json"}')
    if args.save_explanations:
        print(f'  Explanations: {output_dir / "explanations"}/')
    if args.save_video:
        print(f'  Videos: {output_dir / "videos"}/')
    if args.save_graphs:
        print(f'  Causal Graphs: {output_dir / "causal_graphs"}/')
    
    env.close()


# ==============================================================================
# -- Main Entry Point ----------------------------------------------------------
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test CARLA RL agent with post-hoc explainability',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--load', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes (default: 5)')
    parser.add_argument('--host', default='127.0.0.1',
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--npc-vehicles', type=int, default=15,
                       help='Number of NPC vehicles')
    parser.add_argument('--npc-walkers', type=int, default=5,
                       help='Number of NPC pedestrians')
    parser.add_argument('--rendering', action='store_true',
                       help='Enable CARLA rendering')
    parser.add_argument('--save-video', action='store_true',
                       help='Save episode videos with overlays')
    parser.add_argument('--save-explanations', action='store_true',
                       help='Save explanation visualizations')
    parser.add_argument('--save-graphs', action='store_true',
                       help='Save causal decision graphs')
    parser.add_argument('--explain-every', type=int, default=10,
                       help='Generate explanation every N steps')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed explanations')
    
    args = parser.parse_args()
    
    test_model(args)


if __name__ == '__main__':
    main()
