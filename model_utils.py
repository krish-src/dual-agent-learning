"""
Model File Validation and Safe Saving Utilities
================================================

Ensures model files are valid SB3 archives and provides safe saving mechanisms.

Usage in training:
    from model_utils import safe_model_save
    safe_model_save(model, './carla_sac_final.zip')

Usage in testing:
    from model_utils import validate_model_file, load_model_safe
    if validate_model_file('./carla_sac_final.zip'):
        model = load_model_safe('./carla_sac_final.zip', env, device)
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm


def validate_model_file(filepath: str, verbose: bool = True) -> bool:
    """
    Validate that a file is a proper SB3 model archive.
    
    Checks:
    1. File exists
    2. File is a valid ZIP archive
    3. ZIP contains required SB3 files (policy.pth, etc.)
    
    Args:
        filepath: Path to model file
        verbose: Print detailed error messages
    
    Returns:
        True if valid, False otherwise
    """
    filepath = Path(filepath)
    
    # Check 1: File exists
    if not filepath.exists():
        if verbose:
            print(f"❌ ERROR: Model file does not exist: {filepath}")
        return False
    
    # Check 2: File is not empty
    if filepath.stat().st_size == 0:
        if verbose:
            print(f"❌ ERROR: Model file is empty (0 bytes): {filepath}")
        return False
    
    # Check 3: File is a valid ZIP
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Check 4: ZIP contains expected SB3 files
            files_in_zip = set(zf.namelist())
            
            # Required files in SB3 model
            required_files = {'data', 'pytorch_variables.pth'}
            
            # Check if required files exist
            missing_files = required_files - files_in_zip
            if missing_files:
                if verbose:
                    print(f"❌ ERROR: Model ZIP is missing required files: {missing_files}")
                    print(f"   Files found in ZIP: {files_in_zip}")
                return False
            
            # Try to test the ZIP integrity
            bad_file = zf.testzip()
            if bad_file is not None:
                if verbose:
                    print(f"❌ ERROR: Corrupted file in ZIP: {bad_file}")
                return False
            
            if verbose:
                print(f"✓ Model file is valid: {filepath}")
                print(f"  Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
                print(f"  Files in archive: {len(files_in_zip)}")
            
            return True
            
    except zipfile.BadZipFile as e:
        if verbose:
            print(f"❌ ERROR: File is not a valid ZIP archive: {filepath}")
            print(f"   Details: {e}")
            print(f"\n   This usually means:")
            print(f"   1. Training was interrupted during model save")
            print(f"   2. File was corrupted during transfer")
            print(f"   3. Wrong file was selected (not a model file)")
            print(f"\n   Solution: Retrain the model or use a backup checkpoint")
        return False
    
    except Exception as e:
        if verbose:
            print(f"❌ ERROR: Unexpected error validating model file: {e}")
        return False


def safe_model_save(model: BaseAlgorithm, filepath: str, backup: bool = True) -> bool:
    """
    Safely save a model with atomic write and validation.
    
    Process:
    1. Save to temporary file
    2. Validate the temporary file
    3. Atomically rename to final destination
    4. Optionally keep backup of old file
    
    Args:
        model: SB3 model instance
        filepath: Destination path
        backup: Keep backup of existing file
    
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    
    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Save to temporary file in same directory
    # (same directory ensures atomic rename works)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.zip.tmp',
        dir=filepath.parent,
        prefix='.tmp_model_'
    )
    os.close(temp_fd)  # Close file descriptor, we'll use the path
    
    try:
        print(f"Saving model to temporary file: {temp_path}")
        
        # Save model to temp file
        model.save(temp_path)
        
        # Step 2: Validate the saved file
        print("Validating saved model...")
        if not validate_model_file(temp_path, verbose=True):
            print(f"❌ ERROR: Model validation failed after save!")
            os.remove(temp_path)
            return False
        
        # Step 3: Backup existing file if requested
        if backup and filepath.exists():
            backup_path = filepath.with_suffix('.zip.backup')
            print(f"Creating backup: {backup_path}")
            shutil.copy2(filepath, backup_path)
        
        # Step 4: Atomic rename
        print(f"Moving to final location: {filepath}")
        shutil.move(temp_path, filepath)
        
        # Final validation
        if not validate_model_file(filepath, verbose=False):
            print(f"❌ ERROR: Model corrupted during move!")
            return False
        
        print(f"✓ Model saved successfully: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to save model: {e}")
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def load_model_safe(filepath: str, env, device: str = 'auto',
                   algorithm: Optional[str] = None) -> Tuple[Optional[BaseAlgorithm], str]:
    """
    Safely load a model with validation and auto-detection.
    
    Args:
        filepath: Path to model file
        env: Gym environment
        device: Device for model
        algorithm: 'sac', 'ppo', or None for auto-detect
    
    Returns:
        (model, algorithm_name) if successful, (None, '') if failed
    """
    # Validate file first
    print(f"\nValidating model file: {filepath}")
    if not validate_model_file(filepath, verbose=True):
        print("\n" + "="*80)
        print("SOLUTION:")
        print("  1. Check if training completed successfully")
        print("  2. Look for checkpoint files in ./checkpoints/")
        print("  3. If available, use a checkpoint instead:")
        print(f"     python test_with_explainability.py --load ./checkpoints/carla_sac_XXXXX_steps.zip")
        print("  4. If no checkpoints, retrain the model")
        print("="*80 + "\n")
        return None, ''
    
    # Auto-detect algorithm if not specified
    if algorithm is None:
        filepath_lower = filepath.lower()
        if 'sac' in filepath_lower:
            algorithm = 'sac'
        elif 'ppo' in filepath_lower:
            algorithm = 'ppo'
        else:
            # Try SAC first (more common for continuous control)
            algorithm = 'sac'
            print("⚠ WARNING: Cannot detect algorithm from filename, trying SAC first...")
    
    # Load model
    try:
        print(f"\nLoading {algorithm.upper()} model from: {filepath}")
        
        if algorithm.lower() == 'sac':
            model = SAC.load(filepath, env=env, device=device)
            algo_name = 'SAC'
        elif algorithm.lower() == 'ppo':
            model = PPO.load(filepath, env=env, device=device)
            algo_name = 'PPO'
        else:
            print(f"❌ ERROR: Unknown algorithm: {algorithm}")
            return None, ''
        
        print(f"✓ Model loaded successfully: {algo_name}")
        return model, algo_name
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load model as {algorithm.upper()}: {e}")
        
        # If auto-detection, try the other algorithm
        if 'sac' not in filepath.lower() and 'ppo' not in filepath.lower():
            other_algo = 'ppo' if algorithm == 'sac' else 'sac'
            print(f"\nTrying to load as {other_algo.upper()} instead...")
            return load_model_safe(filepath, env, device, other_algo)
        
        return None, ''


def find_latest_checkpoint(checkpoint_dir: str = './checkpoints',
                          algorithm: str = 'sac') -> Optional[str]:
    """
    Find the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        algorithm: 'sac' or 'ppo'
    
    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Pattern: carla_sac_XXXXX_steps.zip
    pattern = f"carla_{algorithm}_*_steps.zip"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Sort by modification time, return latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def check_model_compatibility(model_path: str, env) -> bool:
    """
    Check if a model is compatible with the environment.
    
    Args:
        model_path: Path to model file
        env: Environment instance
    
    Returns:
        True if compatible, False otherwise
    """
    # This is a placeholder - actual implementation would load
    # the model metadata and compare observation/action spaces
    # For now, we just validate the file exists
    return validate_model_file(model_path, verbose=False)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_utils.py <model_file.zip>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    result = validate_model_file(filepath, verbose=True)
    
    if result:
        print("\n✓ Model file is valid and ready to use")
        sys.exit(0)
    else:
        print("\n❌ Model file is invalid or corrupted")
        sys.exit(1)
