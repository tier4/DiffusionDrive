#!/usr/bin/env python3
"""Verify Bench2Drive driving command one-hot encoding in cached data."""

import os
import pickle
import gzip
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import random

def load_cached_sample(cache_dir, token):
    """Load a cached sample's features and targets."""
    # Find the cache file
    log_name = "_".join(token.split("_")[:-1])
    token_dir = Path(cache_dir) / log_name / token
    
    if not token_dir.exists():
        return None, None
        
    # Load features
    feature_file = token_dir / "transfuser_feature.gz"
    target_file = token_dir / "transfuser_target.gz"
    
    features = None
    targets = None
    
    if feature_file.exists():
        with gzip.open(feature_file, 'rb') as f:
            features = pickle.load(f)
            
    if target_file.exists():
        with gzip.open(target_file, 'rb') as f:
            targets = pickle.load(f)
            
    return features, targets

def verify_command_encoding(cache_dir, num_samples=2000):
    """Verify driving command one-hot encoding."""
    
    results = {
        'total_samples': 0,
        'valid_one_hot': 0,
        'invalid_one_hot': 0,
        'command_distribution': defaultdict(int),
        'status_shape_correct': 0,
        'status_shape_incorrect': 0,
        'examples': []
    }
    
    # Get all available tokens
    all_tokens = []
    for log_dir in Path(cache_dir).iterdir():
        if log_dir.is_dir():
            for token_dir in log_dir.iterdir():
                if token_dir.is_dir():
                    all_tokens.append(token_dir.name)
    
    # Random sample
    sampled_tokens = random.sample(all_tokens, min(num_samples, len(all_tokens)))
    
    print(f"Checking {len(sampled_tokens)} cached samples...")
    
    for token in sampled_tokens:
        features, targets = load_cached_sample(cache_dir, token)
        
        if features is None:
            continue
            
        results['total_samples'] += 1
        
        # Check status feature
        if 'status_feature' in features:
            status = features['status_feature']
            
            # Convert to numpy if tensor
            if isinstance(status, torch.Tensor):
                status = status.cpu().numpy()
            
            # Check shape (should be [8]: 4 for one-hot command + 2 velocity + 2 acceleration)
            if status.shape == (8,):
                results['status_shape_correct'] += 1
                
                # Extract one-hot command (first 4 values)
                command_one_hot = status[:4]
                
                # Verify one-hot encoding
                if np.abs(np.sum(command_one_hot) - 1.0) < 0.01:  # Should sum to 1
                    if np.all((command_one_hot == 0) | (command_one_hot == 1)):  # All 0 or 1
                        results['valid_one_hot'] += 1
                        
                        # Find which command
                        command_idx = np.argmax(command_one_hot)
                        results['command_distribution'][command_idx] += 1
                    else:
                        results['invalid_one_hot'] += 1
                        if len(results['examples']) < 5:
                            results['examples'].append({
                                'token': token,
                                'command_one_hot': command_one_hot.tolist(),
                                'sum': np.sum(command_one_hot),
                                'issue': 'Non-binary values'
                            })
                else:
                    results['invalid_one_hot'] += 1
                    if len(results['examples']) < 5:
                        results['examples'].append({
                            'token': token,
                            'command_one_hot': command_one_hot.tolist(),
                            'sum': np.sum(command_one_hot),
                            'issue': 'Sum not 1'
                        })
                
                # Also check velocity and acceleration ranges
                velocity = status[4:6]
                acceleration = status[6:8]
                
                if len(results['examples']) < 3:
                    results['examples'].append({
                        'token': token,
                        'command_one_hot': command_one_hot.tolist(),
                        'velocity': velocity.tolist(),
                        'acceleration': acceleration.tolist(),
                        'full_status': status.tolist()
                    })
            else:
                results['status_shape_incorrect'] += 1
                print(f"Incorrect status shape: {status.shape} for token {token}")
        else:
            print(f"No status_feature found for token {token}")
    
    return results

def main():
    cache_dir = "/workspace/navsim_workspace/cache/bench2drive_Base_cache"
    
    print(f"Verifying Bench2Drive driving command encoding in: {cache_dir}")
    
    results = verify_command_encoding(cache_dir)
    
    print(f"\n=== Driving Command Verification Results ===")
    print(f"Total samples analyzed: {results['total_samples']}")
    print(f"Correct status shape [8]: {results['status_shape_correct']}")
    print(f"Incorrect status shape: {results['status_shape_incorrect']}")
    print(f"Valid one-hot encoding: {results['valid_one_hot']}")
    print(f"Invalid one-hot encoding: {results['invalid_one_hot']}")
    
    if results['valid_one_hot'] > 0:
        validity_rate = results['valid_one_hot'] / (results['valid_one_hot'] + results['invalid_one_hot']) * 100
        print(f"One-hot validity rate: {validity_rate:.1f}%")
    
    print(f"\n=== Command Distribution ===")
    command_names = ['LEFT (0)', 'STRAIGHT (1)', 'RIGHT (2)', 'UNKNOWN (3)']
    for cmd_idx in range(4):
        count = results['command_distribution'][cmd_idx]
        if results['valid_one_hot'] > 0:
            pct = count / results['valid_one_hot'] * 100
            print(f"{command_names[cmd_idx]}: {count} ({pct:.1f}%)")
    
    if results['examples']:
        print(f"\n=== Example Status Features ===")
        for i, example in enumerate(results['examples'][:3]):
            print(f"\nExample {i+1} (token: {example['token']}):")
            if 'full_status' in example:
                print(f"  Full status: {example['full_status']}")
                print(f"  Command one-hot: {example['command_one_hot']}")
                print(f"  Velocity (vx, vy): {example['velocity']}")
                print(f"  Acceleration (ax, ay): {example['acceleration']}")
            else:
                print(f"  Issue: {example['issue']}")
                print(f"  Command one-hot: {example['command_one_hot']}")
                print(f"  Sum: {example['sum']}")
    
    # Summary
    if results['invalid_one_hot'] == 0 and results['valid_one_hot'] > 0:
        print(f"\n✓ All driving commands are properly one-hot encoded!")
    else:
        print(f"\n✗ Found {results['invalid_one_hot']} samples with invalid one-hot encoding")

if __name__ == "__main__":
    main()