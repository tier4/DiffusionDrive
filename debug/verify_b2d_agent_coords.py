#!/usr/bin/env python3
"""Verify Bench2Drive agent states ego-relative coordinates."""

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

def verify_agent_coordinates(cache_dir, num_samples=1000):
    """Verify agent states are ego-relative and within expected bounds."""
    
    results = {
        'total_samples': 0,
        'agent_states_found': 0,
        'agent_labels_found': 0,
        'samples_with_agents': 0,
        'all_agents_zero': 0,
        'position_statistics': {
            'x_range': {'min': float('inf'), 'max': float('-inf')},
            'y_range': {'min': float('inf'), 'max': float('-inf')},
            'distances': []
        },
        'heading_statistics': {
            'min': float('inf'),
            'max': float('-inf'),
            'all_headings': []
        },
        'size_statistics': {
            'length_range': {'min': float('inf'), 'max': float('-inf')},
            'width_range': {'min': float('inf'), 'max': float('-inf')}
        },
        'agent_counts': defaultdict(int),
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
        
        if targets is None:
            continue
            
        results['total_samples'] += 1
        
        # Check agent states and labels
        if 'agent_states' in targets:
            agent_states = targets['agent_states']
            results['agent_states_found'] += 1
            
            # Convert to numpy if tensor
            if isinstance(agent_states, torch.Tensor):
                agent_states = agent_states.cpu().numpy()
            
            # Get labels if available
            agent_labels = None
            if 'agent_labels' in targets:
                agent_labels = targets['agent_labels']
                results['agent_labels_found'] += 1
                if isinstance(agent_labels, torch.Tensor):
                    agent_labels = agent_labels.cpu().numpy()
            
            # Check shape (should be [30, 5] for max 30 agents with x, y, heading, length, width)
            if agent_states.shape == (30, 5):
                # Count valid agents
                if agent_labels is not None:
                    num_valid_agents = np.sum(agent_labels)
                    results['agent_counts'][int(num_valid_agents)] += 1
                    
                    if num_valid_agents > 0:
                        results['samples_with_agents'] += 1
                        
                        # Extract valid agent data
                        valid_agents = agent_states[agent_labels]
                        
                        # Position statistics
                        x_positions = valid_agents[:, 0]
                        y_positions = valid_agents[:, 1]
                        headings = valid_agents[:, 2]
                        lengths = valid_agents[:, 3]
                        widths = valid_agents[:, 4]
                        
                        # Calculate distances from ego
                        distances = np.sqrt(x_positions**2 + y_positions**2)
                        results['position_statistics']['distances'].extend(distances.tolist())
                        
                        # Update ranges
                        results['position_statistics']['x_range']['min'] = min(results['position_statistics']['x_range']['min'], x_positions.min())
                        results['position_statistics']['x_range']['max'] = max(results['position_statistics']['x_range']['max'], x_positions.max())
                        results['position_statistics']['y_range']['min'] = min(results['position_statistics']['y_range']['min'], y_positions.min())
                        results['position_statistics']['y_range']['max'] = max(results['position_statistics']['y_range']['max'], y_positions.max())
                        
                        results['heading_statistics']['min'] = min(results['heading_statistics']['min'], headings.min())
                        results['heading_statistics']['max'] = max(results['heading_statistics']['max'], headings.max())
                        results['heading_statistics']['all_headings'].extend(headings.tolist())
                        
                        results['size_statistics']['length_range']['min'] = min(results['size_statistics']['length_range']['min'], lengths.min())
                        results['size_statistics']['length_range']['max'] = max(results['size_statistics']['length_range']['max'], lengths.max())
                        results['size_statistics']['width_range']['min'] = min(results['size_statistics']['width_range']['min'], widths.min())
                        results['size_statistics']['width_range']['max'] = max(results['size_statistics']['width_range']['max'], widths.max())
                    else:
                        # Check if all states are zero
                        if np.all(agent_states == 0):
                            results['all_agents_zero'] += 1
                
                # Store examples
                if len(results['examples']) < 5 and agent_labels is not None:
                    num_valid = int(np.sum(agent_labels)) if agent_labels is not None else 0
                    results['examples'].append({
                        'token': token,
                        'num_valid_agents': num_valid,
                        'first_agent': agent_states[0].tolist() if num_valid > 0 else None,
                        'all_zero': np.all(agent_states == 0)
                    })
            else:
                print(f"Unexpected agent_states shape: {agent_states.shape} for token {token}")
        else:
            print(f"No agent_states found for token {token}")
    
    return results

def main():
    cache_dir = "/workspace/navsim_workspace/cache/bench2drive_Base_cache"
    
    print(f"Verifying Bench2Drive agent states coordinates...")
    print(f"Cache dir: {cache_dir}")
    
    results = verify_agent_coordinates(cache_dir)
    
    print(f"\n=== Agent States Verification Results ===")
    print(f"Total samples analyzed: {results['total_samples']}")
    print(f"Agent states found: {results['agent_states_found']}")
    print(f"Agent labels found: {results['agent_labels_found']}")
    print(f"Samples with valid agents: {results['samples_with_agents']}")
    print(f"Samples with all zeros: {results['all_agents_zero']}")
    
    if results['agent_states_found'] > 0:
        agent_presence_rate = results['samples_with_agents'] / results['agent_states_found'] * 100
        print(f"Agent presence rate: {agent_presence_rate:.1f}%")
    
    print(f"\n=== Agent Count Distribution ===")
    for count in sorted(results['agent_counts'].keys()):
        num_samples = results['agent_counts'][count]
        pct = num_samples / results['total_samples'] * 100
        print(f"{count} agents: {num_samples} samples ({pct:.1f}%)")
    
    if results['position_statistics']['distances']:
        print(f"\n=== Position Statistics (Ego-Relative) ===")
        distances = np.array(results['position_statistics']['distances'])
        print(f"Distance from ego:")
        print(f"  Mean: {distances.mean():.2f} m")
        print(f"  Max: {distances.max():.2f} m")
        print(f"  Min: {distances.min():.2f} m")
        print(f"  Within 32m (lidar range): {np.sum(distances <= 32) / len(distances) * 100:.1f}%")
        
        print(f"\nX coordinate range: [{results['position_statistics']['x_range']['min']:.2f}, {results['position_statistics']['x_range']['max']:.2f}] m")
        print(f"Y coordinate range: [{results['position_statistics']['y_range']['min']:.2f}, {results['position_statistics']['y_range']['max']:.2f}] m")
    else:
        print(f"\n✗ No valid agents found in any samples!")
    
    if results['heading_statistics']['all_headings']:
        print(f"\n=== Heading Statistics ===")
        headings = np.array(results['heading_statistics']['all_headings'])
        print(f"Min heading: {results['heading_statistics']['min']:.3f} rad")
        print(f"Max heading: {results['heading_statistics']['max']:.3f} rad")
        print(f"Mean heading: {headings.mean():.3f} rad")
        print(f"Std heading: {headings.std():.3f} rad")
    
    if results['size_statistics']['length_range']['min'] != float('inf'):
        print(f"\n=== Size Statistics ===")
        print(f"Length range: [{results['size_statistics']['length_range']['min']:.2f}, {results['size_statistics']['length_range']['max']:.2f}] m")
        print(f"Width range: [{results['size_statistics']['width_range']['min']:.2f}, {results['size_statistics']['width_range']['max']:.2f}] m")
    
    if results['examples']:
        print(f"\n=== Example Agent States ===")
        for i, example in enumerate(results['examples']):
            print(f"\nExample {i+1} (token: {example['token']}):")
            print(f"  Valid agents: {example['num_valid_agents']}")
            print(f"  All zeros: {example['all_zero']}")
            if example['first_agent'] and example['num_valid_agents'] > 0:
                print(f"  First agent (x, y, heading, length, width): {example['first_agent']}")
    
    # Summary
    if results['all_agents_zero'] == results['agent_states_found']:
        print(f"\n✗ CRITICAL: All agent states are zero in all samples!")
        print(f"This matches the issue reported in BENCH2DRIVE_CACHE_INVESTIGATION.md")
    elif results['samples_with_agents'] == 0:
        print(f"\n✗ WARNING: No valid agents found in any samples!")
    else:
        print(f"\n✓ Agent states contain valid data in {results['samples_with_agents']} samples")

if __name__ == "__main__":
    main()