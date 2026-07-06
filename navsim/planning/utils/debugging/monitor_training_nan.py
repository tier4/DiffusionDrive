#!/usr/bin/env python3
"""
Training monitor script to help detect and diagnose NaN issues early.
Run this alongside training to get detailed diagnostics.
"""

import os
import time
import argparse
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime


def parse_tensorboard_events(log_dir):
    """Parse tensorboard events to extract training metrics."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        metrics = {}
        
        # Get scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                latest = events[-1]
                metrics[tag] = {
                    'value': latest.value,
                    'step': latest.step,
                    'wall_time': latest.wall_time
                }
        
        return metrics
    except:
        return {}


def check_training_health(log_dir, config_path=None):
    """Check training health and detect potential issues."""
    issues = []
    warnings = []
    info = []
    
    # Check if training directory exists
    if not Path(log_dir).exists():
        return ["Training directory not found"], [], []
    
    # Parse metrics
    metrics = parse_tensorboard_events(log_dir)
    
    # Check for NaN in losses
    for metric_name, metric_data in metrics.items():
        if 'loss' in metric_name.lower():
            value = metric_data['value']
            if np.isnan(value):
                issues.append(f"NaN detected in {metric_name} at step {metric_data['step']}")
            elif np.isinf(value):
                issues.append(f"Inf detected in {metric_name} at step {metric_data['step']}")
            elif value > 1000:
                warnings.append(f"Very high loss in {metric_name}: {value:.2f}")
    
    # Check gradient norms
    if 'train/grad_norm' in metrics:
        grad_norm = metrics['train/grad_norm']['value']
        if grad_norm > 100:
            warnings.append(f"High gradient norm: {grad_norm:.2f}")
        elif grad_norm < 0.0001:
            warnings.append(f"Very low gradient norm: {grad_norm:.6f}")
        info.append(f"Current gradient norm: {grad_norm:.4f}")
    
    # Check learning rate
    for key in ['lr', 'learning_rate', 'train/lr']:
        if key in metrics:
            lr = metrics[key]['value']
            info.append(f"Current learning rate: {lr:.2e}")
            break
    
    # Check training progress
    max_step = 0
    for metric_data in metrics.values():
        max_step = max(max_step, metric_data['step'])
    
    if max_step > 0:
        info.append(f"Training at step: {max_step}")
    
    # Load and check config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check for risky settings
        if 'lr' in config or 'learning_rate' in config:
            lr_config = config.get('lr', config.get('learning_rate', 0))
            if lr_config > 1e-3:
                warnings.append(f"High learning rate in config: {lr_config}")
    
    return issues, warnings, info


def monitor_loop(log_dir, interval=30):
    """Continuously monitor training."""
    print(f"Monitoring training in: {log_dir}")
    print(f"Checking every {interval} seconds...\n")
    
    last_issues = []
    
    while True:
        try:
            issues, warnings, info = check_training_health(log_dir)
            
            # Clear screen for fresh display
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"=== Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            
            # Display info
            if info:
                print("📊 Status:")
                for item in info:
                    print(f"   {item}")
                print()
            
            # Display warnings
            if warnings:
                print("⚠️  Warnings:")
                for warning in warnings:
                    print(f"   {warning}")
                print()
            
            # Display issues
            if issues:
                print("❌ ISSUES DETECTED:")
                for issue in issues:
                    print(f"   {issue}")
                print("\n🚨 Training may have encountered NaN! Check logs for details.")
                
                # If new issues detected, save diagnostic info
                if issues != last_issues:
                    diagnostic_file = Path(log_dir) / f"nan_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(diagnostic_file, 'w') as f:
                        f.write("NaN Diagnostic Report\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Time: {datetime.now()}\n\n")
                        f.write("Issues:\n")
                        for issue in issues:
                            f.write(f"- {issue}\n")
                        f.write("\nRecommendations:\n")
                        f.write("1. Check if cache was regenerated with fixed normalizations\n")
                        f.write("2. Reduce learning rate further (try 1e-6)\n")
                        f.write("3. Check autograd anomaly detection output\n")
                        f.write("4. Run validation script on current cache\n")
                    
                    print(f"\n💾 Diagnostic info saved to: {diagnostic_file}")
                
                last_issues = issues
            else:
                print("✅ No issues detected")
            
            print("\nPress Ctrl+C to stop monitoring...")
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nError during monitoring: {e}")
        
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Monitor training for NaN issues')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Training log directory (where tensorboard events are)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Check interval in seconds')
    parser.add_argument('--config', type=str,
                        help='Training config file to check settings')
    parser.add_argument('--once', action='store_true',
                        help='Run check once instead of continuous monitoring')
    
    args = parser.parse_args()
    
    if args.once:
        issues, warnings, info = check_training_health(args.log_dir, args.config)
        
        print("=== Training Health Check ===\n")
        
        if info:
            print("Status:")
            for item in info:
                print(f"  {item}")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        if issues:
            print("\nISSUES:")
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("\n✅ No critical issues found")
    else:
        monitor_loop(args.log_dir, args.interval)


if __name__ == "__main__":
    main()