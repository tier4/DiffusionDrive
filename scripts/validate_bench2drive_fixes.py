"""Validation script for Bench2Drive fixes."""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader, Bench2DriveConfig


def validate_heading_distribution(loader, num_samples=1000):
    """Validate heading distribution after fix."""
    print("\n=== Validating Heading Distribution ===")

    headings = []

    for i, token in enumerate(tqdm(loader.scene_tokens[:num_samples], desc="Processing scenes")):
        if i >= num_samples:
            break

        try:
            scene = loader.get_scene(token)
            # Extract ego status from first frame
            anno = scene._load_annotation(0)
            ego_status = scene._extract_ego_status(anno)
            headings.append(ego_status.ego_pose[2])
        except Exception as e:
            print(f"Error processing scene {token}: {e}")
            continue

    headings = np.array(headings)

    print(f"Samples processed: {len(headings)}")
    print(f"Heading stats:")
    print(f"  Min: {np.min(headings):.3f} rad ({np.degrees(np.min(headings)):.1f}°)")
    print(f"  Max: {np.max(headings):.3f} rad ({np.degrees(np.max(headings)):.1f}°)")
    print(f"  Mean: {np.mean(headings):.3f} rad")
    print(f"  Std: {np.std(headings):.3f} rad")

    # Success criteria: std > 0.1 rad (similar to NavSim)
    success = np.std(headings) > 0.1
    print(f"✓ PASS: Heading std > 0.1 rad" if success else "✗ FAIL: Heading std <= 0.1 rad")

    return success, headings


def validate_trajectory_ego_relative(loader, num_samples=1000):
    """Validate trajectories are ego-relative."""
    print("\n=== Validating Trajectory Ego-Relative Accuracy ===")

    start_distances = []

    for i, token in enumerate(tqdm(loader.scene_tokens[:num_samples], desc="Processing scenes")):
        if i >= num_samples:
            break

        try:
            scene = loader.get_scene(token)
            trajectory = scene.get_future_trajectory()

            # Check first waypoint distance from ego (should be small if ego-relative)
            first_waypoint = trajectory[0].numpy()
            distance = np.sqrt(first_waypoint[0] ** 2 + first_waypoint[1] ** 2)
            start_distances.append(distance)
        except Exception as e:
            print(f"Error processing scene {token}: {e}")
            continue

    start_distances = np.array(start_distances)

    print(f"Samples processed: {len(start_distances)}")
    print(f"First waypoint distance from ego:")
    print(f"  Min: {np.min(start_distances):.3f} m")
    print(f"  Max: {np.max(start_distances):.3f} m")
    print(f"  Mean: {np.mean(start_distances):.3f} m")
    print(f"  95th percentile: {np.percentile(start_distances, 95):.3f} m")

    # Success criteria: >95% should start within 1m of ego
    within_1m = np.sum(start_distances < 1.0) / len(start_distances)
    success = within_1m > 0.95
    print(f"Trajectories starting within 1m: {within_1m*100:.1f}%")
    print(
        f"✓ PASS: >95% trajectories ego-relative"
        if success
        else "✗ FAIL: <95% trajectories ego-relative"
    )

    return success, start_distances


def validate_agent_detection(loader, num_samples=1000):
    """Validate agent detection after fix."""
    print("\n=== Validating Agent Detection ===")

    scenes_with_agents = 0
    total_agents_detected = 0
    agent_types_count = {5: 0, 6: 0}  # vehicle: 5, pedestrian: 6

    for i, token in enumerate(tqdm(loader.scene_tokens[:num_samples], desc="Processing scenes")):
        if i >= num_samples:
            break

        try:
            scene = loader.get_scene(token)
            agent_states, agent_labels, agent_types = scene.get_agents()

            if torch.any(agent_labels):
                scenes_with_agents += 1
                num_agents = torch.sum(agent_labels).item()
                total_agents_detected += num_agents

                # Count agent types
                for j in range(len(agent_labels)):
                    if agent_labels[j]:
                        agent_type = agent_types[j].item()
                        if agent_type in agent_types_count:
                            agent_types_count[agent_type] += 1

        except Exception as e:
            print(f"Error processing scene {token}: {e}")
            continue

    detection_rate = scenes_with_agents / min(num_samples, len(loader.scene_tokens))
    avg_agents_per_scene = total_agents_detected / min(num_samples, len(loader.scene_tokens))

    print(f"Samples processed: {min(num_samples, len(loader.scene_tokens))}")
    print(f"Scenes with detected agents: {scenes_with_agents} ({detection_rate*100:.1f}%)")
    print(f"Total agents detected: {total_agents_detected}")
    print(f"Average agents per scene: {avg_agents_per_scene:.2f}")
    print(f"Agent types:")
    print(f"  Vehicles (class 5): {agent_types_count[5]}")
    print(f"  Pedestrians (class 6): {agent_types_count[6]}")

    # Success criteria: >50% detection rate
    success = detection_rate > 0.5
    print(f"✓ PASS: >50% agent detection rate" if success else "✗ FAIL: <50% agent detection rate")

    return success, detection_rate


def validate_bev_content(loader, num_samples=500):
    """Validate BEV maps contain vehicles/pedestrians."""
    print("\n=== Validating BEV Content ===")

    scenes_with_vehicles = 0
    scenes_with_pedestrians = 0

    for i, token in enumerate(tqdm(loader.scene_tokens[:num_samples], desc="Processing scenes")):
        if i >= num_samples:
            break

        try:
            scene = loader.get_scene(token)
            bev_map = scene.get_bev_semantic_map()

            unique_classes = torch.unique(bev_map).numpy()

            if 5 in unique_classes:  # Vehicle class
                scenes_with_vehicles += 1
            if 6 in unique_classes:  # Pedestrian class
                scenes_with_pedestrians += 1

        except Exception as e:
            print(f"Error processing scene {token}: {e}")
            continue

    vehicle_rate = scenes_with_vehicles / min(num_samples, len(loader.scene_tokens))
    pedestrian_rate = scenes_with_pedestrians / min(num_samples, len(loader.scene_tokens))

    print(f"Samples processed: {min(num_samples, len(loader.scene_tokens))}")
    print(f"BEV maps with vehicles (class 5): {scenes_with_vehicles} ({vehicle_rate*100:.1f}%)")
    print(
        f"BEV maps with pedestrians (class 6): {scenes_with_pedestrians} ({pedestrian_rate*100:.1f}%)"
    )

    # Success criteria: >50% with vehicles, ~0.3% with pedestrians
    success = vehicle_rate > 0.5
    print(f"✓ PASS: >50% BEV vehicle presence" if success else "✗ FAIL: <50% BEV vehicle presence")

    return success, (vehicle_rate, pedestrian_rate)


def main():
    """Run all validation tests."""
    print("=== Bench2Drive Fixes Validation ===")

    # Configuration
    data_root = Path("/workspace/Bench2Drive-mini")

    # Get available scenarios
    scenarios = [d.name for d in data_root.iterdir() if d.is_dir()]
    print(f"Found {len(scenarios)} scenarios in mini dataset")

    # Create scene loader
    config = Bench2DriveConfig(
        data_root=data_root,
        scenarios=scenarios[:10],  # Use first 10 scenarios for validation
        sampling_rate=5,
        num_frames=30,
    )

    print("\nInitializing scene loader...")
    loader = Bench2DriveSceneLoader(config)
    print(f"Total scenes available: {len(loader.scene_tokens)}")

    # Run validation tests
    results = {}

    # 1. Heading distribution
    success, data = validate_heading_distribution(loader, num_samples=100)
    results["heading_distribution"] = {"success": success, "data": data}

    # 2. Trajectory ego-relative
    success, data = validate_trajectory_ego_relative(loader, num_samples=100)
    results["trajectory_ego_relative"] = {"success": success, "data": data}

    # 3. Agent detection
    success, data = validate_agent_detection(loader, num_samples=100)
    results["agent_detection"] = {"success": success, "data": data}

    # 4. BEV content
    success, data = validate_bev_content(loader, num_samples=50)
    results["bev_content"] = {"success": success, "data": data}

    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    all_success = all(result["success"] for result in results.values())

    for test_name, result in results.items():
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"{test_name}: {status}")

    if all_success:
        print("\n✓ ALL TESTS PASSED - Bench2Drive fixes validated!")
    else:
        print("\n✗ SOME TESTS FAILED - Please review the results above")

    return results


if __name__ == "__main__":
    results = main()
