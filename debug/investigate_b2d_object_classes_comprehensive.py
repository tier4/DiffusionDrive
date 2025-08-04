#!/usr/bin/env python3
"""Comprehensive investigation of Bench2Drive object classes.

This script analyzes all annotation files to discover:
1. All unique object classes (string and numeric)
2. Class occurrence counts
3. Presence of CARLA numeric labels (especially 12 for Pedestrian, 13 for Rider)
4. Object attributes for each class type
"""

import json
import gzip
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random
from typing import Dict, List, Any, Set, Tuple

# CARLA class labels for reference
CARLA_CLASSES = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail",
}


def analyze_object_classes(bench2drive_dir: str, max_files: int = None) -> Dict[str, Any]:
    """Analyze object classes across all Bench2Drive annotation files."""

    base_path = Path(bench2drive_dir)

    # Results storage
    results = {
        "total_files_analyzed": 0,
        "total_objects": 0,
        "class_counts": defaultdict(int),
        "class_types": set(),  # Track if classes are string or numeric
        "class_samples": defaultdict(list),  # Store sample objects for each class
        "ego_vehicle_count": 0,
        "files_with_numeric_classes": 0,
        "files_with_string_classes": 0,
        "mixed_class_files": 0,
        "pedestrian_data": {"found": False, "count": 0, "formats": set()},
        "carla_numeric_classes_found": set(),
        "unknown_classes": set(),
    }

    # Collect all annotation files
    anno_files = []
    for scenario_dir in base_path.iterdir():
        if scenario_dir.is_dir():
            anno_dir = scenario_dir / "anno"
            if anno_dir.exists():
                anno_files.extend(list(anno_dir.glob("*.json.gz")))

    # Limit files if requested
    if max_files:
        random.shuffle(anno_files)
        anno_files = anno_files[:max_files]

    print(f"Analyzing {len(anno_files)} annotation files...")

    for anno_file in tqdm(anno_files, total=len(anno_files)):
        try:
            with gzip.open(anno_file, "rt") as f:
                anno = json.load(f)

            results["total_files_analyzed"] += 1

            if "bounding_boxes" not in anno:
                continue

            file_has_numeric = False
            file_has_string = False

            for bbox in anno["bounding_boxes"]:
                results["total_objects"] += 1

                obj_class = bbox.get("class", "unknown")

                # Track class type
                if isinstance(obj_class, (int, float)):
                    file_has_numeric = True
                    obj_class = int(obj_class)
                    results["class_types"].add("numeric")

                    # Check if it's a known CARLA class
                    if obj_class in CARLA_CLASSES:
                        results["carla_numeric_classes_found"].add(obj_class)
                    else:
                        results["unknown_classes"].add(f"numeric_{obj_class}")

                elif isinstance(obj_class, str):
                    file_has_string = True
                    results["class_types"].add("string")

                    if obj_class not in [
                        "ego_vehicle",
                        "vehicle",
                        "car",
                        "truck",
                        "bus",
                        "motorcycle",
                        "bicycle",
                        "pedestrian",
                        "walker",
                        "person",
                        "rider",
                        "cyclist",
                        "traffic_light",
                        "traffic_sign",
                        "static",
                        "dynamic",
                        "barrier",
                        "cone",
                    ]:
                        results["unknown_classes"].add(f"string_{obj_class}")

                # Count occurrences
                results["class_counts"][str(obj_class)] += 1

                # Special tracking for ego vehicle
                if obj_class == "ego_vehicle":
                    results["ego_vehicle_count"] += 1

                # Check for pedestrians
                if isinstance(obj_class, int) and obj_class in [12, 13]:
                    results["pedestrian_data"]["found"] = True
                    results["pedestrian_data"]["count"] += 1
                    results["pedestrian_data"]["formats"].add(f"numeric_{obj_class}")
                elif isinstance(obj_class, str) and obj_class in [
                    "pedestrian",
                    "walker",
                    "person",
                    "rider",
                    "cyclist",
                ]:
                    results["pedestrian_data"]["found"] = True
                    results["pedestrian_data"]["count"] += 1
                    results["pedestrian_data"]["formats"].add(f"string_{obj_class}")

                # Store sample objects (limit to 5 per class)
                class_key = str(obj_class)
                if len(results["class_samples"][class_key]) < 5:
                    sample = {
                        "file": str(anno_file),
                        "class": obj_class,
                        "location": bbox.get("location", []),
                        "rotation": bbox.get("rotation", []),
                        "extent": bbox.get("extent", []),
                        "has_world2ego": "world2ego" in bbox,
                        "has_world2vehicle": "world2vehicle" in bbox,
                        "has_speed": "speed" in bbox,
                        "all_fields": list(bbox.keys()),
                    }
                    results["class_samples"][class_key].append(sample)

            # Track file class type patterns
            if file_has_numeric and file_has_string:
                results["mixed_class_files"] += 1
            elif file_has_numeric:
                results["files_with_numeric_classes"] += 1
            elif file_has_string:
                results["files_with_string_classes"] += 1

        except Exception as e:
            print(f"Error processing {anno_file}: {e}")

    # Convert sets to lists for JSON serialization
    results["class_types"] = list(results["class_types"])
    results["carla_numeric_classes_found"] = sorted(list(results["carla_numeric_classes_found"]))
    results["unknown_classes"] = sorted(list(results["unknown_classes"]))
    results["pedestrian_data"]["formats"] = list(results["pedestrian_data"]["formats"])

    return results


def print_analysis_summary(results: Dict[str, Any]):
    """Print a comprehensive summary of the analysis."""

    print("\n" + "=" * 80)
    print("BENCH2DRIVE OBJECT CLASS ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nFiles Analyzed: {results['total_files_analyzed']}")
    print(f"Total Objects: {results['total_objects']}")

    print(f"\nClass Format Distribution:")
    print(f"  - Files with numeric classes: {results['files_with_numeric_classes']}")
    print(f"  - Files with string classes: {results['files_with_string_classes']}")
    print(f"  - Files with mixed classes: {results['mixed_class_files']}")
    print(f"  - Class types found: {results['class_types']}")

    print("\n" + "-" * 40)
    print("OBJECT CLASS COUNTS")
    print("-" * 40)

    # Sort by count
    sorted_classes = sorted(results["class_counts"].items(), key=lambda x: x[1], reverse=True)

    for class_name, count in sorted_classes[:20]:  # Top 20 classes
        percentage = (count / results["total_objects"]) * 100

        # Add CARLA class name if numeric
        carla_name = ""
        try:
            class_num = int(class_name)
            if class_num in CARLA_CLASSES:
                carla_name = f" ({CARLA_CLASSES[class_num]})"
        except:
            pass

        print(f"  {class_name}{carla_name}: {count:,} ({percentage:.2f}%)")

    if len(sorted_classes) > 20:
        print(f"  ... and {len(sorted_classes) - 20} more classes")

    print("\n" + "-" * 40)
    print("PEDESTRIAN DATA ANALYSIS")
    print("-" * 40)

    if results["pedestrian_data"]["found"]:
        print(f"✓ Pedestrians FOUND in dataset!")
        print(f"  Total pedestrian objects: {results['pedestrian_data']['count']}")
        print(f"  Formats found: {results['pedestrian_data']['formats']}")
    else:
        print("✗ No pedestrians found in analyzed files")

    print("\n" + "-" * 40)
    print("CARLA NUMERIC CLASSES FOUND")
    print("-" * 40)

    if results["carla_numeric_classes_found"]:
        print("The following CARLA numeric classes were detected:")
        for class_id in results["carla_numeric_classes_found"]:
            count = results["class_counts"].get(str(class_id), 0)
            print(f"  Class {class_id}: {CARLA_CLASSES[class_id]} ({count:,} objects)")
    else:
        print("No CARLA numeric classes found")

    print("\n" + "-" * 40)
    print("UNKNOWN/UNEXPECTED CLASSES")
    print("-" * 40)

    if results["unknown_classes"]:
        print("The following unexpected classes were found:")
        for unknown in results["unknown_classes"]:
            print(f"  - {unknown}")
    else:
        print("No unknown classes found")

    print("\n" + "-" * 40)
    print("SAMPLE OBJECT STRUCTURES")
    print("-" * 40)

    # Show samples for key classes
    key_classes = ["ego_vehicle", "vehicle", "14", "12", "13", "pedestrian", "traffic_light"]

    for class_name in key_classes:
        if class_name in results["class_samples"] and results["class_samples"][class_name]:
            sample = results["class_samples"][class_name][0]
            print(f"\nSample '{class_name}' object:")
            print(f"  Fields: {sample['all_fields']}")
            print(f"  Has world2ego: {sample['has_world2ego']}")
            print(f"  Has world2vehicle: {sample['has_world2vehicle']}")
            print(f"  Has speed: {sample['has_speed']}")


def save_results(results: Dict[str, Any], output_path: str):
    """Save analysis results to JSON file."""
    # Convert defaultdicts to regular dicts for JSON serialization
    results["class_counts"] = dict(results["class_counts"])
    results["class_samples"] = dict(results["class_samples"])

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    bench2drive_dir = "/workspace/Bench2Drive-Base"
    output_path = "analysis_results/b2d_object_classes_comprehensive.json"

    # Ensure output directory exists
    Path("analysis_results").mkdir(exist_ok=True)

    # Run analysis
    results = analyze_object_classes(bench2drive_dir)

    # Print summary
    print_analysis_summary(results)

    # Save results
    save_results(results, output_path)

    # Generate recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FIX PLAN")
    print("=" * 80)

    if "numeric" in results["class_types"]:
        print("\n1. Bench2Drive uses CARLA numeric class labels!")
        print("   - Must handle numeric classes in addition to strings")
        print("   - Create mapping for CARLA numeric → NavSim classes")

    if results["pedestrian_data"]["found"]:
        print("\n2. Pedestrians ARE present in the dataset!")
        print(f"   - Found in formats: {results['pedestrian_data']['formats']}")
        print("   - Must update agent detection to include pedestrian classes")
        print("   - Must update BEV rendering to show pedestrians (class 6)")

    if results["carla_numeric_classes_found"]:
        print("\n3. Multiple CARLA vehicle types detected:")
        for class_id in [14, 15, 16, 18, 19]:
            if class_id in results["carla_numeric_classes_found"]:
                print(f"   - Class {class_id} ({CARLA_CLASSES[class_id]}) → NavSim class 5")


if __name__ == "__main__":
    main()
