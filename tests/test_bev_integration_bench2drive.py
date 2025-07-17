"""
Integration test for BEV semantic generation with Bench2Drive dataset.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig, 
    Bench2DriveSceneLoader
)
from navsim.common.bench2drive_scene import Bench2DriveScene


class TestBench2DriveBEVIntegration:
    """Test BEV generation with real Bench2Drive data."""
    
    @pytest.fixture
    def scene_loader(self):
        """Create a scene loader with mini dataset."""
        config = Bench2DriveConfig(
            data_root=Path("/workspace/Bench2Drive-mini"),
            scenarios=["ConstructionObstacle_Town05_Route68_Weather8"],
            sampling_rate=5,
            num_frames=30,
            num_history_frames=4,
            num_future_frames=26,
            extract_tar=False,
        )
        return Bench2DriveSceneLoader(config)
    
    def test_bev_generation_from_scene(self, scene_loader):
        """Test BEV generation from a real scene."""
        if len(scene_loader) == 0:
            pytest.skip("No scenes available in test dataset")
        
        # Get first scene
        token = scene_loader.scene_tokens[0]
        scene = scene_loader.get_scene(token)
        
        # Test BEV generation at different frames
        for frame_idx in [0, 5, 10]:
            bev_map = scene.get_bev_semantic_map(frame_idx)
            
            # Check shape and type
            assert bev_map.shape == (128, 256)
            assert bev_map.dtype == torch.float32
            
            # Check value range
            assert bev_map.min() >= 0
            assert bev_map.max() <= 6
            
            # Check that it's not all zeros
            assert bev_map.sum() > 0, f"BEV map at frame {frame_idx} is all zeros"
            
            # Check semantic classes present
            unique_classes = torch.unique(bev_map).numpy()
            print(f"Frame {frame_idx} - Unique classes: {unique_classes}")
            
            # Should have at least background (0) and road (1)
            assert 0 in unique_classes, "Background class missing"
            if frame_idx < len(scene.frames) - 8:  # If we have future frames
                assert 1 in unique_classes, "Road class missing"
    
    def test_bev_with_agents(self, scene_loader):
        """Test that vehicles appear in BEV map."""
        if len(scene_loader) == 0:
            pytest.skip("No scenes available")
        
        # Find a scene with agents
        scene_with_agents = None
        for token in scene_loader.scene_tokens[:5]:  # Check first 5 scenes
            scene = scene_loader.get_scene(token)
            agents, labels = scene.get_agents(0)
            if labels.any():
                scene_with_agents = scene
                break
        
        if scene_with_agents is None:
            pytest.skip("No scenes with agents found")
        
        # Generate BEV
        bev_map = scene_with_agents.get_bev_semantic_map(0)
        
        # Check for vehicle class (5)
        unique_classes = torch.unique(bev_map).numpy()
        print(f"BEV with agents - classes: {unique_classes}")
        
        # If agents were detected, vehicles should appear in BEV
        agents, labels = scene_with_agents.get_agents(0)
        if labels.any():
            assert 5 in unique_classes, "Vehicle class missing despite agents detected"
    
    def test_bev_visualization(self, scene_loader, tmp_path):
        """Visualize BEV generation for debugging."""
        if len(scene_loader) == 0:
            pytest.skip("No scenes available")
        
        # Get first scene
        token = scene_loader.scene_tokens[0]
        scene = scene_loader.get_scene(token)
        
        # Generate BEV for middle frame
        frame_idx = 15
        bev_map = scene.get_bev_semantic_map(frame_idx).numpy()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # BEV semantic map
        cmap = plt.cm.colors.ListedColormap([
            '#2E2E2E',  # 0: Background
            '#808080',  # 1: Road  
            '#FFA500',  # 2: Walkways
            '#FFFF00',  # 3: Lane centerlines
            '#800080',  # 4: Static objects
            '#FF0000',  # 5: Vehicles
            '#00FF00',  # 6: Pedestrians
        ])
        
        im1 = ax1.imshow(bev_map, cmap=cmap, origin='lower', vmin=0, vmax=6)
        ax1.set_title(f'BEV Semantic Map (Frame {frame_idx})')
        ax1.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im1, ax=ax1, ticks=range(7))
        cbar.ax.set_yticklabels(['Bg', 'Road', 'Walk', 'Lane', 'Static', 'Veh', 'Ped'])
        
        # Class distribution
        unique, counts = np.unique(bev_map, return_counts=True)
        class_names = ['Background', 'Road', 'Walkway', 'Lane', 'Static', 'Vehicle', 'Pedestrian']
        
        bars = ax2.bar(unique, counts)
        ax2.set_xticks(unique)
        ax2.set_xticklabels([class_names[int(u)] for u in unique], rotation=45)
        ax2.set_ylabel('Pixel Count')
        ax2.set_title('Class Distribution')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add percentages
        total = counts.sum()
        for bar, count in zip(bars, counts):
            pct = 100 * count / total
            if pct > 1:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        output_path = tmp_path / "bev_generation_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to: {output_path}")
        assert output_path.exists()
    
    def test_trajectory_consistency(self, scene_loader):
        """Test that trajectory creates consistent road patterns."""
        if len(scene_loader) == 0:
            pytest.skip("No scenes available")
        
        # Get a scene
        token = scene_loader.scene_tokens[0]
        scene = scene_loader.get_scene(token)
        
        # Generate BEV maps for consecutive frames
        bev_maps = []
        for i in range(3):
            bev_map = scene.get_bev_semantic_map(i).numpy()
            bev_maps.append(bev_map)
        
        # Check that road patterns are somewhat consistent
        # The road should exist in similar regions
        road_masks = [(bev == 1) for bev in bev_maps]
        
        # Calculate overlap between consecutive frames
        for i in range(len(road_masks) - 1):
            overlap = np.logical_and(road_masks[i], road_masks[i+1])
            overlap_ratio = overlap.sum() / max(road_masks[i].sum(), road_masks[i+1].sum())
            
            # Should have some overlap (at least 20%)
            assert overlap_ratio > 0.2, f"Road overlap too low between frames {i} and {i+1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])