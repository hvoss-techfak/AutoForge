"""Tests for FlatForge functionality."""

import numpy as np
import os
import pytest

from autoforge.Helper.OutputHelper import generate_flatforge_stls


def test_generate_flatforge_stls_basic(tmp_path):
    """Test basic FlatForge STL generation."""
    # Create a simple test case
    H, W = 10, 10
    
    # Height map: pixels have different heights (layers)
    disc_height_image = np.zeros((H, W), dtype=int)
    disc_height_image[2:8, 2:8] = 5  # Center area has 5 layers
    
    # Material assignment: alternate between 3 materials across 5 layers
    disc_global = np.array([0, 1, 2, 1, 0])  # 5 layers with different materials
    
    # Material data
    material_colors_np = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
    ])
    material_names = ["Brand1 - Red", "Brand2 - Green", "Brand3 - Blue"]
    material_TDs_np = np.array([5.0, 10.0, 50.0])  # Blue is most transparent
    
    # Generate FlatForge STLs
    stl_files = generate_flatforge_stls(
        disc_global=disc_global,
        disc_height_image=disc_height_image,
        material_colors_np=material_colors_np,
        material_names=material_names,
        material_TDs_np=material_TDs_np,
        layer_height=0.04,
        background_height=0.24,
        background_color_hex="#000000",
        maximum_x_y_size=100.0,
        output_folder=str(tmp_path),
        cap_layers=0,
        alpha_mask=None,
    )
    
    # Verify STL files were created
    assert len(stl_files) > 0, "Should generate at least one STL file"
    
    # Check that files exist
    for stl_file in stl_files:
        assert os.path.exists(stl_file), f"STL file {stl_file} should exist"
        assert os.path.getsize(stl_file) > 100, f"STL file {stl_file} should have content"
    
    # Should have STLs for each unique material plus background
    # Materials: 0, 1, 2 (3 unique) + background = 4 files minimum
    assert len(stl_files) >= 4, f"Should have at least 4 STL files (3 materials + background), got {len(stl_files)}"
    
    # Verify the material_mask is 2D (regression test for the 3D mask bug)
    # This is implicitly tested by the function not crashing


def test_generate_flatforge_stls_with_cap(tmp_path):
    """Test FlatForge STL generation with cap layers."""
    H, W = 8, 8
    
    disc_height_image = np.ones((H, W), dtype=int) * 3  # All pixels at 3 layers
    disc_global = np.array([0, 1, 0])  # 3 layers alternating materials
    
    material_colors_np = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
    ])
    material_names = ["Brand1 - Red", "Brand2 - Green"]
    material_TDs_np = np.array([5.0, 30.0])  # Green is more transparent
    
    stl_files = generate_flatforge_stls(
        disc_global=disc_global,
        disc_height_image=disc_height_image,
        material_colors_np=material_colors_np,
        material_names=material_names,
        material_TDs_np=material_TDs_np,
        layer_height=0.04,
        background_height=0.24,
        background_color_hex="#FFFFFF",
        maximum_x_y_size=50.0,
        output_folder=str(tmp_path),
        cap_layers=2,  # Add 2 cap layers
        alpha_mask=None,
    )
    
    assert len(stl_files) > 0, "Should generate STL files"
    
    # Should include a cap layer STL
    cap_stl_found = any("Cap" in os.path.basename(f) for f in stl_files)
    assert cap_stl_found, "Should generate a cap layer STL"
    
    for stl_file in stl_files:
        assert os.path.exists(stl_file), f"STL file {stl_file} should exist"


def test_generate_flatforge_stls_with_alpha_mask(tmp_path):
    """Test FlatForge STL generation with alpha mask."""
    H, W = 10, 10
    
    # Height map
    disc_height_image = np.ones((H, W), dtype=int) * 4
    disc_global = np.array([0, 1, 0, 1])
    
    # Alpha mask: only center 6x6 area is valid
    # Use a 2D alpha mask (H, W) so the function produces a 2D valid_mask
    alpha_mask = np.zeros((H, W), dtype=np.uint8)
    alpha_mask[2:8, 2:8] = 255
    
    material_colors_np = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    material_names = ["Red", "Green"]
    material_TDs_np = np.array([5.0, 25.0])
    
    stl_files = generate_flatforge_stls(
        disc_global=disc_global,
        disc_height_image=disc_height_image,
        material_colors_np=material_colors_np,
        material_names=material_names,
        material_TDs_np=material_TDs_np,
        layer_height=0.04,
        background_height=0.24,
        background_color_hex="#000000",
        maximum_x_y_size=100.0,
        output_folder=str(tmp_path),
        cap_layers=0,
        alpha_mask=alpha_mask,
    )
    
    assert len(stl_files) > 0, "Should generate STL files with alpha mask"
    
    for stl_file in stl_files:
        assert os.path.exists(stl_file), f"STL file {stl_file} should exist"
        assert os.path.getsize(stl_file) > 100, f"STL file {stl_file} should have content"


def test_generate_flatforge_stls_with_clear_areas(tmp_path):
    """Test FlatForge STL generation with clear/transparent areas."""
    H, W = 8, 8
    
    # Height map: all pixels have 4 layers
    disc_height_image = np.ones((H, W), dtype=int) * 4
    
    # Material assignment: only 2 layers have materials, layers 2 and 3 are clear
    # Layer 0: material 0
    # Layer 1: material 1
    # Layer 2: clear (no material in disc_global, but pixels have this height)
    # Layer 3: clear
    # Create an explicit global mapping including -1 for clear layers
    disc_global = np.array([0, 1, -1, -1])  # -1 means clear/no material

    material_colors_np = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    material_names = ["Red", "Green"]
    material_TDs_np = np.array([5.0, 25.0])
    
    # For this test, we'll create a scenario where some pixels have heights
    # but the global assignment leaves them empty
    disc_height_image[4:6, 4:6] = 3  # Some pixels only go to layer 3
    
    stl_files = generate_flatforge_stls(
        disc_global=disc_global,
        disc_height_image=disc_height_image,
        material_colors_np=material_colors_np,
        material_names=material_names,
        material_TDs_np=material_TDs_np,
        layer_height=0.04,
        background_height=0.24,
        background_color_hex="#000000",
        maximum_x_y_size=100.0,
        output_folder=str(tmp_path),
        cap_layers=0,
        alpha_mask=None,
    )
    
    assert len(stl_files) > 0, "Should generate STL files"
    
    # With explicit clear layers we should at least get material + background
    assert len(stl_files) >= 2, f"Should have at least 2 STL files, got {len(stl_files)}"


def test_flatforge_stl_naming(tmp_path):
    """Test that FlatForge STL files are named correctly."""
    H, W = 5, 5
    
    disc_height_image = np.ones((H, W), dtype=int) * 2
    disc_global = np.array([0, 1])
    
    material_colors_np = np.array([
        [1.0, 0.0, 0.0],  # Red = FF0000
        [0.0, 0.5, 1.0],  # Blue-ish = 0080FF
    ])
    material_names = ["BrandA - RedPLA", "BrandB - BluePLA"]
    material_TDs_np = np.array([5.0, 30.0])
    
    stl_files = generate_flatforge_stls(
        disc_global=disc_global,
        disc_height_image=disc_height_image,
        material_colors_np=material_colors_np,
        material_names=material_names,
        material_TDs_np=material_TDs_np,
        layer_height=0.04,
        background_height=0.24,
        background_color_hex="#000000",
        maximum_x_y_size=100.0,
        output_folder=str(tmp_path),
        cap_layers=0,
        alpha_mask=None,
    )
    
    # Check naming format: should include material name and color hex
    stl_names = [os.path.basename(f) for f in stl_files]
    
    # Should have background with hex color
    background_stls = [n for n in stl_names if "Background" in n]
    assert len(background_stls) > 0, "Should have background STL"
    assert any("000000" in n for n in background_stls), "Background should include hex color"
    
    # Material STLs should include sanitized names
    material_stls = [n for n in stl_names if "BrandA" in n or "BrandB" in n]
    assert len(material_stls) >= 2, "Should have STLs for both materials"


def test_flatforge_empty_height_map(tmp_path):
    """Test FlatForge with an empty height map (no layers)."""
    H, W = 5, 5
    
    disc_height_image = np.zeros((H, W), dtype=int)  # No height anywhere
    disc_global = np.array([])
    
    material_colors_np = np.array([
        [1.0, 0.0, 0.0],
    ])
    material_names = ["Red"]
    material_TDs_np = np.array([5.0])
    
    stl_files = generate_flatforge_stls(
        disc_global=disc_global,
        disc_height_image=disc_height_image,
        material_colors_np=material_colors_np,
        material_names=material_names,
        material_TDs_np=material_TDs_np,
        layer_height=0.04,
        background_height=0.24,
        background_color_hex="#000000",
        maximum_x_y_size=100.0,
        output_folder=str(tmp_path),
        cap_layers=0,
        alpha_mask=None,
    )
    
    # Should return empty or minimal files
    assert isinstance(stl_files, list), "Should return a list"
