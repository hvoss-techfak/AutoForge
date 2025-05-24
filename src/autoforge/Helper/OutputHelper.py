import io
import json
import os
import struct
import uuid

import numpy as np
import trimesh
from trimesh import Trimesh

from autoforge.Helper.FilamentHelper import load_materials_data


def extract_filament_swaps(disc_global, disc_height_image, background_layers):
    """
    Given the discrete global material assignment (disc_global) and the discrete height image,
    extract the list of material indices (one per swap point) and the corresponding slider
    values (which indicate at which layer the material change occurs).

    Args:
        disc_global (jnp.ndarray): Discrete global material assignments.
        disc_height_image (jnp.ndarray): Discrete height image.
        background_layers (int): Number of background layers.

    Returns:
        tuple: A tuple containing:
            - filament_indices (list): List of material indices for each swap point.
            - slider_values (list): List of layer numbers where a material change occurs.
    """
    # L is the total number of layers printed (maximum value in the height image)
    L = int(np.max(np.array(disc_height_image)))
    filament_indices = []
    slider_values = []
    prev = int(disc_global[0])
    for i in range(L):
        current = int(disc_global[i])
        # If this is the first layer or the material changes from the previous layer…
        if current != prev:
            slider = i + background_layers
            slider_values.append(slider)
            filament_indices.append(prev)
        prev = current
    # Add the last material index
    filament_indices.append(prev)
    slider = i + background_layers
    slider_values.append(slider)

    return filament_indices, slider_values


def generate_project_file(
    project_filename,
    args,
    disc_global,
    disc_height_image,
    width_mm,
    height_mm,
    stl_filename,
    csv_filename,
):
    """
    Export a project file containing the printing parameters, including:
      - Key dimensions and layer information (from your command-line args and computed outputs)
      - The filament_set: a list of filament definitions (each corresponding to a color swap)
        where the same material may be repeated if used at different swap points.
      - slider_values: a list of layer numbers (indices) where a filament swap occurs.

    The filament_set entries are built using the full material data from the CSV file.

    Args:
        project_filename (str): Path to the output project file.
        args (Namespace): Command-line arguments containing printing parameters.
        disc_global (jnp.ndarray): Discrete global material assignments.
        disc_height_image (jnp.ndarray): Discrete height image.
        width_mm (float): Width of the model in millimeters.
        height_mm (float): Height of the model in millimeters.
        stl_filename (str): Path to the STL file.
        csv_filename (str): Path to the CSV file containing material data.
    """
    # Compute the number of background layers (as in your main())
    background_layers = int(args.background_height / args.layer_height)

    # Load full material data from CSV
    material_data = load_materials_data(args)

    # Extract the swap points from the discrete solution
    filament_indices, slider_values = extract_filament_swaps(
        disc_global, disc_height_image, background_layers
    )

    # Build the filament_set list. For each swap point, we look up the corresponding material from CSV.
    # Here we map CSV columns to the project file’s expected keys.
    filament_set = []
    for idx in filament_indices:
        mat = material_data[idx]
        filament_entry = {
            "Brand": mat["Brand"],
            "Color": mat["Color"],
            "Name": mat["Name"],
            # Convert Owned to a boolean (in case it is read as a string)
            "Owned": str(mat["Owned"]).strip().lower() == "true"
            if "Owned" in mat
            else False,
            "Transmissivity": float(mat["Transmissivity"])
            if not float(mat["Transmissivity"]).is_integer()
            else int(mat["Transmissivity"]),
            "Type": mat["Type"] if "Type" in mat else "PLA",
            "uuid": mat["Uuid"] if "Uuid" in mat else str(uuid.uuid4()),
        }
        filament_set.append(filament_entry)

    # add black as the first filament with background height as the first slider value
    filament_set.insert(
        0,
        {
            "Brand": "Autoforge",
            "Color": args.background_color,
            "Name": "Background",
            "Owned": False,
            "Transmissivity": 0.1,
            "Type": "PLA",
            "uuid": str(uuid.uuid4()),
        },
    )
    # add black to slider value
    slider_values.insert(0, (args.background_height // args.layer_height))

    # reverse order of filament set
    filament_set = filament_set[::-1]

    # Build the project file dictionary.
    # Many keys are filled in with default or derived values.
    project_data = {
        "base_layer_height": args.layer_height,  # you may adjust this if needed
        "blue_shift": 0,
        "border_height": args.background_height,  # here we use the background height
        "border_width": 3,
        "borderless": True,
        "bright_adjust_zero": False,
        "brightness_compensation_name": "Standard",
        "bw_tolerance": 8,
        "color_match_method": 0,
        "depth_mode": 2,
        "edit_image": False,
        "extra_gap": 2,
        "filament_set": filament_set,
        "flatten": False,
        "full_range": False,
        "green_shift": 0,
        "gs_threshold": 0,
        "height_in_mm": height_mm,
        "hsl_invert": False,
        "ignore_blue": False,
        "ignore_green": False,
        "ignore_red": False,
        "invert_blue": False,
        "invert_green": False,
        "invert_red": False,
        "inverted_color_pop": False,
        "layer_height": args.layer_height,
        "legacy_luminance": False,
        "light_intensity": -1,
        "light_temperature": 1,
        "lighting_visualizer": 0,
        "luminance_factor": 0,
        "luminance_method": 2,
        "luminance_offset": 0,
        "luminance_offset_max": 100,
        "luminance_power": 2,
        "luminance_weight": 100,
        "max_depth": args.background_height + args.layer_height * args.max_layers,
        "median": 0,
        "mesh_style_edit": True,
        "min_depth": 0.48,
        "min_detail": 0.2,
        "negative": True,
        "red_shift": 0,
        "reverse_litho": True,
        "slider_values": slider_values,
        "smoothing": 0,
        "srgb_linearize": False,
        "stl": os.path.basename(stl_filename),
        "strict_tolerance": False,
        "transparency": True,
        "version": "0.7.0",
        "width_in_mm": width_mm,
    }

    # Write out the project file as JSON
    with open(project_filename, "w") as f:
        json.dump(project_data, f, indent=4)


def generate_stl(
    height_map, filename, background_height, maximum_x_y_size, alpha_mask=None
):
    """
    Generate a binary STL file from a height map with an optional alpha mask.
    If alpha_mask is provided, vertices where alpha < 128 are omitted.
    This function builds a manifold mesh consisting of:
      - a top surface (only quads whose four vertices are valid),
      - side walls along the boundary edges of the top surface, and
      - a bottom face covering the valid region.

    Args:
        height_map (np.ndarray): 2D array representing the height map.
        filename (str): The name of the output STL file.
        background_height (float): The height of the background in the STL model.
        maximum_x_y_size (float): Maximum size (in millimeters) for the x and y dimensions.
        alpha_mask (np.ndarray): Optional alpha mask (same shape as height_map).
            A pixel is “valid” only if its alpha is ≥ 128.
    """
    H, W = height_map.shape

    # Compute valid mask: every pixel is valid if no alpha mask is provided.
    valid_mask = (
        np.ones((H, W), dtype=bool) if alpha_mask is None else (alpha_mask >= 128)
    )

    # --- Vectorized Creation of Vertices ---
    # Create a meshgrid of coordinates. Note that the y coordinate is flipped so that row 0 is at the top.
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    x = j.astype(np.float32)
    y = (H - 1 - i).astype(np.float32)
    z = height_map.astype(np.float32) + background_height

    top_vertices = np.stack([x, y, z], axis=2)
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, :, 2] = 0

    # Scale vertices so the maximum x or y dimension equals maximum_x_y_size.
    original_max = max(W - 1, H - 1)
    scale = maximum_x_y_size / original_max
    top_vertices[:, :, :2] *= scale
    bottom_vertices[:, :, :2] *= scale

    # --- Top and Bottom Surfaces ---
    # Only use cells (quads) where all four corners are valid.
    quad_valid = (
        valid_mask[:-1, :-1]
        & valid_mask[:-1, 1:]
        & valid_mask[1:, 1:]
        & valid_mask[1:, :-1]
    )
    valid_i, valid_j = np.nonzero(quad_valid)
    num_quads = len(valid_i)

    # Define the four corners of each valid quad.
    v0 = top_vertices[valid_i, valid_j]
    v1 = top_vertices[valid_i, valid_j + 1]
    v2 = top_vertices[valid_i + 1, valid_j + 1]
    v3 = top_vertices[valid_i + 1, valid_j]

    # Top surface: using triangles (v2, v1, v0) and (v3, v2, v0)
    top_triangles = np.concatenate(
        [np.stack([v2, v1, v0], axis=1), np.stack([v3, v2, v0], axis=1)], axis=0
    )

    # Bottom face (using bottom vertices; note the reversed order so normals point downward)
    bv0 = bottom_vertices[valid_i, valid_j]
    bv1 = bottom_vertices[valid_i, valid_j + 1]
    bv2 = bottom_vertices[valid_i + 1, valid_j + 1]
    bv3 = bottom_vertices[valid_i + 1, valid_j]

    bottom_triangles = np.concatenate(
        [np.stack([bv0, bv1, bv2], axis=1), np.stack([bv0, bv2, bv3], axis=1)], axis=0
    )

    # --- Side Walls ---
    # Determine boundary edges from the grid of valid quads.
    # For each quad edge, if there is no neighboring valid quad sharing that edge, it is a boundary.
    side_triangles_list = []

    # Left edges: for quads in column 0 or when left neighbor is not valid.
    left_cond = np.zeros_like(quad_valid, dtype=bool)
    left_cond[:, 0] = quad_valid[:, 0]
    left_cond[:, 1:] = quad_valid[:, 1:] & (~quad_valid[:, :-1])
    li, lj = np.nonzero(left_cond)
    lv0 = top_vertices[li, lj]
    lv1 = top_vertices[li + 1, lj]
    lb0 = bottom_vertices[li, lj]
    lb1 = bottom_vertices[li + 1, lj]
    left_tris = np.concatenate(
        [np.stack([lv0, lv1, lb1], axis=1), np.stack([lv0, lb1, lb0], axis=1)], axis=0
    )
    side_triangles_list.append(left_tris)

    # Right edges: for quads in the last column or when right neighbor is not valid.
    right_cond = np.zeros_like(quad_valid, dtype=bool)
    right_cond[:, -1] = quad_valid[:, -1]
    right_cond[:, :-1] = quad_valid[:, :-1] & (~quad_valid[:, 1:])
    ri, rj = np.nonzero(right_cond)
    rv0 = top_vertices[ri, rj + 1]
    rv1 = top_vertices[ri + 1, rj + 1]
    rb0 = bottom_vertices[ri, rj + 1]
    rb1 = bottom_vertices[ri + 1, rj + 1]
    right_tris = np.concatenate(
        [np.stack([rv0, rv1, rb1], axis=1), np.stack([rv0, rb1, rb0], axis=1)], axis=0
    )
    side_triangles_list.append(right_tris)

    # Top edges: for quads in the first row or when the above neighbor is not valid.
    top_cond = np.zeros_like(quad_valid, dtype=bool)
    top_cond[0, :] = quad_valid[0, :]
    top_cond[1:, :] = quad_valid[1:, :] & (~quad_valid[:-1, :])
    ti, tj = np.nonzero(top_cond)
    tv0 = top_vertices[ti, tj]
    tv1 = top_vertices[ti, tj + 1]
    tb0 = bottom_vertices[ti, tj]
    tb1 = bottom_vertices[ti, tj + 1]
    top_wall_tris = np.concatenate(
        [np.stack([tv0, tv1, tb1], axis=1), np.stack([tv0, tb1, tb0], axis=1)], axis=0
    )
    side_triangles_list.append(top_wall_tris)

    # Bottom edges: for quads in the last row or when the below neighbor is not valid.
    bottom_cond = np.zeros_like(quad_valid, dtype=bool)
    bottom_cond[-1, :] = quad_valid[-1, :]
    bottom_cond[:-1, :] = quad_valid[:-1, :] & (~quad_valid[1:, :])
    bi, bj = np.nonzero(bottom_cond)
    bv0_edge = top_vertices[bi + 1, bj]
    bv1_edge = top_vertices[bi + 1, bj + 1]
    bb0 = bottom_vertices[bi + 1, bj]
    bb1 = bottom_vertices[bi + 1, bj + 1]
    bottom_wall_tris = np.concatenate(
        [
            np.stack([bv0_edge, bv1_edge, bb1], axis=1),
            np.stack([bv0_edge, bb1, bb0], axis=1),
        ],
        axis=0,
    )
    side_triangles_list.append(bottom_wall_tris)

    # Combine all side wall triangles.
    side_triangles = (
        np.concatenate(side_triangles_list, axis=0)
        if side_triangles_list
        else np.empty((0, 3, 3), dtype=np.float32)
    )

    # --- Combine All Triangles ---
    all_triangles = np.concatenate(
        [top_triangles, side_triangles, bottom_triangles], axis=0
    )

    # --- Compute Normals Vectorized ---
    v1_arr = all_triangles[:, 0, :]
    v2_arr = all_triangles[:, 1, :]
    v3_arr = all_triangles[:, 2, :]
    normals = np.cross(v2_arr - v1_arr, v3_arr - v1_arr)
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1  # Prevent division by zero
    normals /= norms[:, np.newaxis]

    num_triangles = all_triangles.shape[0]

    # --- Create a Structured Array for Binary STL ---
    stl_dtype = np.dtype(
        [
            ("normal", np.float32, (3,)),
            ("v1", np.float32, (3,)),
            ("v2", np.float32, (3,)),
            ("v3", np.float32, (3,)),
            ("attr", np.uint16),
        ]
    )
    stl_data = np.empty(num_triangles, dtype=stl_dtype)
    stl_data["normal"] = normals
    stl_data["v1"] = all_triangles[:, 0, :]
    stl_data["v2"] = all_triangles[:, 1, :]
    stl_data["v3"] = all_triangles[:, 2, :]
    stl_data["attr"] = 0

    # Write to an in-memory buffer
    buffer = io.BytesIO()
    header_str = "Binary STL generated from heightmap with alpha mask"
    header = header_str.encode("utf-8").ljust(80, b" ")
    buffer.write(header)
    buffer.write(struct.pack("<I", num_triangles))
    buffer.write(stl_data.tobytes())
    buffer.seek(0)

    # Load the mesh from the in-memory buffer using trimesh.
    mesh: Trimesh = trimesh.load(buffer, file_type="stl")
    mesh.merge_vertices()
    mesh.export(filename)


def generate_swap_instructions(
    discrete_global,
    discrete_height_image,
    h,
    background_layers,
    background_height,
    material_names,
):
    """
    Generate swap instructions based on discrete material assignments.

    Args:
        discrete_global (jnp.ndarray): Array of discrete global material assignments.
        discrete_height_image (jnp.ndarray): Array representing the discrete height image.
        h (float): Layer thickness.
        background_layers (int): Number of background layers.
        background_height (float): Height of the background in mm.
        material_names (list): List of material names.

    Returns:
        list: A list of strings containing the swap instructions.
    """
    L = int(np.max(np.array(discrete_height_image)))
    instructions = []
    if L == 0:
        instructions.append("No layers printed.")
        return instructions
    instructions.append("Start with your background color")
    for i in range(0, L):
        if i == 0 or int(discrete_global[i]) != int(discrete_global[i - 1]):
            ie = i + 1
            instructions.append(
                f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}"
            )
    instructions.append(
        "For the rest, use " + material_names[int(discrete_global[L - 1])]
    )
    return instructions
