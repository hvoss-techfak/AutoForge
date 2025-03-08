import json
import os
import struct
import uuid

import numpy as np

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
            slider = (i + background_layers) - 1
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
    material_data = load_materials_data(csv_filename)

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
            "Color": mat[" Color"],
            "Name": mat[" Name"],
            # Convert Owned to a boolean (in case it is read as a string)
            "Owned": str(mat[" Owned"]).strip().lower() == "true",
            "Transmissivity": float(mat[" TD"])
            if not float(mat[" TD"]).is_integer()
            else int(mat[" TD"]),
            "Type": mat[" Type"],
            "uuid": mat[" Uuid"],
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
    slider_values.insert(0, (args.background_height // args.layer_height) - 1)

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


def generate_stl(height_map, filename, background_height, maximum_x_y_size):
    """
    Generate a binary STL file from a height map with shared boundary vertices
    to fix non-manifold edge warnings, scaling the x and y dimensions to a maximum
    size specified in millimeters.

    Args:
        height_map (np.ndarray): 2D array representing the height map.
        filename (str): The name of the output STL file.
        background_height (float): The height of the background in the STL model.
        maximum_x_y_size (float): Maximum size (in millimeters) for the x and y dimensions.
    """
    H, W = height_map.shape

    # Create the top vertices grid.
    # Note: y is computed as H-1-i.
    top_vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            top_vertices[i, j, 0] = j
            top_vertices[i, j, 1] = H - 1 - i  # y coordinate: top row has highest value
            top_vertices[i, j, 2] = height_map[i, j] + background_height

    # Create the bottom vertices for the boundary by copying x and y coordinates,
    # but setting z to 0.
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, :, 2] = 0

    # --- Scale vertices so that the maximum x or y dimension equals maximum_x_y_size ---
    # The current extents in x and y are 0 to (W - 1) and 0 to (H - 1), respectively.
    # We use the maximum of these as our original size.
    original_max = max(W - 1, H - 1)
    scale = maximum_x_y_size / original_max
    top_vertices[:, :, 0] *= scale
    top_vertices[:, :, 1] *= scale
    bottom_vertices[:, :, 0] *= scale
    bottom_vertices[:, :, 1] *= scale

    triangles = []

    def add_triangle(v1, v2, v3):
        """Add a triangle defined by vertices v1, v2, and v3."""
        triangles.append((v1, v2, v3))

    # --- Top Surface ---
    for i in range(H - 1):
        for j in range(W - 1):
            v0 = top_vertices[i, j]
            v1 = top_vertices[i, j + 1]
            v2 = top_vertices[i + 1, j + 1]
            v3 = top_vertices[i + 1, j]
            # Use reversed order so that normals face upward.
            add_triangle(v2, v1, v0)
            add_triangle(v3, v2, v0)

    # --- Side Walls ---
    # Top edge (i = 0)
    for j in range(W - 1):
        vt0 = top_vertices[0, j]
        vt1 = top_vertices[0, j + 1]
        vb0 = bottom_vertices[0, j]
        vb1 = bottom_vertices[0, j + 1]
        add_triangle(vt0, vt1, vb1)
        add_triangle(vt0, vb1, vb0)

    # Bottom edge (i = H - 1)
    for j in range(W - 1):
        vt0 = top_vertices[H - 1, j]
        vt1 = top_vertices[H - 1, j + 1]
        vb0 = bottom_vertices[H - 1, j]
        vb1 = bottom_vertices[H - 1, j + 1]
        add_triangle(vt1, vt0, vb0)
        add_triangle(vt1, vb0, vb1)

    # Left edge (j = 0)
    for i in range(H - 1):
        vt0 = top_vertices[i, 0]
        vt1 = top_vertices[i + 1, 0]
        vb0 = bottom_vertices[i, 0]
        vb1 = bottom_vertices[i + 1, 0]
        add_triangle(vt1, vt0, vb0)
        add_triangle(vt1, vb0, vb1)

    # Right edge (j = W - 1)
    for i in range(H - 1):
        vt0 = top_vertices[i, W - 1]
        vt1 = top_vertices[i + 1, W - 1]
        vb0 = bottom_vertices[i, W - 1]
        vb1 = bottom_vertices[i + 1, W - 1]
        add_triangle(vt0, vt1, vb1)
        add_triangle(vt0, vb1, vb0)

    # --- Bottom Face ---
    # Use the four corner vertices from the bottom_vertices grid.
    b_top_left = bottom_vertices[0, 0]
    b_top_right = bottom_vertices[0, W - 1]
    b_bottom_right = bottom_vertices[H - 1, W - 1]
    b_bottom_left = bottom_vertices[H - 1, 0]
    # Triangulate the bottom face (ensure the normal faces downward).
    add_triangle(b_bottom_right, b_top_right, b_top_left)
    add_triangle(b_bottom_left, b_bottom_right, b_top_left)

    num_triangles = len(triangles)

    # --- Write Binary STL ---
    with open(filename, "wb") as f:
        header_str = "Binary STL generated from heightmap"
        header = header_str.encode("utf-8")
        header = header.ljust(80, b" ")
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        for tri in triangles:
            v1, v2, v3 = tri
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = np.array([0, 0, 0], dtype=np.float32)
            else:
                normal = normal / norm
            f.write(
                struct.pack(
                    "<12fH",
                    normal[0],
                    normal[1],
                    normal[2],
                    v1[0],
                    v1[1],
                    v1[2],
                    v2[0],
                    v2[1],
                    v2[2],
                    v3[0],
                    v3[1],
                    v3[2],
                    0,
                )
            )


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
            ie = i
            instructions.append(
                f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}"
            )
    instructions.append(
        "For the rest, use " + material_names[int(discrete_global[L - 1])]
    )
    return instructions
