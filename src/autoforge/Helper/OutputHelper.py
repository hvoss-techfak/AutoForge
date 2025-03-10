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

    # Compute valid mask: True if no alpha mask or alpha >= 128.
    if alpha_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)
    else:
        valid_mask = alpha_mask >= 128

    # --- Create vertices for the top surface ---
    # We compute a grid of vertices (even for masked-out regions) but later only use those that are valid.
    top_vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            top_vertices[i, j, 0] = j
            # Note: y coordinate is set so that row 0 is at the top.
            top_vertices[i, j, 1] = H - 1 - i
            top_vertices[i, j, 2] = height_map[i, j] + background_height

    # Create corresponding bottom vertices (same x,y but z = 0)
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, :, 2] = 0

    # --- Scale vertices so that the maximum x or y dimension equals maximum_x_y_size ---
    original_max = max(W - 1, H - 1)
    scale = maximum_x_y_size / original_max
    top_vertices[:, :, 0] *= scale
    top_vertices[:, :, 1] *= scale
    bottom_vertices[:, :, 0] *= scale
    bottom_vertices[:, :, 1] *= scale

    triangles = []  # List to collect all triangles (each as a tuple of three vertices)

    def add_triangle(v1, v2, v3):
        """Append a triangle (defined by 3 vertices) to the triangle list."""
        triangles.append((v1, v2, v3))

    # --- Top Surface ---
    # For the top surface, we iterate over each cell (quad) in the grid.
    # Only if all 4 corners of the quad are valid do we add two triangles.
    # We also store the grid indices (i,j) of the vertices used so we can later extract the boundary edges.
    top_triangles_indices = []  # Each element is a tuple of 3 (i,j) indices.
    for i in range(H - 1):
        for j in range(W - 1):
            if (
                valid_mask[i, j]
                and valid_mask[i, j + 1]
                and valid_mask[i + 1, j + 1]
                and valid_mask[i + 1, j]
            ):
                # Define the four corners of the quad.
                v0 = top_vertices[i, j]
                v1 = top_vertices[i, j + 1]
                v2 = top_vertices[i + 1, j + 1]
                v3 = top_vertices[i + 1, j]
                # For the top surface, use a reversed order so that the computed normal (via cross product) faces upward.
                add_triangle(v2, v1, v0)
                add_triangle(v3, v2, v0)
                # Save the corresponding grid indices (for later boundary detection)
                top_triangles_indices.append(((i, j), (i, j + 1), (i + 1, j + 1)))
                top_triangles_indices.append(((i + 1, j), (i + 1, j + 1), (i, j)))

    # --- Side Walls ---
    # Instead of simply using the rectangular border, we compute the boundary of the top surface.
    # We count each undirected edge from the top surface triangles and select those that appear only once.
    edge_count = {}

    def add_edge(idx1, idx2):
        # Store an undirected edge as a sorted tuple of the grid indices.
        key = tuple(sorted((idx1, idx2)))
        edge_count[key] = edge_count.get(key, 0) + 1

    for tri in top_triangles_indices:
        add_edge(tri[0], tri[1])
        add_edge(tri[1], tri[2])
        add_edge(tri[2], tri[0])

    # A boundary edge appears only once.
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Helper functions to get the coordinate for a given grid index.
    def get_top_vertex(idx):
        i, j = idx
        return top_vertices[i, j]

    def get_bottom_vertex(idx):
        i, j = idx
        return bottom_vertices[i, j]

    # For each boundary edge, create a vertical wall.
    for edge in boundary_edges:
        idx1, idx2 = edge
        v_top1 = get_top_vertex(idx1)
        v_top2 = get_top_vertex(idx2)
        v_bottom1 = get_bottom_vertex(idx1)
        v_bottom2 = get_bottom_vertex(idx2)
        # Create a quad (two triangles) connecting the top edge to the bottom.
        add_triangle(v_top1, v_top2, v_bottom2)
        add_triangle(v_top1, v_bottom2, v_bottom1)

    # --- Bottom Face ---
    # For the bottom face, we mimic the top-surface logic but using the bottom vertices.
    # The ordering is reversed (relative to the top face) so that the normals face downward.
    for i in range(H - 1):
        for j in range(W - 1):
            if (
                valid_mask[i, j]
                and valid_mask[i, j + 1]
                and valid_mask[i + 1, j + 1]
                and valid_mask[i + 1, j]
            ):
                v0 = bottom_vertices[i, j]
                v1 = bottom_vertices[i, j + 1]
                v2 = bottom_vertices[i + 1, j + 1]
                v3 = bottom_vertices[i + 1, j]
                # Order so that the normal will point downward.
                add_triangle(v0, v1, v2)
                add_triangle(v0, v2, v3)

    num_triangles = len(triangles)

    # --- Write Binary STL File ---
    with open(filename, "wb") as f:
        header_str = "Binary STL generated from heightmap with alpha mask"
        header = header_str.encode("utf-8")
        header = header.ljust(80, b" ")
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        for tri in triangles:
            v1, v2, v3 = tri
            # Compute the normal as the normalized cross product of (v2-v1) and (v3-v1).
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
            ie = i + 1
            instructions.append(
                f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}"
            )
    instructions.append(
        "For the rest, use " + material_names[int(discrete_global[L - 1])]
    )
    return instructions
