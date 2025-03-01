import json
import os
import struct
import uuid
from itertools import permutations

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from torchvision.models import VGG16_Weights
from tqdm import tqdm


def extract_colors_from_swatches(swatch_data):

    # we keep only data with transmission distance
    swatch_data = [swatch for swatch in swatch_data if swatch["td"]]

    #For now we load it and convert it in the same way as the hueforge csv files
    out = {}
    for swatch in swatch_data:
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        color = swatch["hex_color"]
        td = swatch["td"]
        out[(brand, name)] = (color, td)

    #convert to the same format as the hueforge csv files
    material_names = [brand + " - " + name for (brand, name) in out.keys()]
    material_colors = np.array([hex_to_rgb("#"+color) for color, _ in out.values()], dtype=np.float64)
    material_TDs = np.array([td for _, td in out.values()], dtype=np.float64)
    colors_list = [color for color, _ in out.values()]

    return material_colors, material_TDs, material_names, colors_list

def swatch_data_to_table(swatch_data):
    """
    Converts swatch JSON data into a table (list of dicts) with columns:
    "Brand", "Name", "Transmission Distance", "Hex Color".
    """
    table = []
    for swatch in swatch_data:
        if not swatch["td"]:
            continue
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        hex_color = swatch["hex_color"]
        td = swatch["td"]
        table.append({
            "Brand": brand,
            "Name": name,
            "Transmission Distance": td,
            "Hex Color": f"#{hex_color}"
        })
    return table


def resize_image(img, max_size):
    h_img, w_img, _ = img.shape
    if w_img >= h_img:
        new_w = max_size
        new_h = int(max_size * h_img / w_img)
    else:
        new_h = max_size
        new_w = int(max_size * w_img / h_img)
    img_out = cv2.resize(img, (new_w, new_h))
    return img_out


def generate_stl(height_map, filename, background_height, scale=1.0):
    """
    Generate a binary STL file from a height map.

    Args:
        height_map (np.ndarray): 2D array representing the height map.
        filename (str): The name of the output STL file.
        background_height (float): The height of the background in the STL model.
        scale (float, optional): Scale factor for the x and y dimensions. Defaults to 1.0.
    """
    H, W = height_map.shape
    vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # Original coordinates: x = j*scale, y = (H - 1 - i), z = height + background
            vertices[i, j, 0] = j * scale
            vertices[i, j, 1] = (H - 1 - i)  # (Consider applying scale if needed)
            vertices[i, j, 2] = height_map[i, j] + background_height

    triangles = []

    def add_triangle(v1, v2, v3):
        """
        Add a triangle to the list of triangles.

        Args:
            v1 (np.ndarray): First vertex of the triangle.
            v2 (np.ndarray): Second vertex of the triangle.
            v3 (np.ndarray): Third vertex of the triangle.
        """
        triangles.append((v1, v2, v3))

    for i in range(H - 1):
        for j in range(W - 1):
            v0 = vertices[i, j]
            v1 = vertices[i, j + 1]
            v2 = vertices[i + 1, j + 1]
            v3 = vertices[i + 1, j]
            # Reversed order so normals face upward
            add_triangle(v2, v1, v0)
            add_triangle(v3, v2, v0)

    for j in range(W - 1):
        v0 = vertices[0, j]
        v1 = vertices[0, j + 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)
    for j in range(W - 1):
        v0 = vertices[H - 1, j]
        v1 = vertices[H - 1, j + 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, 0]
        v1 = vertices[i + 1, 0]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, W - 1]
        v1 = vertices[i + 1, W - 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)

    v0 = np.array([0, 0, 0], dtype=np.float32)
    v1 = np.array([(W - 1) * scale, 0, 0], dtype=np.float32)
    v2 = np.array([(W - 1) * scale, (H - 1) * scale, 0], dtype=np.float32)
    v3 = np.array([0, (H - 1) * scale, 0], dtype=np.float32)
    add_triangle(v2, v1, v0)
    add_triangle(v3, v2, v0)

    num_triangles = len(triangles)

    # Write the binary STL file.
    with open(filename, 'wb') as f:
        header_str = "Binary STL generated from heightmap"
        header = header_str.encode('utf-8')
        header = header.ljust(80, b' ')
        f.write(header)
        f.write(struct.pack('<I', num_triangles))
        for tri in triangles:
            v1, v2, v3 = tri
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = np.array([0, 0, 0], dtype=np.float32)
            else:
                normal = normal / norm
            f.write(struct.pack('<12fH',
                                  normal[0], normal[1], normal[2],
                                  v1[0], v1[1], v1[2],
                                  v2[0], v2[1], v2[2],
                                  v3[0], v3[1], v3[2],
                                  0))



def generate_swap_instructions(discrete_global, discrete_height_image, h, background_layers, background_height, material_names):
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
            instructions.append(f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}")
    instructions.append("For the rest, use " + material_names[int(discrete_global[L - 1])])
    return instructions


def initialize_pixel_height_logits(target):
    """
    Initialize pixel height logits based on the luminance of the target image.

    Assumes target is a jnp.array of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.

    Args:
        target (jnp.ndarray): The target image array with shape (H, W, 3).

    Returns:
        jnp.ndarray: The initialized pixel height logits.
    """
    # Compute normalized luminance in [0,1]
    normalized_lum = (0.299 * target[..., 0] +
                      0.587 * target[..., 1] +
                      0.114 * target[..., 2]) / 255.0
    # To avoid log(0) issues, add a small epsilon.
    eps = 1e-6
    # Convert normalized luminance to logits using the inverse sigmoid (logit) function.
    # This ensures that jax.nn.sigmoid(pixel_height_logits) approximates normalized_lum.
    pixel_height_logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))
    return pixel_height_logits


def load_materials_data(csv_filename):
    """
    Load the full material data from the CSV file.

    Args:
        csv_filename (str): Path to the CSV file containing material data.

    Returns:
        list: A list of dictionaries (one per material) with keys such as
              "Brand", "Type", "Color", "Name", "TD", "Owned", and "Uuid".
    """
    df = pd.read_csv(csv_filename)
    # Use a consistent key naming. For example, convert 'TD' to 'Transmissivity' and 'Uuid' to 'uuid'
    records = df.to_dict(orient="records")
    return records


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
            slider = (i + background_layers)-1
            slider_values.append(slider)
            filament_indices.append(prev)
        prev = current
    # Add the last material index
    filament_indices.append(prev)
    slider = i + background_layers
    slider_values.append(slider)

    return filament_indices, slider_values


def generate_project_file(project_filename, args, disc_global, disc_height_image,
                          width_mm, height_mm, stl_filename, csv_filename):
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
    filament_indices, slider_values = extract_filament_swaps(disc_global, disc_height_image, background_layers)

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
            "Transmissivity": float(mat[" TD"]) if not float(mat[" TD"]).is_integer() else int(mat[" TD"]),
            "Type": mat[" Type"],
            "uuid": mat[" Uuid"]
        }
        filament_set.append(filament_entry)

    # add black as the first filament with background height as the first slider value
    filament_set.insert(0, {
            "Brand": "Autoforge",
            "Color": args.background_color,
            "Name": "Background",
            "Owned": False,
            "Transmissivity": 0.1,
            "Type": "PLA",
            "uuid": str(uuid.uuid4())
    })
    # add black to slider value
    slider_values.insert(0, (args.background_height//args.layer_height)-1)

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
        "width_in_mm": width_mm
    }

    # Write out the project file as JSON
    with open(project_filename, "w") as f:
        json.dump(project_data, f, indent=4)

def init_height_map(target,max_layers,h,eps = 1e-6,random_seed=None):
    """
        Initialize pixel height logits based on the luminance of the target image.

        Assumes target is a jnp.array of shape (H, W, 3) in the range [0, 255].
        Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.

        Args:
            target (jnp.ndarray): The target image array with shape (H, W, 3).

        Returns:
            jnp.ndarray: The initialized pixel height logits.
        """

    target_np = np.asarray(target).reshape(-1, 3)

    kmeans = KMeans(n_clusters=max_layers,random_state=random_seed).fit(target_np)
    labels = kmeans.labels_
    labels = labels.reshape(target.shape[0], target.shape[1])
    centroids = kmeans.cluster_centers_

    def luminance(col):
        return 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]

    # --- Step 2: Second clustering of centroids into bands ---
    num_bands = 9
    band_kmeans = KMeans(n_clusters=num_bands,random_state=random_seed).fit(centroids)
    band_labels = band_kmeans.labels_

    # Group centroids by band and sort within each band by luminance
    bands = []  # each entry will be (band_avg_luminance, sorted_indices_in_this_band)
    for b in range(num_bands):
        indices = np.where(band_labels == b)[0]
        if len(indices) == 0:
            continue
        lum_vals = np.array([luminance(centroids[i]) for i in indices])
        sorted_indices = indices[np.argsort(lum_vals)]
        band_avg = np.mean(lum_vals)
        bands.append((band_avg, sorted_indices))

    # --- Step 3: Compute a representative color for each band in Lab space ---
    # (Using the average of the centroids in that band)
    band_reps = []  # will hold Lab colors
    for _, indices in bands:
        band_avg_rgb = np.mean(centroids[indices], axis=0)
        # Normalize if needed (assumes image pixel values are 0-255)
        band_avg_rgb_norm = band_avg_rgb / 255.0 if band_avg_rgb.max() > 1 else band_avg_rgb
        # Convert to Lab (expects image in [0,1])
        lab = rgb2lab(np.array([[band_avg_rgb_norm]]))[0, 0, :]
        band_reps.append(lab)

    # --- Step 4: Identify darkest and brightest bands based on L channel ---
    L_values = [lab[0] for lab in band_reps]
    start_band = np.argmin(L_values)  # darkest band index
    end_band = np.argmax(L_values)  # brightest band index

    # --- Step 5: Find the best ordering for the middle bands ---
    # We want to order the bands so that the total perceptual difference (Euclidean distance in Lab)
    # between consecutive bands is minimized, while forcing the darkest band first and brightest band last.
    all_indices = list(range(len(bands)))
    middle_indices = [i for i in all_indices if i not in (start_band, end_band)]

    min_total_distance = np.inf
    best_order = None
    total = len(middle_indices)*len(middle_indices)
    # Try all permutations of the middle bands
    ie = 0
    tbar = tqdm(permutations(middle_indices),total=total,desc="Finding best ordering for color bands:")
    for perm in tbar:
        candidate = [start_band] + list(perm) + [end_band]
        total_distance = 0
        for i in range(len(candidate) - 1):
            total_distance += np.linalg.norm(band_reps[candidate[i]] - band_reps[candidate[i + 1]])
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_order = candidate
            tbar.set_description(f"Finding best ordering for color bands: Total distance = {min_total_distance:.2f}")
        ie += 1
        if ie > 500000:
            break

    new_order = []
    for band_idx in best_order:
        # Each band tuple is (band_avg, sorted_indices)
        new_order.extend(bands[band_idx][1].tolist())

    # Remap each pixel's label so that it refers to its new palette index
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
    new_labels = np.vectorize(lambda x: mapping[x])(labels)

    new_labels = new_labels.astype(np.float32) / new_labels.max()

    normalized_lum = np.array(new_labels, dtype=np.float64)
    #convert out to inverse sigmoid logit function
    pixel_height_logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))

    H, W, _ = target.shape
    return pixel_height_logits

def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.

    Args:
        hex_str (str): The hex color string (e.g., '#RRGGBB').

    Returns:
        list: A list of three floats representing the RGB values normalized to [0, 1].
    """
    hex_str = hex_str.lstrip('#')
    return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def load_materials(csv_filename):
    """
    Load material data from a CSV file.

    Args:
        csv_filename (str): Path to the hueforge CSV file containing material data.

    Returns:
        tuple: A tuple containing:
            - material_colors (jnp.ndarray): Array of material colors in float64.
            - material_TDs (jnp.ndarray): Array of material transmission/opacity parameters in float64.
            - material_names (list): List of material names.
            - colors_list (list): List of color hex strings.
    """
    df = pd.read_csv(csv_filename)
    material_names = [brand + " - " + name for brand, name in zip(df["Brand"].tolist(), df[" Name"].tolist())]
    material_TDs = (df[' TD'].astype(float)).to_numpy()
    colors_list = df[' Color'].tolist()
    # Use float64 for material colors.
    material_colors = np.array([hex_to_rgb(color) for color in colors_list], dtype=np.float64)
    material_TDs = np.array(material_TDs, dtype=np.float64)
    return material_colors, material_TDs, material_names, colors_list

def count_distinct_colors(dg: torch.Tensor) -> int:
    """
    Count how many distinct color/material IDs appear in dg.
    """
    unique_mats = torch.unique(dg)
    return len(unique_mats)

def count_swaps(dg: torch.Tensor) -> int:
    """
    Count how many color changes (swaps) occur between adjacent layers.
    """
    # A 'swap' is whenever dg[i] != dg[i+1].
    return int((dg[:-1] != dg[1:]).sum().item())

def merge_color(dg: torch.Tensor, c_from: int, c_to: int) -> torch.Tensor:
    """
    Return a copy of dg where every layer with material c_from is replaced by c_to.
    """
    dg_new = dg.clone()
    dg_new[dg_new == c_from] = c_to
    return dg_new

def find_color_bands(dg: torch.Tensor):
    """
    Return a list of (start_idx, end_idx, color_id) for each contiguous band
    in 'dg'. Example: if dg = [0,0,1,1,1,2,2], we get:
       [(0,1,0), (2,4,1), (5,6,2)]
    """
    bands = []
    dg_cpu = dg.detach().cpu().numpy()
    start_idx = 0
    current_color = dg_cpu[0]
    n = len(dg_cpu)

    for i in range(1, n):
        if dg_cpu[i] != current_color:
            # finish previous band
            bands.append((start_idx, i - 1, current_color))
            # start new band
            start_idx = i
            current_color = dg_cpu[i]
    # finish last band
    bands.append((start_idx, n - 1, current_color))

    return bands


def merge_bands(dg: torch.Tensor,
                band_a: (int, int, int),
                band_b: (int, int, int),
                direction: str):
    """
    Merge band_a and band_b.  If direction == "forward", we unify the color of band_b
    to be band_a's color. If direction == "backward", unify band_a's color to band_b.

    band_a, band_b = (start_idx, end_idx, color_id)

    We assume band_a and band_b are *adjacent* in the layering order, i.e. band_a.end+1 == band_b.start
    or vice versa.
    """
    dg_new = dg.clone()
    c_a = band_a[2]
    c_b = band_b[2]

    if direction == "forward":
        # unify band_b to c_a
        dg_new[(band_b[0]):(band_b[1] + 1)] = c_a
    else:
        # unify band_a to c_b
        dg_new[(band_a[0]):(band_a[1] + 1)] = c_b

    return dg_new



class MultiLayerVGGPerceptualLoss(nn.Module):
    def __init__(self, layers: list = None, weights: list = None):
        """
        Uses a pretrained VGG16 model to extract features from multiple layers.
        By default, it uses layers [3, 8, 15, 22] (approximately conv1_2, conv2_2, conv3_3, conv4_3).
        """
        super(MultiLayerVGGPerceptualLoss, self).__init__()
        # Choose layers from VGG16.features
        if layers is None:
            layers = [8,]
        self.layers = layers

        # Load pretrained VGG16 and freeze parameters.
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        for param in vgg.parameters():
            param.requires_grad = False

        # We want to run the network up to the highest required layer.
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)]).eval()

        # Weights for each selected layer loss; default: equal weighting.
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        self.weights = weights

        # Register ImageNet normalization constants as buffers.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x and y are expected to be of shape (N, 3, H, W) with values in [0, 255].
        They are normalized to ImageNet stats and then passed through VGG16.
        The loss is computed as a weighted sum of MSE losses on the selected layers.
        """
        # Normalize images
        x = (x / 255.0 - self.mean) / self.std
        y = (y / 255.0 - self.mean) / self.std

        loss = 0.0
        out = x
        # Loop through VGG layers and compute losses at the selected layers.
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i in self.layers:
                # Extract corresponding feature for y by running y through the same layers.
                with torch.no_grad():
                    out_y = y
                    for j in range(i + 1):
                        out_y = self.vgg[j](out_y)
                loss += self.weights[self.layers.index(i)] * F.mse_loss(out, out_y)
        return loss

