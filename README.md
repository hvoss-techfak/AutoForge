# AutoForge

AutoForge is a Python tool for generating 3D printed layered models from an input image. Using a learned optimization strategy with a Gumbel softmax formulation, AutoForge assigns materials per layer and produces both a discretized composite image and a 3D-printable STL file. It also generates swap instructions to guide the printer through material changes during a multi-material print. 

**TLDR:** It uses a picture to generate a 3D layer image that you can print with a 3d printer. Similar to [Hueforge](https://shop.thehueforge.com/), but without the manual work (and without the artistic control).


## You can now run Autoforge for free in your browser thanks to [Huggingface space support](https://huggingface.co/spaces/hvoss-techfak/Autoforge).

## Example
All examples use only the 13 BambuLab Basic filaments, currently available in Hueforge 0.8.9, the background color is set to black.
The pruning is set to a maximum of 8 color and 20 swaps, so each image uses at most 8 different colors and swaps the filament at most 20 times. 
<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <h3>Input Image</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/chameleon.jpg" width="200" />
  </div>
  <div style="text-align: center;">
    <h3>Autoforge Output</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/chameleon_discretized.png" width="200" />
  </div>
</div>

## Features

- **Image-to-Model Conversion**: Converts an input image into a layered model suitable for 3D printing.
- **Learned Optimization**: Optimizes per-pixel height and per-layer material assignments using PyTorch.
- **Learned Heightmap**: Optimizes the height of the layered model to create more detailed prints.
- **Gumbel Softmax Sampling**: Leverages the Gumbel softmax method to decide material assignments for each layer.
- **STL File Generation**: Exports an ASCII STL file based on the optimized height map.
- **Swap Instructions**: Generates clear swap instructions for changing materials during printing.
- **Live Visualization**: (Optional) Displays live composite images during the optimization process.
- **Hueforge export**: Outputs a project file that can be opened with hueforge.

## [Google Colab Notebook](https://colab.research.google.com/drive/13sWCss9-ew2xnoJl4c4SSpComSNi-1ih)

If you're new to Google Colab, follow these simple steps to run autoforge and get your output:

1. **Open the Notebook:**  
   Click [here](https://colab.research.google.com/drive/13sWCss9-ew2xnoJl4c4SSpComSNi-1ih) to open the notebook in Google Colab.

2. **Select GPU Runtime:**  
   In Colab, go to the menu and click **Runtime → Change runtime type**. Under **Hardware accelerator**, choose **T4 GPU** (or any other available GPU type).

3. **Run All Cells:**  
   Once the notebook loads, click **Runtime → Run all** to execute the cells in sequence. Follow the on-screen prompts to configure the materials CSV, upload your image, and run Autoforge.

4. **Review and Download:**  
   After the process completes, you'll see a preview of the output image. Finally, the notebook will zip the output folder for you to download.

## Manual Installation

To install AutoForge, simply install the current version from PyPI:
```bash
   pip install -U autoforge
```

If you have problems running the code on your gpu, please refer to the [Pytorch Homepage](https://pytorch.org/) for help. \
CUDA, ROCm, and MPS (Apple Metal) are supported, but you need to install the correct version of pytorch for your system.

## Usage

The script is run from the command line and accepts several arguments. Below is an example command:

> **Note:** You will need [Hueforge](https://shop.thehueforge.com/) installed to export your filament CSV.  
> To get your CSV file, simply go to the "Filaments" menu in Hueforge, click the export button, select your filaments, and export them as a CSV file.

```bash
autoforge --input_image path/to/input_image.jpg --csv_file path/to/materials.csv 
```

We also support json files. If you want to use your personal Hueforge library (found in %APPDATA%\HueForge\Filaments\personal_library.json) you can run the command with:
```bash
autoforge --input_image path/to/input_image.jpg --json_file %APPDATA%\HueForge\Filaments\personal_library.json
```


> If you want to limit the amount of colors the program can use, you can set these as command line arguments. \
> For Example: 8 colors and a maximum of 20 swaps:

```bash
autoforge --input_image path/to/input_image.jpg --csv_file path/to/materials.csv --pruning_max_colors 8 --pruning_max_swaps 20
```


### Command Line Arguments

- `--config` *(Optional)* Path to a configuration file with the settings.


- `--input_image` **(Required)** Path to the input image.
- `--csv_file` Path to the CSV file containing material data. The CSV should include columns for the brand, name, color (hex code), and TD values.
- `--json_file` Path to the json file containing material data. \
 **Note:** Either a csv or json file has to be given.

- `--output_folder` Folder where output files will be saved (default: `output`).
- `--iterations` Number of optimization iterations (default: 2000).
- `--learning_rate` Learning rate for optimization (default: 0.015).
- `--layer_height` Layer thickness in millimeters (default: 0.04).
- `--max_layers` Maximum number of layers (default: 75).  
  **Note:** This is about 3mm + the background height
- `--background_height` Height of the background in millimeters (default: 0.4).  
  **Note:** The background height must be divisible by the layer height.
- `--background_color` Background color in hexadecimal format (default: `#000000` aka Black).  
  **Note:** The solver currently assumes that you have a solid color in the background, which means a color with a TD value of 4 or less (if you have a background height of 0.4).
- `--init_tau` Initial tau value for Gumbel-Softmax (default: 1.0).
- `--final_tau` Final tau value for the Gumbel-Softmax formulation (default: 0.01).
- `--visualize` enable live visualization of the composite image during optimization (default: True).
- `--stl_output_size` Size of the longest dimension of the output STL file in millimeters (default: 150).
- `--nozzle_diameter` Diameter of the printer nozzle in millimeters (default: 0.4).  
  **Note:** Details smaller than half this value will be ignored.
- `--early_stopping` Number of steps without improvement before stopping (default: 1500).
- 

- `--perform_pruning`  Perform pruning after optimization (default: True).  
  **Note:** This is highly recommended even if you don't have a color/color swap limit, as it actually increases the quality of the output.
- `--fast_pruning`  Perform pruning in chunks. 10-15x speedup compared to accurate method (default: True).
- `--fast_pruning_percent` Size of fast pruning chunks in percent (default: 0.05) (5%).
- `--pruning_max_colors` Max number of colors allowed after pruning (default: 100).
- `--pruning_max_swaps` Max number of swaps allowed after pruning (default: 100).
- `--pruning_max_layer` Max number of layers allowed after pruning (default: 75).
- `--min_layers`  Minimum number of layers (default: 0). Used to limit height of pruning.
- `--random_seed` Random seed for reproducibility (default: 0 (disabled)).
- `--mps` Flag to use the Metal Performance Shaders (MPS) backend if available.


- `--tensorboard` Flag to enable TensorBoard logging.
- `--run_name` *(Optional)* Name of the run used for TensorBoard logging.



### Experimental Deph Anything V2 parameters
I got a request to add an initializing function that takes the original depth of the image into account when initializing. \
In theory this should give you a nice background/foreground separation, but in practice it can degrade the quality of the output. \
In addition you need the VRAM to run the depth model. \
I currently don't recommend it as the output is not as good as without it, but I will leave it in for now. \

- `--use_depth_anything`: Use a depth anything v2 model to initialize the height map (default: False).
  **Note:** This will give you a nice background/foreground separation, but will could degrade quality in some cases.
  **Note:** In addition you need the VRAM to run the depth model.
- `--depth_strength`: Weight for blending even spacing with the cluster’s average depth when using depth initialization. (default: 0.25)
- `--depth_threshold`: Threshold for splitting two distinct colors based on depth. (default: 0.05)
- `--min_cluster_value`: Minimum normalized value for the lowest cluster (to avoid pure black). (default: 0.1)
- `--w_depth`: Weight for depth difference in ordering. (default: 0.5)
- `--w_lum`: Weight for luminance difference in ordering. (default: 1.0)
- `--order_blend`: Blending factor between original luminance ordering (0) and depth-informed ordering (1). (default: 0.1)


## Outputs

After running, the following files will be created in your specified output folder:

- **Discrete Composite Image**: `final_model.png`
- **STL File**: `final_model.stl`
- **Hueforge Project File**: `project_file.hfp` 
- **Swap Instructions**: `swap_instructions.txt`

## Development

To have a "nightly" version of the repository or have live updating changes during development please do the following: 

```bash
git clone https://github.com/hvoss-techfak/AutoForge.git
cd AutoForge
conda create -n forge python=3.11
conda activate forge
pip install -e .
```

If the installed pytorch version has no cuda support execute the following:

```bash
conda activate forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Known Bugs

- The optimizer can sometimes get stuck in a local minimum. If this happens, try running the optimization again with different settings.

## License

AutoForge © 2025 by Hendric Voss is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The software is provided as-is and comes with no warranty or guarantee of support.


## Acknowledgements

First and foremost:
- [Hueforge](https://shop.thehueforge.com/) for providing the inspiration for this project.
Without it, this project would not have been possible.

AutoForge makes use of several open source libraries:

- [PyTorch](https://pytorch.org/)
- [Optax](https://github.com/deepmind/optax)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [TQDM](https://github.com/tqdm/tqdm)
- [ConfigArgParse](https://github.com/bw2/ConfigArgParse)

Example Images: \
<a href="https://www.vecteezy.com/free-photos/nature">Nature Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/ai-generated">Ai Generated Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/animal">Animal Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/people">People Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/psychedelic">Psychedelic Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/ocean">Ocean Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/psychic">Psychic Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/psychedelic">Psychedelic Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/pattern">Pattern Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/forest">Forest Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/stick-figure-kids">Stick Figure Kids Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/forest">Forest Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/nature">Nature Stock photos by Vecteezy</a> \
<a href="https://www.vecteezy.com/free-photos/ai-generated">Ai Generated Stock photos by Vecteezy</a>\
<a href="https://www.vecteezy.com/free-photos/animal">Animal Stock photos by Vecteezy</a> 


Happy printing!
