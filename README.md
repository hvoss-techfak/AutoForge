# AutoForge

AutoForge is a Python tool for generating 3D printed layered models from an input image. Using a learned optimization strategy with a Gumbel softmax formulation, AutoForge assigns materials per layer and produces both a discretized composite image and a 3D-printable STL file. It also generates swap instructions to guide the printer through material changes during a multi-material print. \

**TLDR:** It uses a picture to generate a 3D layer image that you can print with a 3d printer. Similar to [Hueforge](https://shop.thehueforge.com/), but without the manual work (and without the artistic control).


## Important Information

We recently switched from JAX to PyTorch, to allow for a more streamlined development process. \
This can have some unforeseen consequences, so please report any bugs you find. 

This release should give better gpu support for more users, especially on windows. \
If you have problems running the code on your gpu, please refer to the [Pytorch Homepage](https://pytorch.org/) for help. \
Both CUDA and ROCm are supported, but you need to install the correct version of pytorch for your system.

## Example
All examples use only the 13 BambuLab Basic filaments, currently available in Hueforge.
<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <h3>Input Image</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat.jpg" width="200" />
  </div>
  <div style="text-align: center;">
    <h3>Autoforge Output (75 color layers (3.0mm) + 10 gray background layers (0.4mm)) </h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_discretized.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_discretized.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat_discretized.jpg" width="200" />
  </div>
  <div style="text-align: center;">
    <h3>Imported Hueforge View</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_hueforge.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_hueforge.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat_hueforge.png" width="200" />
  </div>
</div>

If someone uses this program and 3d prints something with it, please let me know :) \
I would love to see what you made!


## Features

- **Image-to-Model Conversion**: Converts an input image into a layered model suitable for 3D printing.
- **Learned Optimization**: Optimizes per-pixel height and per-layer material assignments using PyTorch.
- **Gumbel Softmax Sampling**: Leverages the Gumbel softmax method to decide material assignments for each layer.
- **STL File Generation**: Exports an ASCII STL file based on the optimized height map.
- **Swap Instructions**: Generates clear swap instructions for changing materials during printing.
- **Live Visualization**: (Optional) Displays live composite images during the optimization process.
- **Hueforge export**: Outputs a project file that can be opened with hueforge.


## Installation

To install AutoForge, simply install the current version from PyPI:
```bash
   pip install autoforge
```

## Usage

The script is run from the command line and accepts several arguments. Below is an example command:

> **Note:** You will need [Hueforge](https://shop.thehueforge.com/) installed to export your filament CSV.  
> To get your CSV file, simply go to the "Filaments" menu in Hueforge, click the export button, select your filaments, and export them as a CSV file.

> If you want to limit the amount of colors the program can use, you can simply delete the colors you don't want from the CSV file.

```bash
autoforge --input_image path/to/input_image.jpg --csv_file path/to/materials.csv --output_folder outputs 
```

### Command Line Arguments

- `--config`: *(Optional)* Path to a configuration file with the settings.
- `--input_image`: **(Required)** Path to the input image.
- `--csv_file`: **(Required)** Path to the CSV file containing material data. The CSV should include columns for the brand, name, color (hex code), and TD values.
- `--output_folder`: Folder where output files will be saved. (default: `./outputs`).
- `--iterations`: Number of optimization iterations (default: 5000).
- `--learning_rate`: Learning rate for the optimizer (default: 1e-2).
- `--layer_height`: Layer thickness in millimeters (default: 0.04).
- `--max_layers`: Maximum number of layers (default: 75). \
  **Note:** This is about 3mm + the background height
- `--min_layers`: Minimum number of layers (default: 0). Used to limit height of pruning 
- `--background_height`: Height of the background in millimeters (default: 0.4).  \
  **Note:** The background height must be divisible by the layer height.
- `--background_color`: Background color in hexadecimal format (default: `#000000` aka Black). \
  **Note:** The solver currently assumes that you have a solid color in the background, which means a color with a TD value of 4 or less (if you have a background height of 0.4)
- `--output_size`: Maximum dimension for target image (default: 1024).
- `--solver_size`: Maximum dimension for solver (fast) image (default: 128). \
  **Note:** We solve on a smaller size as this is many times faster, but also a bit less accurate. Increase if you need more accuracy.
- `--init_tau`: Initial tau value for Gumbel-Softmax (default: 1.0).
- `--final_tau`: Final tau value for the Gumbel-Softmax formulation (default: 0.01).
- `--stl_output_size`: Size of the output stl file in mm (default: 200) (20cm).
- `--visualize`: Flag to enable live visualization of the composite image during optimization.
- `--tensorboard`: Enable TensorBoard logging
- `--run-name`: Name of the run used for TensorBoard logging (optional).
- `--perform_pruning`: Perform pruning after optimization (default: True). \
  **Note:** This is highly recommended even if you don't have a color/color swap limit, as it actually increases the quality of the output.
- `--pruning_max_colors`: Max number of colors allowed after pruning (default: 100).
- `--pruning_max_swaps`: Max number of swaps allowed after pruning (default: 100).
- `--random_seed`: Random seed for reproducibility (default: 0 (disabled) ).

### Experimental Deph Anything V2 parameters
I got a request to add a initializing function that takes the original deph of the image into account when initializing. \
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

- **Discrete Composite Image**: `discrete_comp.png`
- **STL File**: `final_model.stl`
- **Swap Instructions**: `swap_instructions.txt`

Just a heads-up, this program is mainly concerned with realistic output and will give you VERY long swap instructions.
Expect to switch your filament every 1-2 layers!

For more artistic control or to reduce the number of swaps, consider buying [Hueforge](https://shop.thehueforge.com/).

## Known Bugs

- There is a color discrepancy between our output and hueforge. If anyone has an idea what the problem is, please don't hesitate to submit a pull request :) \
Although I would love to be it fully color compatible with hueforge, I don't think this will happen without a lot of work or the hueforge source code.
- The optimizer can sometimes get stuck in a local minimum. If this happens, try running the optimization again with different settings.
- Hueforge can't open STL files under linux. This is a known bug in hueforge.
- The first version used a learnable height map which was nice but resulted in a lot of problems. Right now we compute the height map from the luminance values and slightly adjust the result. If somebody has an idea to make this better, please don't hesitate to submit a pull request :)

## License

AutoForge © 2025 by Hendric Voss is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The software is provided as-is and comes with no warranty or guarantee of support.


## Acknowledgements

First and foremost:
- [Hueforge](https://shop.thehueforge.com/) for providing the filament data and inspiration for this project.
Without it, this project would not have been possible.

AutoForge makes use of several open source libraries:

- [PyTorch](https://pytorch.org/)
- [Optax](https://github.com/deepmind/optax)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [TQDM](https://github.com/tqdm/tqdm)
- [ConfigArgParse](https://github.com/bw2/ConfigArgParse)

Example Images:  
<a href="https://www.vecteezy.com/free-photos/anime-girl">Anime Girl Stock photos by Vecteezy</a>  
<a href="https://www.vecteezy.com/free-photos/nature">Nature Stock photos by Vecteezy</a>
<a href="https://www.vecteezy.com/free-photos/ai-generated">Ai Generated Stock photos by Vecteezy</a>

Happy printing!