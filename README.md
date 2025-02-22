# AutoForge

AutoForge is a Python tool for generating 3D printed layered models from an input image. Using a learned optimization strategy with a Gumbel softmax formulation, AutoForge assigns materials per layer and produces both a discretized composite image and a 3D-printable STL file. It also generates swap instructions to guide the printer through material changes during a multi-material print. \

**TLDR:** It uses a picture to generate a 3D layer image that you can print with a 3d printer. Similar to [Hueforge](https://shop.thehueforge.com/), but without the manual work (and without the artistic control).




## Example
All examples use only the 13 BambuLab Basic filaments, currently available in Hueforge.
<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <h3>Input Image</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi.jpg" width="300" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature.jpg" width="300" />
  </div>
  <div style="text-align: center;">
    <h3>Autoforge Output (75 color layers (3.0mm) + 10 gray background layers (0.4mm)) </h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_discretized.jpg" width="300" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_discretized.jpg" width="300" />
  </div>
  <div style="text-align: center;">
    <h3>Imported Hueforge View</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_hueforge.png" width="300" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_hueforge.png" width="300" />
  </div>
</div>

If someone uses this program and 3d prints something with it, please let me know :) \
I would love to see what you made!


## Features

- **Image-to-Model Conversion**: Converts an input image into a layered model suitable for 3D printing.
- **Learned Optimization**: Optimizes per-pixel height and per-layer material assignments using JAX and Optax.
- **Gumbel Softmax Sampling**: Leverages the Gumbel softmax method to decide material assignments for each layer.
- **STL File Generation**: Exports an ASCII STL file based on the optimized height map.
- **Swap Instructions**: Generates clear swap instructions for changing materials during printing.
- **Live Visualization**: (Optional) Displays live composite images during the optimization process.
- **Hueforge export**: Outputs a project file that can be opened with hueforge.


## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/hvoss-techfak/AutoForge.git
   cd AutoForge
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

See the [requirements.txt](requirements.txt) file for a complete list of dependencies.  
The optimizer is built using JAX, which benefits from a CUDA-compatible GPU for optimal performance.  
Without a dedicated GPU the process can take significantly longer (up to 10-20x slower).  
If you have a GPU, you can install the GPU version of JAX by running:

```bash
pip install -U "jax[cuda12]"
```

Currently, jax gpu support is only available under linux. \
If you use Windows, please install WSL2 to use the GPU version of jax.

## Usage

The script is run from the command line and accepts several arguments. Below is an example command:

> **Note:** You will need [Hueforge](https://shop.thehueforge.com/) installed to export your filament CSV.  
> To get your CSV file, simply go to the "Filaments" menu in Hueforge, click the export button, select your filaments, and export them as a CSV file.

> If you want to limit the amount of colors the program can use, you can simply delete the colors you don't want from the CSV file.

```bash
python auto_forge.py --input_image path/to/input_image.jpg --csv_file path/to/materials.csv --output_folder outputs 
```

### Command Line Arguments

- `--config`: *(Optional)* Path to a configuration file with the settings.
- `--input_image`: **(Required)** Path to the input image.
- `--csv_file`: **(Required)** Path to the CSV file containing material data. The CSV should include columns for the brand, name, color (hex code), and TD values.
- `--output_folder`: **(Required)** Folder where output files will be saved.
- `--iterations`: Number of optimization iterations (default: 20000).
- `--learning_rate`: Learning rate for the optimizer (default: 5e-3).
- `--layer_height`: Layer thickness in millimeters (default: 0.04).
- `--max_layers`: Maximum number of layers (default: 75). 
  **Note:** This is about 3mm + the background height
- `--background_height`: Height of the background in millimeters (default: 0.4).  
  **Note:** The background height must be divisible by the layer height.
- `--background_color`: Background color in hexadecimal format (default: `#8e9089` aka Bambulab Grey).
- `--max_size`: Maximum dimension (width or height) for the resized target image (default: 512).
- `--decay`: Final tau value for the Gumbel-Softmax formulation (default: 0.01).
- `--loss`: Loss function to use. Choices are `mse`, `perceptual`, or `perceptual_l1` (default: `mse`).
- `--visualize`: Flag to enable live visualization of the composite image during optimization.

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

## Acknowledgements

First and foremost:
- [Hueforge](https://shop.thehueforge.com/) for providing the filament data and inspiration for this project.
Without it, this project would not have been possible.

AutoForge makes use of several open source libraries:

- [JAX](https://github.com/google/jax)
- [Optax](https://github.com/deepmind/optax)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [TQDM](https://github.com/tqdm/tqdm)
- [ConfigArgParse](https://github.com/bw2/ConfigArgParse)

Example Images:  
<a href="https://www.vecteezy.com/free-photos/anime-girl">Anime Girl Stock photos by Vecteezy</a>  
<a href="https://www.vecteezy.com/free-photos/nature">Nature Stock photos by Vecteezy</a>

Happy printing!