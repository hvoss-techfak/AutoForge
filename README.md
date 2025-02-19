**README.md**


# AutoForge

AutoForge is a Python tool for generating 3D printed layered models from an input image. Using a learned optimization strategy with a Gumbel softmax formulation, AutoForge assigns materials per layer and produces both a discretized composite image and a 3D-printable STL file. It also generates swap instructions to guide the printer through material changes during a multi-material print.

## Features

- **Image-to-Model Conversion**: Converts an input image into a layered model suitable for 3D printing.
- **Learned Optimization**: Optimizes per-pixel height and per-layer material assignments using JAX and Optax.
- **Gumbel Softmax Sampling**: Leverages the Gumbel softmax method to decide material assignments for each layer.
- **STL File Generation**: Exports an ASCII STL file based on the optimized height map.
- **Swap Instructions**: Generates clear swap instructions for changing materials during printing.
- **Live Visualization**: (Optional) Displays live composite images during the optimization process.

## Example
## Example
<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <h3>Input Image</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi.jpg" width="300" />
  </div>
  <div style="text-align: center;">
    <h3>Autoforge Output</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_discretized.png" width="300" />
  </div>
</div>
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

## Usage

The script is run from the command line and accepts several arguments. Below is an example command:
Please note that you will need [Hueforge](https://shop.thehueforge.com/) installed to export your filament csv. \
To get your csv file, simply go to the "Filaments" menu in Hueforge, click the export button, select your filaments and export your filaments as a csv file.
```bash
python auto_forge.py --input_image path/to/input_image.jpg \
                     --csv_file path/to/materials.csv \
                     --output_folder outputs \
                     --iterations 20000 \
                     --learning_rate 0.01 \
                     --target_max 512 \
                     --visualize
```

### Command Line Arguments

- `--input_image`: **(Required)** Path to the input image.
- `--csv_file`: **(Required)** Path to the CSV file containing material data. The CSV should include columns for the brand, name, color (hex code), and TD values.
- `--output_folder`: **(Required)** Folder where output files will be saved.
- `--iterations`: Number of optimization iterations (default: 20000).
- `--learning_rate`: Learning rate for the optimizer (default: 0.01).
- `--max_size`: Maximum dimension (width or height) for the resized target image (default: 512).
- `--decay`: Final tau value for Gumbel-Softmax loss (if unsure, leave as default).
- `--visualize`: Flag to enable live visualization of the composite image during optimization.
- `--config`: *(Optional)* Path to a configuration file with the above settings.

## Outputs

After running, the following files will be created in your specified output folder:

- **Discrete Composite Image**: `discrete_comp.png`
- **STL File**: `final_model.stl`
- **Swap Instructions**: `swap_instructions.txt`

## Requirements

See the [requirements.txt](requirements.txt) file for a complete list of dependencies.\
The optimizer is built using JAX, which requires a CUDA-compatible GPU for optimal performance.\
Without a dedicated gpu the process can take a very long time (up to 10x longer).\
If you have a gpu, you can install the gpu version of jax by running:
```bash
pip install -U "jax[cuda12]"
```

## License

AutoForge © 2025 by Hendric Voss is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements

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
