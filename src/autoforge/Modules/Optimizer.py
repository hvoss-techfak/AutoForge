import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from autoforge.Helper.OptimizerHelper import composite_image, discretize_solution
from autoforge.Helper.PruningHelper import prune_colors_and_swaps, find_color_bands
from autoforge.Loss.LossFunctions import loss_fn, compute_loss


class FilamentOptimizer:
    def __init__(
        self,
        args: argparse.Namespace,
        target: torch.Tensor,
        pixel_height_logits_init: np.ndarray,
        material_colors: torch.Tensor,
        material_TDs: torch.Tensor,
        background: torch.Tensor,
        device: torch.device,
        perception_loss_module: torch.nn.Module,
    ):
        """
        Initialize an optimizer instance.

        Args:
            args (argparse.Namespace): Command-line arguments.
            target (torch.Tensor): Target image tensor.
            pixel_height_logits_init (np.ndarray): Initial pixel height logits.
            material_colors (torch.Tensor): Tensor of material colors.
            material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
            background (torch.Tensor): Background color tensor.
            device (torch.device): Device to run the optimization on.
            perception_loss_module (torch.nn.Module): Module to compute perceptual loss.
        """
        self.args = args
        self.target = target  # smaller (solver) resolution, shape [H,W,3], float32
        self.H, self.W = target.shape[:2]
        self.pixel_height_logits = torch.tensor(
            pixel_height_logits_init, dtype=torch.float32, device=device
        )
        self.pixel_height_logits.requires_grad_(True)

        # Basic hyper-params
        self.material_colors = material_colors
        self.material_TDs = material_TDs
        self.background = background
        self.max_layers = args.max_layers
        self.h = args.layer_height
        self.learning_rate = args.learning_rate
        self.final_tau = args.final_tau
        self.init_tau = args.init_tau
        self.device = device
        self.best_swaps = 0
        self.perception_loss_module = perception_loss_module
        self.visualize_flag = args.visualize

        # Initialize TensorBoard writer
        if args.tensorboard:
            if args.run_name:
                self.writer = SummaryWriter(log_dir=f'runs/{args.run_name}')
            else:
                self.writer = SummaryWriter()
        else:
            self.writer = None

        # We have an initial guess for 'global_logits'
        num_materials = material_colors.shape[0]
        global_logits_init = (
            torch.ones(
                (self.max_layers, num_materials), dtype=torch.float32, device=device
            )
            * -1.0
        )
        for i in range(self.max_layers):
            global_logits_init[i, i % num_materials] = 1.0

        global_logits_init += torch.rand_like(global_logits_init) * 0.2 - 0.1
        global_logits_init.requires_grad_(True)

        self.loss = None

        self.params = {
            "pixel_height_logits": self.pixel_height_logits,
            "global_logits": global_logits_init,
        }

        # Tau schedule
        self.num_steps_done = 0
        self.warmup_steps = args.iterations // 4
        self.decay_rate = -math.log(self.final_tau) / (
            args.iterations - self.warmup_steps
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            [self.params["pixel_height_logits"], self.params["global_logits"]],
            lr=self.learning_rate,
        )

        # Setup best discrete solution tracking
        self.best_discrete_loss = float("inf")
        self.best_params = None
        self.best_tau = None
        self.best_seed = None

        # If you want a figure for real-time visualization:
        if self.visualize_flag:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 4, figsize=(14, 6))
            self.target_im_ax = self.ax[0].imshow(
                np.array(self.target.cpu(), dtype=np.uint8)
            )
            self.ax[0].set_title("Target Image")
            self.current_comp_ax = self.ax[1].imshow(
                np.zeros((self.H, self.W, 3), dtype=np.uint8)
            )
            self.ax[1].set_title("Current Composite")

            self.depth_map_ax = self.ax[2].imshow(
                np.zeros((self.H, self.W), dtype=np.uint8), cmap="viridis"
            )
            self.ax[2].set_title("Current Depth Map")

            self.best_comp_ax = self.ax[3].imshow(
                np.zeros((self.H, self.W, 3), dtype=np.uint8)
            )
            self.ax[3].set_title("Best Discrete Composite")
            plt.pause(0.1)

    def _get_tau(self):
        """
        Compute tau for height & global given how many steps we've done.

        Returns:
            Tuple[float, float]: Tau values for height and global.
        """
        i = self.num_steps_done
        tau_init = self.init_tau
        if i < self.warmup_steps:
            return tau_init, tau_init
        else:
            # simple exponential decay
            t = max(
                self.final_tau,
                tau_init * math.exp(-self.decay_rate * (i - self.warmup_steps)),
            )
            return t, t

    def step(self, record_best: bool = False):
        """
        Perform exactly one gradient-descent update step.

        Args:
            record_best (bool, optional): Whether to record the best discrete solution. Defaults to False.

        Returns:
            float: The loss value of the current step.
        """
        self.optimizer.zero_grad()

        tau_height, tau_global = self._get_tau()

        loss = loss_fn(
            self.params,
            target=self.target,
            tau_height=tau_height,
            tau_global=tau_global,
            h=self.h,
            max_layers=self.max_layers,
            material_colors=self.material_colors,
            material_TDs=self.material_TDs,
            background=self.background,
            perception_loss_module=self.perception_loss_module,
            add_penalty_loss=False,
        )
        loss.backward()
        self.optimizer.step()

        self.num_steps_done += 1

        # Optionally track the best "discrete" solution after a certain iteration
        if record_best:
            self._maybe_update_best_discrete()
        loss = loss.item()
        self.loss = loss

        return loss

    def log_to_tensorboard(self, interval: int = 100, namespace: str = "", step: int = None):
        """
        Log metrics and images to TensorBoard.

        Args:
            interval (int, optional): Interval for logging images. Defaults to 100.
            namespace (str, optional): Namespace prefix for logs. If provided, logs will be prefixed with this value. Defaults to "".
            step (int, optional): Optional override for the step number to log. Defaults to None.
        """
        if not self.log_to_tensorboard or self.writer is None:
            return

        # Prepare namespace prefix
        prefix = f"{namespace}/" if namespace else ""

        steps = step if step is not None else self.num_steps_done

        # Log metrics
        self.writer.add_scalar(f'Loss/{prefix}best_discrete', self.best_discrete_loss, steps)
        self.writer.add_scalar(f'Loss/{prefix}best_swaps', self.best_swaps, steps)

        tau_height, tau_global = self._get_tau()

        # Metrics that are only relevant for the main optimization loop
        if not prefix:
            self.writer.add_scalar('Params/tau_height', tau_height, steps)
            self.writer.add_scalar('Params/tau_global', tau_global, steps)
            self.writer.add_scalar('Params/lr', self.optimizer.param_groups[0]['lr'], steps)
            self.writer.add_scalar('Loss/train', self.loss, steps)

        # Log images periodically
        if (steps + 1) % interval == 0:
            with torch.no_grad():
                comp_img = composite_image(
                    self.params["pixel_height_logits"],
                    self.params["global_logits"],
                    tau_height,
                    tau_global,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background
                )
                self.writer.add_images(f'Current Output/{prefix}composite', comp_img.permute(2, 0, 1).unsqueeze(0) / 255.0, steps)

    def visualize(self, interval: int = 25):
        """
        Update the figure if visualize_flag is True.

        Args:
            interval (int, optional): Interval of steps to update the visualization. Defaults to 25.
        """
        if not self.visualize_flag:
            return

        # Update only every 'interval' steps for speed
        if (self.num_steps_done % interval) != 0:
            return

        with torch.no_grad():
            tau_h, tau_g = self._get_tau()
            comp = composite_image(
                self.params["pixel_height_logits"],
                self.params["global_logits"],
                tau_h,
                tau_g,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
                mode="continuous",
            )
        comp_np = np.clip(comp.cpu().detach().numpy(), 0, 255).astype(np.uint8)
        self.current_comp_ax.set_data(comp_np)

        if self.best_params is not None:
            with torch.no_grad():
                best_comp = composite_image(
                    self.best_params["pixel_height_logits"],
                    self.best_params["global_logits"],
                    self.final_tau,
                    self.final_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    mode="discrete",
                    rng_seed=self.best_seed,
                )
            best_comp_np = np.clip(best_comp.cpu().detach().numpy(), 0, 255).astype(
                np.uint8
            )
            self.best_comp_ax.set_data(best_comp_np)

        # Update the depth map correctly.
        with torch.no_grad():
            height_map = (self.max_layers * self.h) * torch.sigmoid(
                self.params["pixel_height_logits"]
            )
            height_map = height_map.cpu().detach().numpy()

        # Normalize safely, checking for a constant image.
        if np.allclose(height_map.max(), height_map.min()):
            height_map_norm = np.zeros_like(height_map)
        else:
            height_map_norm = (height_map - height_map.min()) / (
                height_map.max() - height_map.min()
            )

        height_map_uint8 = (height_map_norm * 255).astype(np.uint8)
        self.depth_map_ax.set_data(height_map_uint8)
        # Update the color limits so that the colormap correctly maps the range 0–255.
        self.depth_map_ax.set_clim(0, 255)

        self.fig.suptitle(
            f"Step {self.num_steps_done}, Tau: {tau_g:.4f}, Loss: {self.loss:.4f}, Best Discrete Loss: {self.best_discrete_loss:.4f}"
        )

        plt.pause(0.01)

    def get_current_parameters(self):
        """
        Return a copy of the current parameters (pixel_height_logits, global_logits).

        Returns:
            Dict[str, torch.Tensor]: Current parameters.
        """
        return {
            "pixel_height_logits": self.params["pixel_height_logits"].detach().clone(),
            "global_logits": self.params["global_logits"].detach().clone(),
        }

    def get_discretized_solution(
        self, best: bool = False, custom_height_logits: torch.Tensor = None
    ):
        """
        Return the discrete global assignment and the discrete pixel-height map
        for the current solution, using the current tau.

        Args:
            best (bool, optional): Whether to use the best solution. Defaults to False.
            custom_height_logits (torch.Tensor, optional): Custom height logits to use. We currently use this for the full size image. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Discrete global assignment and pixel-height map.
        """
        if best and self.best_params is None:
            return None, None

        current_params = self.best_params if best else self.params
        if custom_height_logits is not None:
            current_params["pixel_height_logits"] = custom_height_logits

        if best:
            disc_global, disc_height_image = discretize_solution(
                current_params,
                self.final_tau,
                self.h,
                self.max_layers,
                rng_seed=self.best_seed,
            )
            return disc_global, disc_height_image
        else:
            tau_height, tau_global = self._get_tau()
            with torch.no_grad():
                disc_global, disc_height_image = discretize_solution(
                    current_params,
                    tau_height,
                    self.h,
                    self.max_layers,
                    rng_seed=random.randrange(1, 1000000),
                )
            return disc_global, disc_height_image

    def prune(
        self,
        max_colors_allowed: int,
        max_swaps_allowed: int,
        disc_global: torch.Tensor = None,
        disc_height_image: torch.Tensor = None,
        tau_g: float = None,
    ):
        """
        Discretize and run pruning on the current solution in-place.

        Args:
            max_colors_allowed (int): Maximum number of colors allowed after pruning.
            max_swaps_allowed (int): Maximum number of swaps allowed after pruning.
            disc_global (torch.Tensor, optional): Discrete global assignment. Defaults to None.
            disc_height_image (torch.Tensor, optional): Pixel-height map. Defaults to None.
            tau_g (float, optional): Tau value for global logits. Defaults to None.


        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pruned discrete global assignment and pixel-height map.
        """
        if disc_global is None or disc_height_image is None:
            disc_global, disc_height_image = self.get_discretized_solution()
        if tau_g is None:
            tau_h, tau_g = self._get_tau()

        # Now run pruning
        disc_global_pruned = prune_colors_and_swaps(
            disc_global,
            self.params["pixel_height_logits"],
            self.target,
            self.h,
            self.max_layers,
            self.material_colors,
            self.material_TDs,
            self.background,
            max_colors_allowed,
            max_swaps_allowed,
            tau_for_comp=tau_g,
            rng_seed=self.best_seed,
            perception_loss_module=self.perception_loss_module,
        )

        return disc_global_pruned

    def _maybe_update_best_discrete(self):
        """
        Discretize the current solution, compute the discrete-mode loss,
        and update the best solution if it improves.
        """

        for i in range(1):
            # draw random integer seed
            seed = np.random.randint(0, 1000000)

            # 1) Discretize
            tau_h, tau_g = self.final_tau, self.final_tau
            disc_global, disc_height_image = discretize_solution(
                self.params, tau_g, self.h, self.max_layers, rng_seed=seed
            )

            # 2) Compute discrete-mode composite
            with torch.no_grad():
                comp_disc = composite_image(
                    self.params["pixel_height_logits"],
                    self.params["global_logits"],
                    self.final_tau,
                    self.final_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    mode="discrete",  # or "discrete" if you prefer fully discrete
                    rng_seed=seed,
                )

            # 3) Compute the "discrete" loss function
            #    If you have a separate function for that, just call it.
            #    Otherwise, reuse the standard compute_loss with add_penalty_loss=False, etc.
            current_disc_loss = compute_loss(
                material_assignment=disc_global,  # shape [max_layers,], discrete
                comp=comp_disc,
                target=self.target,
                perception_loss_module=self.perception_loss_module,
                tau_global=tau_g,
                num_materials=self.material_colors.shape[0],
                add_penalty_loss=False,
            ).item()

            # 4) Update if better
            if current_disc_loss < self.best_discrete_loss:
                self.best_discrete_loss = current_disc_loss
                self.best_params = self.get_current_parameters()
                self.best_tau = tau_g
                self.best_seed = seed
                self.best_swaps = len(find_color_bands(disc_global)) - 1

    def rng_seed_search(self, start_loss: float, num_seeds: int):
        """
        Search for the best seed for the best discrete solution.

        Args:
            start_loss (float): Initial loss value.
            num_seeds (int): Number of seeds to search.

        Returns:
            int: Best seed found.
        """
        best_seed = None
        best_loss = start_loss
        for i in tqdm(range(num_seeds), desc="Searching for new best seed"):
            seed = np.random.randint(0, 1000000)
            disc_global, disc_height_image = discretize_solution(
                self.best_params, self.final_tau, self.h, self.max_layers, rng_seed=seed
            )
            comp_disc = composite_image(
                self.best_params["pixel_height_logits"],
                self.best_params["global_logits"],
                self.final_tau,
                self.final_tau,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
                mode="discrete",
                rng_seed=seed,
            )
            current_disc_loss = compute_loss(
                material_assignment=disc_global,
                comp=comp_disc,
                target=self.target,
                perception_loss_module=self.perception_loss_module,
                tau_global=self.final_tau,
                num_materials=self.material_colors.shape[0],
                add_penalty_loss=False,
            ).item()
            if current_disc_loss < best_loss:
                best_loss = current_disc_loss
                best_seed = seed
        return best_seed, best_loss

    def __del__(self):
        """
        Clean up resources when the optimizer is destroyed.
        """
        if self.writer is not None:
            self.writer.close()
