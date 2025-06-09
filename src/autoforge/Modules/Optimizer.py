import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoforge.Helper.CAdamW import CAdamW
from autoforge.Helper.OptimizerHelper import (
    composite_image_cont,
    composite_image_disc,
    deterministic_gumbel_softmax,
    PrecisionManager,
)

from autoforge.Loss.LossFunctions import loss_fn, compute_loss


class FilamentOptimizer:
    def __init__(
        self,
        args: argparse.Namespace,
        target: torch.Tensor,
        pixel_height_logits_init: np.ndarray,
        pixel_height_labels: np.ndarray,
        global_logits_init: np.ndarray,
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

        self.precision = PrecisionManager(device)

        pixel_height_labels = np.round(pixel_height_labels)

        # replace entire entries of pixel_height_logits with 0
        # pixel_height_logits_init *= 1.0
        # set pixel_height_logits where pixel_height_labels is 0 to -13.815512 (the lowest init sigmoid value)
        # pixel_height_logits_init[pixel_height_labels == 0] = -13.815512

        self.pixel_height_logits = torch.tensor(
            pixel_height_logits_init, dtype=torch.float32, device=device
        )
        self.pixel_height_logits.requires_grad_(False)

        print("layers", int(pixel_height_labels.flatten().max()))
        self.cluster_layers = int(pixel_height_labels.flatten().max()) + 1
        self.pixel_height_labels = torch.tensor(
            pixel_height_labels, dtype=torch.int32, device=device
        )
        self.height_offsets = torch.nn.Parameter(
            torch.zeros(self.cluster_layers, 1, device=device)
        )  # Trainable

        # Basic hyper-params
        self.material_colors = material_colors
        self.material_TDs = material_TDs
        self.background = background
        self.max_layers = args.max_layers
        self.h = args.layer_height
        self.learning_rate = args.learning_rate
        self.current_learning_rate = args.learning_rate
        self.final_tau = args.final_tau
        self.vis_tau = args.final_tau
        self.init_tau = args.init_tau
        self.device = device
        self.best_swaps = 0
        self.perception_loss_module = perception_loss_module
        self.visualize_flag = args.visualize

        # Initialize TensorBoard writer
        if args.tensorboard:
            if args.run_name:
                self.writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
            else:
                self.writer = SummaryWriter()
        else:
            self.writer = None

        if global_logits_init is None:
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
        global_logits_init = torch.from_numpy(global_logits_init).to(
            dtype=torch.float32, device=device
        )
        global_logits_init.requires_grad_(True)

        self.loss = None

        self.params = {
            "pixel_height_logits": self.pixel_height_logits,
            "global_logits": global_logits_init,
            "height_offsets": self.height_offsets,
        }

        # Tau schedule
        self.num_steps_done = 0
        self.warmup_steps = min(
            args.iterations - 1, args.warmup_fraction * args.iterations
        )
        self.decay_rate = (self.init_tau - self.final_tau) / (
            args.iterations - self.warmup_steps
        )

        # Initialize optimizer
        self.optimizer = CAdamW(
            [self.params["global_logits"], self.height_offsets],
            lr=self.learning_rate,
        )

        # Setup best discrete solution tracking
        self.best_discrete_loss = float("inf")
        self.best_params = None
        self.best_tau = None
        self.best_seed = None
        self.best_step = None

        # If you want a figure for real-time visualization:
        if self.visualize_flag:
            if self.args.disable_visualization_for_gradio != 1:
                plt.ion()
            self.fig, self.ax = plt.subplots(2, 3, figsize=(14, 6))

            self.target_im_ax = self.ax[0, 0].imshow(
                np.array(self.target.cpu(), dtype=np.uint8)
            )
            self.ax[0, 0].set_title("Target Image")

            self.current_comp_ax = self.ax[0, 1].imshow(
                np.zeros((self.H, self.W, 3), dtype=np.uint8)
            )
            self.ax[0, 1].set_title("Current Composite")

            self.best_comp_ax = self.ax[0, 2].imshow(
                np.zeros((self.H, self.W, 3), dtype=np.uint8)
            )
            self.ax[0, 2].set_title("Best Discrete Composite")
            if self.args.disable_visualization_for_gradio != 1:
                plt.pause(0.1)

            self.depth_map_ax = self.ax[1, 0].imshow(
                np.zeros((self.H, self.W), dtype=np.uint8), cmap="viridis"
            )
            self.ax[1, 0].set_title("Current Height Map")

            self.diff_depth_map_ax = self.ax[1, 1].imshow(
                np.zeros((self.H, self.W), dtype=np.uint8), cmap="viridis"
            )
            self.ax[1, 1].set_title("Height Map Changes")

            # target_vis = self.target.cpu().detach().numpy()
            # focus_vis = self.focus_map.cpu().detach().numpy()
            #
            # overlay = overlay_mask(
            #     to_pil_image(target_vis),
            #     to_pil_image(focus_vis, mode="F"),
            #     alpha=0.5,
            # )

            # Compute and store the initial height map for later difference computation.
            with torch.no_grad():
                initial_height = (self.max_layers * self.h) * torch.sigmoid(
                    self.params["pixel_height_logits"]
                )
            self.initial_height_map = initial_height.cpu().detach().numpy()

    def _apply_height_offset(self, pixel_logits=None, height_offsets=None):
        if pixel_logits is None:
            pixel_logits = self.pixel_height_logits
        if height_offsets is None:
            height_offsets = self.height_offsets

        offsets = torch.zeros_like(pixel_logits)
        nonzero_mask = self.pixel_height_labels != 0
        offsets[nonzero_mask] = height_offsets[
            self.pixel_height_labels[nonzero_mask]
        ].squeeze(-1)
        return pixel_logits + offsets

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
            # simple linear decay
            t = max(
                self.final_tau, tau_init - self.decay_rate * (i - self.warmup_steps)
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
        if self.pixel_height_logits.grad is not None:
            self.pixel_height_logits.grad = None

        self.optimizer.zero_grad()

        warmup_steps = int(
            self.args.iterations * self.args.learning_rate_warmup_fraction
        )

        if self.num_steps_done < warmup_steps and warmup_steps > 0:
            lr_scale = self.num_steps_done / warmup_steps
            self.current_learning_rate = lr_scale * self.learning_rate
        else:
            self.current_learning_rate = self.learning_rate

        for g in self.optimizer.param_groups:
            g["lr"] = self.current_learning_rate

        tau_height, tau_global = self._get_tau()

        # start_fraction = self.args.height_logits_learning_start_fraction
        # full_fraction = self.args.height_logits_learning_full_fraction

        # total_iters = self.args.iterations
        # current_step = self.num_steps_done
        # if current_step < start_fraction * total_iters:
        #     scaling = 0.0
        # elif current_step < full_fraction * total_iters:
        #     scaling = (current_step - start_fraction * total_iters) / ((full_fraction - start_fraction) * total_iters)
        # else:
        #     scaling = 1.0

        effective_logits = self._apply_height_offset()

        loss = loss_fn(
            {
                "pixel_height_logits": effective_logits,
                "global_logits": self.params["global_logits"],
            },
            target=self.target,
            tau_height=tau_height,
            tau_global=tau_global,
            h=self.h,
            max_layers=self.max_layers,
            material_colors=self.material_colors,
            material_TDs=self.material_TDs,
            background=self.background,
            add_penalty_loss=10.0,
            focus_map=None,
            focus_strength=0.0,
        )

        self.precision.backward_and_step(loss, self.optimizer)

        self.num_steps_done += 1

        # Optionally track the best "discrete" solution after a certain iteration
        if record_best:
            self._maybe_update_best_discrete()
        # torch.cuda.empty_cache()
        loss = loss.item()
        self.loss = loss

        return loss

    def discretize_solution(
        self,
        params: dict,
        tau_global: float,
        h: float,
        max_layers: int,
        rng_seed: int = -1,
    ):
        """
        Convert continuous logs to discrete layer counts and discrete color IDs.

        Args:
            params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
            tau_global (float): Temperature parameter for global material assignment.
            h (float): Height of each layer.
            max_layers (int): Maximum number of layers.
            rng_seed (int, optional): Random seed for deterministic sampling. Defaults to -1.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Discrete global material assignments, shape [max_layers].
                - torch.Tensor: Discrete height image, shape [H, W].
        """
        pixel_logits = params["pixel_height_logits"]
        pixel_offset = params["height_offsets"]
        effective_logits = self._apply_height_offset(
            pixel_logits=pixel_logits, height_offsets=pixel_offset
        )

        global_logits = params["global_logits"]
        pixel_heights = (max_layers * h) * torch.sigmoid(effective_logits)
        discrete_height_image = torch.round(pixel_heights / h).to(torch.int32)
        discrete_height_image = torch.clamp(discrete_height_image, 0, max_layers)

        num_layers = global_logits.shape[0]
        discrete_global_vals = []
        for j in range(num_layers):
            p = deterministic_gumbel_softmax(
                global_logits[j], tau_global, hard=True, rng_seed=rng_seed + j
            )
            discrete_global_vals.append(torch.argmax(p))
        discrete_global = torch.stack(discrete_global_vals, dim=0)
        return discrete_global, discrete_height_image

    def log_to_tensorboard(
        self, interval: int = 100, namespace: str = "", step: int = None
    ):
        """
        Log metrics and images to TensorBoard.

        Args:
            interval (int, optional): Interval for logging images. Defaults to 100.
            namespace (str, optional): Namespace prefix for logs. If provided, logs will be prefixed with this value. Defaults to "".
            step (int, optional): Optional override for the step number to log. Defaults to None.
        """
        with torch.no_grad():
            if not self.log_to_tensorboard or self.writer is None:
                return

            # Prepare namespace prefix
            prefix = f"{namespace}/" if namespace else ""

            steps = step if step is not None else self.num_steps_done

            # Log metrics
            self.writer.add_scalar(
                f"Loss/{prefix}best_discrete", self.best_discrete_loss, steps
            )
            self.writer.add_scalar(f"Loss/{prefix}best_swaps", self.best_swaps, steps)

            tau_height, tau_global = self._get_tau()

            # Metrics that are only relevant for the main optimization loop
            if not prefix:
                self.writer.add_scalar("Params/tau_height", tau_height, steps)
                self.writer.add_scalar("Params/tau_global", tau_global, steps)
                self.writer.add_scalar(
                    "Params/lr", self.optimizer.param_groups[0]["lr"], steps
                )
                self.writer.add_scalar("Loss/train", self.loss, steps)

            # Log images periodically
            if (steps + 1) % interval == 0:
                with torch.no_grad():
                    effective_logits = self._apply_height_offset()
                    comp_img = composite_image_cont(
                        effective_logits,
                        self.params["global_logits"],
                        tau_height,
                        tau_global,
                        self.h,
                        self.max_layers,
                        self.material_colors,
                        self.material_TDs,
                        self.background,
                    )
                    self.writer.add_images(
                        f"Current Output/{prefix}composite",
                        comp_img.permute(2, 0, 1).unsqueeze(0) / 255.0,
                        steps,
                    )

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
            effective_logits = self._apply_height_offset()
            comp = composite_image_cont(
                effective_logits,
                self.params["global_logits"],
                tau_h,
                tau_g,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
            )
            comp_np = np.clip(comp.cpu().detach().numpy(), 0, 255).astype(np.uint8)
            self.current_comp_ax.set_data(comp_np)

            if self.best_params is not None:
                # Update the depth map correctly.
                effective_best_logits = self._apply_height_offset(
                    self.best_params["pixel_height_logits"],
                    self.best_params["height_offsets"],
                )
                best_comp = composite_image_disc(
                    effective_best_logits,
                    self.best_params["global_logits"],
                    self.vis_tau,
                    self.vis_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    rng_seed=self.best_seed,
                )
                best_comp_np = np.clip(best_comp.cpu().detach().numpy(), 0, 255).astype(
                    np.uint8
                )
                self.best_comp_ax.set_data(best_comp_np)

            # Update the depth map correctly.
            effective_logits = self._apply_height_offset()
            height_map = (self.max_layers * self.h) * torch.sigmoid(effective_logits)
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
            self.depth_map_ax.set_clim(0, 255)

            # Compute and update the difference depth map (current - initial)
            diff_map = height_map - self.initial_height_map
            # print(diff_map.min(), diff_map.max())
            # Normalize the difference map safely.
            # if np.allclose(diff_map.max(), diff_map.min()):
            #     diff_map_norm = np.zeros_like(diff_map)
            # else:
            #     diff_map_norm = (diff_map - diff_map.min()) / (
            #         diff_map.max() - diff_map.min()
            #     )
            self.diff_depth_map_ax.set_data(diff_map)
            self.diff_depth_map_ax.set_clim(-2.5, 2.5)

            self.fig.suptitle(
                f"Step {self.num_steps_done}/{self.args.iterations}, Tau: {tau_g:.4f}, Loss: {self.loss:.4f}, Best Discrete Loss: {self.best_discrete_loss:.4f}"
            )
            if self.args.disable_visualization_for_gradio != 1:
                plt.pause(0.01)
            plt.savefig(self.args.output_folder + "/vis_temp.png")

    def get_current_parameters(self):
        """
        Return a copy of the current parameters (pixel_height_logits, global_logits).

        Returns:
            Dict[str, torch.Tensor]: Current parameters.
        """
        return {
            "pixel_height_logits": self.pixel_height_logits.detach().clone(),
            "global_logits": self.params["global_logits"].detach().clone(),
            "height_offsets": self.height_offsets.detach().clone(),
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

        current_params = self.best_params.copy() if best else self.params
        if custom_height_logits is not None:
            current_params["pixel_height_logits"] = self._apply_height_offset(
                custom_height_logits
            )

        if best:
            disc_global, disc_height_image = self.discretize_solution(
                self.best_params,
                self.vis_tau,
                self.h,
                self.max_layers,
                rng_seed=self.best_seed,
            )
            return disc_global, disc_height_image
        else:
            tau_height, tau_global = self._get_tau()
            with torch.no_grad():
                disc_global, disc_height_image = self.discretize_solution(
                    current_params,
                    tau_height,
                    self.h,
                    self.max_layers,
                    rng_seed=random.randrange(1, 1000000),
                )
            return disc_global, disc_height_image

    def get_best_discretized_image(
        self,
        custom_height_logits: torch.Tensor = None,
        custom_global_logits: torch.Tensor = None,
    ):
        with torch.no_grad():
            effective_logits = self._apply_height_offset(
                self.best_params["pixel_height_logits"],
                self.best_params["height_offsets"],
            )
            best_comp = composite_image_disc(
                effective_logits
                if custom_height_logits is None
                else self._apply_height_offset(custom_height_logits),
                self.best_params["global_logits"]
                if custom_global_logits is None
                else custom_global_logits,
                self.vis_tau,
                self.vis_tau,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
                rng_seed=self.best_seed,
            )
        return best_comp

    def prune(
        self,
        max_colors_allowed: int,
        max_swaps_allowed: int,
        min_layers_allowed: int,
        max_layers_allowed: int,
        search_seed: bool = True,
        fast_pruning: bool = False,
        fast_pruning_percent: float = 0.20,
    ):
        # Now run pruning
        from autoforge.Helper.PruningHelper import (
            prune_num_colors,
            prune_num_swaps,
            prune_redundant_layers,
            optimise_swap_positions,
        )

        if search_seed:
            self.rng_seed_search(self.best_discrete_loss, 100, autoset_seed=True)

        #clear pytorch and system cache to reduce vram usage
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        prune_num_colors(
            self,
            max_colors_allowed,
            self.vis_tau,
            None,
            fast=fast_pruning,
            chunking_percent=fast_pruning_percent,
        )

        prune_num_swaps(
            self,
            max_swaps_allowed,
            self.vis_tau,
            None,
            fast=fast_pruning,
            chunking_percent=fast_pruning_percent,
        )

        prune_redundant_layers(
            self,
            None,
            min_layers_allowed,
            max_layers_allowed,
            fast=fast_pruning,
            chunking_percent=fast_pruning_percent,
        )

        optimise_swap_positions(self)

    def _maybe_update_best_discrete(self):
        """
        Discretize the current solution, compute the discrete-mode loss,
        and update the best solution if it improves.
        """

        for i in range(1):
            # draw random integer seed
            seed = np.random.randint(0, 1000000)

            # 1) Discretize
            tau_h, tau_g = self.vis_tau, self.vis_tau
            with torch.no_grad():
                effective_logits = self._apply_height_offset()
                disc_global, disc_height_image = self.discretize_solution(
                    self.params, tau_g, self.h, self.max_layers, rng_seed=seed
                )

                # 2) Compute discrete-mode composite
                with torch.no_grad():
                    comp_disc = composite_image_disc(
                        effective_logits,
                        self.params["global_logits"],
                        self.vis_tau,
                        self.vis_tau,
                        self.h,
                        self.max_layers,
                        self.material_colors,
                        self.material_TDs,
                        self.background,
                        rng_seed=seed,
                    )

                current_disc_loss = compute_loss(
                    comp=comp_disc,
                    target=self.target,
                ).item()
                from autoforge.Helper.PruningHelper import find_color_bands

                # 4) Update if better
                if current_disc_loss < self.best_discrete_loss:
                    self.best_discrete_loss = current_disc_loss
                    self.best_params = self.get_current_parameters()
                    self.best_tau = tau_g
                    self.best_seed = seed
                    self.best_swaps = len(find_color_bands(disc_global)) - 1
                    self.best_step = self.num_steps_done

    def rng_seed_search(
        self, start_loss: float, num_seeds: int, autoset_seed: bool = False
    ):
        """
        Search for the best seed for the best discrete solution.

        Args:
            start_loss (float): Initial loss value.
            num_seeds (int): Number of seeds to search.
            autoset_seed (bool, optional): Whether to automatically set the seed. Defaults to False.

        Returns:
            int: Best seed found.
        """
        best_seed = None
        best_loss = start_loss
        for i in tqdm(range(num_seeds), desc="Searching for new best seed"):
            seed = np.random.randint(0, 1000000)
            effective_logits = self._apply_height_offset(
                self.best_params["pixel_height_logits"],
                self.best_params["height_offsets"],
            )
            comp_disc = composite_image_disc(
                effective_logits,
                self.best_params["global_logits"],
                self.vis_tau,
                self.vis_tau,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
                rng_seed=seed,
            )
            current_disc_loss = compute_loss(
                comp=comp_disc,
                target=self.target,
            ).item()
            if current_disc_loss < best_loss:
                best_loss = current_disc_loss
                best_seed = seed
        if autoset_seed and best_loss < start_loss:
            self.best_seed = best_seed
        return best_seed, best_loss

    def __del__(self):
        """
        Clean up resources when the optimizer is destroyed.
        """
        if self.writer is not None:
            self.writer.close()
