import argparse
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoforge.Helper.CAdamW import CAdamW
from autoforge.Helper.OptimizerHelper import (
    composite_image_cont_clusters,
    composite_image_disc_clusters,
    cluster_logits_to_pixel_height_logits,
    _cluster_sample_heights,
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
        perception_loss_module: Optional[torch.nn.Module],
        focus_map: Optional[torch.Tensor] = None,
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
            focus_map (torch.Tensor | None): Optional priority mask [H,W] in [0,1]. Higher -> higher loss weight.
        """
        self.args = args
        self.target = target  # smaller (solver) resolution, shape [H,W,3], float32
        self.H, self.W = target.shape[:2]

        self.precision = PrecisionManager(device)

        # Labels define cluster assignment per pixel. We'll learn one height distribution per cluster.
        pixel_height_labels = np.round(pixel_height_labels)
        self.pixel_height_labels = torch.tensor(
            pixel_height_labels, dtype=torch.long, device=device
        )
        # Number of clusters (including background if label==0 is used)
        print("layers", int(self.pixel_height_labels.flatten().max().item()))
        self.cluster_layers = int(self.pixel_height_labels.flatten().max().item()) + 1

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

        # Priority mask
        self.focus_map = None
        if focus_map is not None:
            fm = focus_map
            if fm.dim() == 3 and fm.shape[-1] == 1:
                fm = fm.squeeze(-1)
            self.focus_map = fm.to(device=self.device, dtype=torch.float32)

        # Initialize TensorBoard writer
        if args.tensorboard:
            if args.run_name:
                self.writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
            else:
                self.writer = SummaryWriter()
        else:
            self.writer = None

        # Flag used by log_to_tensorboard()
        self.tensorboard_log = bool(getattr(args, "tensorboard", False))

        # Initialize global logits (unchanged)
        if global_logits_init is None:
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
        if isinstance(global_logits_init, np.ndarray):
            global_logits_init = torch.from_numpy(global_logits_init).to(
                dtype=torch.float32, device=device
            )
        elif torch.is_tensor(global_logits_init):
            global_logits_init = global_logits_init.to(
                dtype=torch.float32, device=device
            )
        else:
            raise TypeError("global_logits_init must be a numpy array or torch Tensor")
        global_logits_init.requires_grad_(True)

        # New: Per-cluster height distribution logits over layers [C, L]
        # We'll initialize from the provided per-pixel initial logits by estimating
        # a histogram of heights per cluster.
        # Convert initial per-pixel logits to a coarse height estimate z in [0, L]
        if isinstance(pixel_height_logits_init, np.ndarray):
            phl0 = torch.from_numpy(pixel_height_logits_init).to(
                dtype=torch.float32, device=device
            )
        elif torch.is_tensor(pixel_height_logits_init):
            phl0 = pixel_height_logits_init.to(dtype=torch.float32, device=device)
        else:
            raise TypeError("pixel_height_logits_init must be a numpy array or torch Tensor")
        with torch.no_grad():
            s = torch.sigmoid(phl0)
            z_est = torch.clamp((self.max_layers * s), 0.0, float(self.max_layers))
            z_idx = torch.clamp(torch.round(z_est), 0, self.max_layers).to(torch.long)
            C, L = self.cluster_layers, self.max_layers
            counts = torch.zeros((C, L + 1), device=device, dtype=torch.float32)
            z_idx_clamped = torch.clamp(z_idx, 0, L - 1)
            flat_labels = self.pixel_height_labels.view(-1)
            flat_z = z_idx_clamped.view(-1)
            for c in range(C):
                mask = flat_labels == c
                if mask.any():
                    hist = torch.bincount(flat_z[mask], minlength=L)
                    counts[c, :L] = hist.to(torch.float32)
            init_cluster_logits = torch.log(counts[:, :L] + 1.0)
            for c in range(C):
                if torch.all(init_cluster_logits[c].isneginf() | torch.isnan(init_cluster_logits[c])):
                    init_cluster_logits[c] = 0.0
        self.cluster_height_logits = torch.nn.Parameter(init_cluster_logits.clone().detach())

        # Transitional compatibility: snapshot per-pixel logits derived from clusters
        self.pixel_height_logits = cluster_logits_to_pixel_height_logits(
            self.cluster_height_logits.detach(),
            self.pixel_height_labels,
            self.init_tau,
            self.max_layers,
            hard=False,
            rng_seed=0,
        ).detach().clone()

        self.loss = None

        # Registered learnable params for the optimizer
        self.params = {
            "cluster_height_logits": self.cluster_height_logits,
            "global_logits": global_logits_init,
        }

        # Tau schedule
        self.num_steps_done = 0
        self.warmup_steps = min(
            args.iterations - 1, args.warmup_fraction * args.iterations
        )
        self.decay_rate = (self.init_tau - self.final_tau) / (
            args.iterations - self.warmup_steps
        )

        # Initialize optimizer (remove height offsets and pixel logits)
        self.optimizer = CAdamW(
            [self.params["global_logits"], self.cluster_height_logits],
            lr=self.learning_rate,
        )

        # Setup best discrete solution tracking
        self.best_discrete_loss = float("inf")
        self.best_params = None
        self.best_tau = None
        self.best_seed = None
        self.best_step = None
        # Default zero height offsets for backward-compat modules
        self.height_offsets = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)

        # Visualization setup
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

            # Priority mask visualization in bottom-right
            if self.focus_map is not None:
                fm_np = self.focus_map.cpu().detach().numpy()
                fm_min, fm_max = float(fm_np.min()), float(fm_np.max())
                if fm_max - fm_min > 1e-8:
                    fm_norm = (fm_np - fm_min) / (fm_max - fm_min)
                else:
                    fm_norm = np.zeros_like(fm_np)
                fm_uint8 = (fm_norm * 255).astype(np.uint8)
                self.priority_mask_ax = self.ax[1, 2].imshow(
                    fm_uint8, cmap="magma", vmin=0, vmax=255
                )
                self.ax[1, 2].set_title("Priority Mask")
            else:
                self.ax[1, 2].text(
                    0.5,
                    0.5,
                    "No Priority Mask",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                    transform=self.ax[1, 2].transAxes,
                )
                self.ax[1, 2].set_axis_off()

            # Store the initial height map for later difference computation.
            with torch.no_grad():
                init_logits = cluster_logits_to_pixel_height_logits(
                    self.cluster_height_logits,
                    self.pixel_height_labels,
                    self.init_tau,
                    self.max_layers,
                    hard=False,
                    rng_seed=0,
                )
                initial_height = (self.max_layers * self.h) * torch.sigmoid(init_logits)
            self.initial_height_map = initial_height.cpu().detach().numpy()

    # --- New height modeling helpers ---
    def _cluster_probs(self, tau_height: float, hard: bool = False, rng_seed: int = 0) -> torch.Tensor:
        """
        Returns per-cluster distribution over layers [C, L].
        If hard=True uses deterministic gumbel softmax to produce one-hot per cluster.
        """
        logits = self.cluster_height_logits
        eps = 1e-8
        t = max(tau_height, eps)
        if hard:
            # Produce one-hot per cluster
            C = logits.shape[0]
            outs = []
            for c in range(C):
                one_hot = deterministic_gumbel_softmax(logits[c], t, True, rng_seed + c)
                outs.append(one_hot)
            return torch.stack(outs, dim=0)  # [C,L]
        else:
            return torch.softmax(logits / t, dim=1)

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
        self.optimizer.zero_grad()
        warmup_steps = int(self.args.iterations * self.args.learning_rate_warmup_fraction)
        if self.num_steps_done < warmup_steps and warmup_steps > 0:
            lr_scale = self.num_steps_done / warmup_steps
            self.current_learning_rate = lr_scale * self.learning_rate
        else:
            self.current_learning_rate = self.learning_rate
        for g in self.optimizer.param_groups:
            g["lr"] = self.current_learning_rate
        tau_height, tau_global = self._get_tau()

        # Direct cluster-based forward pass (height gradient flows through cluster_height_logits)
        loss = loss_fn(
            {
                "cluster_height_logits": self.cluster_height_logits,
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
            focus_map=self.focus_map,
            focus_strength=10.0,
            pixel_height_labels=self.pixel_height_labels,
        )

        self.precision.backward_and_step(loss, self.optimizer)

        self.num_steps_done += 1

        if record_best:
            self._maybe_update_best_discrete()
        loss_val = loss.item()
        self.loss = loss_val

        return loss_val

    def discretize_solution(
        self,
        params: dict,
        tau_global: float,
        h: float,
        max_layers: int,
        rng_seed: int = -1,
    ):
        """
        Convert to discrete layer counts and discrete color IDs.

        Returns:
            tuple: (discrete global materials [max_layers], discrete height image [H,W])
        """
        # Heights: hard sample per cluster using deterministic Gumbel-Softmax
        seed = 0 if rng_seed < 0 else rng_seed
        pixel_logits = cluster_logits_to_pixel_height_logits(
            self.cluster_height_logits,
            self.pixel_height_labels,
            self.vis_tau,
            self.max_layers,
            hard=True,
            rng_seed=seed,
        )
        # Directly compute discrete height in layers
        z_disc = torch.round((self.max_layers * torch.sigmoid(pixel_logits)) / 1.0)
        discrete_height_image = torch.clamp(z_disc.to(torch.int32), 0, max_layers)

        # Materials: same as before
        global_logits = params["global_logits"]
        num_layers = global_logits.shape[0]
        discrete_global_vals = []
        for j in range(num_layers):
            p = deterministic_gumbel_softmax(
                global_logits[j], tau_global, hard=True, rng_seed=seed + j
            )
            discrete_global_vals.append(torch.argmax(p))
        discrete_global = torch.stack(discrete_global_vals, dim=0)
        return discrete_global, discrete_height_image

    def log_to_tensorboard(
        self, interval: int = 100, namespace: str = "", step: int = None
    ):
        """
        Log metrics and images to TensorBoard.
        """
        with torch.no_grad():
            if not self.tensorboard_log or self.writer is None:
                return

            prefix = f"{namespace}/" if namespace else ""
            steps = step if step is not None else self.num_steps_done

            self.writer.add_scalar(f"Loss/{prefix}best_discrete", self.best_discrete_loss, steps)
            self.writer.add_scalar(f"Loss/{prefix}best_swaps", self.best_swaps, steps)

            tau_height, tau_global = self._get_tau()

            if not prefix:
                self.writer.add_scalar("Params/tau_height", tau_height, steps)
                self.writer.add_scalar("Params/tau_global", tau_global, steps)
                self.writer.add_scalar("Params/lr", self.optimizer.param_groups[0]["lr"], steps)
                self.writer.add_scalar("Loss/train", self.loss, steps)

            if (steps + 1) % interval == 0:
                with torch.no_grad():
                    comp_img = composite_image_cont_clusters(
                        self.cluster_height_logits,
                        self.pixel_height_labels,
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
        """
        if not self.visualize_flag or (self.num_steps_done % interval) != 0:
            return

        with torch.no_grad():
            tau_h, tau_g = self._get_tau()
            comp = composite_image_cont_clusters(
                self.cluster_height_logits,
                self.pixel_height_labels,
                self.params["global_logits"],
                tau_h,
                tau_g,
                self.h,
                self.max_layers,
                self.material_colors,
                self.material_TDs,
                self.background,
            )
            comp_np = np.clip(comp.cpu().numpy(), 0, 255).astype(np.uint8)
            self.current_comp_ax.set_data(comp_np)

            if self.best_params is not None:
                best_comp = composite_image_disc_clusters(
                    self.best_params["cluster_height_logits"],
                    self.pixel_height_labels,
                    self.best_params["global_logits"],
                    self.vis_tau,
                    self.vis_tau,
                    self.h,
                    self.max_layers,
                    self.material_colors,
                    self.material_TDs,
                    self.background,
                    rng_seed=self.best_seed if self.best_seed is not None else -1,
                )
                self.best_comp_ax.set_data(np.clip(best_comp.cpu().numpy(), 0, 255).astype(np.uint8))

            # Sample for height map visualization (shows discrete sampled heights)
            z_idx = _cluster_sample_heights(self.cluster_height_logits, tau_h, int(self.num_steps_done))
            z_map = z_idx[self.pixel_height_labels].to(torch.float32) * self.h
            height_map = z_map.cpu().numpy()
            if np.allclose(height_map.max(), height_map.min()):
                height_map_norm = np.zeros_like(height_map)
            else:
                height_map_norm = (height_map - height_map.min()) / (height_map.max() - height_map.min())
            self.depth_map_ax.set_data((height_map_norm * 255).astype(np.uint8))
            self.depth_map_ax.set_clim(0, 255)

            diff_map = height_map - self.initial_height_map
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
        Return a copy of the current parameters.
        """
        return {
            "cluster_height_logits": self.cluster_height_logits.detach().clone(),
            "global_logits": self.params["global_logits"].detach().clone(),
            "pixel_height_logits": cluster_logits_to_pixel_height_logits(
                self.cluster_height_logits.detach(),
                self.pixel_height_labels,
                self._get_tau()[0],
                self.max_layers,
                hard=False,
                rng_seed=0,
            ).detach().clone(),
            "height_offsets": self.height_offsets.detach().clone(),
        }

    def get_discretized_solution(
        self, best: bool = False, custom_height_logits: torch.Tensor = None
    ):
        """
        Return the discrete global assignment and the discrete pixel-height map
        for the current solution, using the current tau.
        """
        if best and self.best_params is None:
            return None, None

        current_params = self.best_params.copy() if best else self.params
        if best:
            disc_global, disc_height_image = self.discretize_solution(
                current_params,
                self.vis_tau,
                self.h,
                self.max_layers,
                rng_seed=self.best_seed if self.best_seed is not None else -1,
            )
            return disc_global, disc_height_image
        else:
            tau_height, tau_global = self._get_tau()
            with torch.no_grad():
                disc_global, disc_height_image = self.discretize_solution(
                    current_params,
                    tau_global,
                    self.h,
                    self.max_layers,
                    rng_seed=random.randrange(1, 1000000),
                )
            return disc_global, disc_height_image

    def get_best_discretized_image(
        self,
        custom_global_logits: torch.Tensor = None,
    ):
        with torch.no_grad():
            best_comp = composite_image_disc_clusters(
                self.best_params["cluster_height_logits"],
                self.pixel_height_labels,
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
                rng_seed=self.best_seed if self.best_seed is not None else -1,
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

        for _ in range(1):
            seed = np.random.randint(0, 1000000)

            tau_g = self.vis_tau
            with torch.no_grad():
                disc_global, disc_height_image = self.discretize_solution(
                    self.params, tau_g, self.h, self.max_layers, rng_seed=seed
                )

                comp_disc = composite_image_disc_clusters(
                    self.cluster_height_logits,
                    self.pixel_height_labels,
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
                    focus_map=self.focus_map,
                ).item()
                from autoforge.Helper.PruningHelper import find_color_bands

                if current_disc_loss < self.best_discrete_loss:
                    self.best_discrete_loss = current_disc_loss
                    self.best_params = {
                        "cluster_height_logits": self.cluster_height_logits.detach().clone(),
                        "global_logits": self.params["global_logits"].detach().clone(),
                        "pixel_height_logits": cluster_logits_to_pixel_height_logits(
                            self.cluster_height_logits.detach(),
                            self.pixel_height_labels,
                            self.vis_tau,
                            self.max_layers,
                            hard=True,
                            rng_seed=seed,
                        ).detach().clone(),
                        "height_offsets": self.height_offsets.detach().clone(),
                    }
                    self.best_tau = tau_g
                    self.best_seed = seed
                    self.best_swaps = len(find_color_bands(disc_global)) - 1
                    self.best_step = self.num_steps_done

    def rng_seed_search(
        self, start_loss: float, num_seeds: int, autoset_seed: bool = False
    ):
        """
        Search for the best seed for the best discrete solution.
        """
        best_seed = None
        best_loss = start_loss
        for _ in tqdm(range(num_seeds), desc="Searching for new best seed"):
            seed = np.random.randint(0, 1000000)
            comp_disc = composite_image_disc_clusters(
                self.cluster_height_logits,
                self.pixel_height_labels,
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
                focus_map=self.focus_map,
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

    def _apply_height_offset(self, pixel_height_logits: torch.Tensor, height_offsets: torch.Tensor | None):
        """Compatibility helper for legacy pruning code: apply additive height offsets to pixel logits.
        If height_offsets is None, returns pixel_height_logits unchanged.
        Resizes offsets if needed to match the logits spatial resolution.
        """
        if height_offsets is None:
            return pixel_height_logits
        offs = height_offsets
        # Ensure device/dtype
        offs = offs.to(device=pixel_height_logits.device, dtype=pixel_height_logits.dtype)
        if offs.shape != pixel_height_logits.shape:
            # Resize to logits shape using bilinear on NCHW
            offs_nchw = offs.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            H, W = pixel_height_logits.shape
            offs_resized = torch.nn.functional.interpolate(
                offs_nchw, size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
            offs = offs_resized
        return pixel_height_logits + offs
