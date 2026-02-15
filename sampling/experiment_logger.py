"""Experiment logger for FSampler research data collection."""
import os
import json
import csv
import time
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import folder_paths


class ExperimentLogger:
    """Handles saving FSampler experiment runs to disk with all metadata and outputs."""

    def __init__(self, base_path=None):
        """
        Initialize the experiment logger.

        Args:
            base_path: Base directory for experiments (defaults to ComfyUI/output/experiments/)
                      If empty string, uses default. If provided, uses that exact path.
        """
        if base_path is None or base_path == "":
            # Default to ComfyUI/output/experiments/ using ComfyUI's standard path helper
            output_dir = folder_paths.get_output_directory()
            base_path = os.path.join(output_dir, "experiments")

        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_run(self, image_tensor, metadata_json, baseline_image_tensor=None,
                 positive_prompt="", negative_prompt="", is_baseline=False):
        """
        Save a complete experiment run to disk.

        Args:
            image_tensor: Output image from VAE decode (torch tensor or PIL Image)
            metadata_json: JSON string containing FSampler metadata
            baseline_image_tensor: Optional baseline image for comparison
            positive_prompt: Positive prompt string
            negative_prompt: Negative prompt string
            is_baseline: Whether this run is a baseline (no skip mode)

        Returns:
            Path to the created experiment directory
        """
        # Parse metadata
        try:
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        except Exception as e:
            print(f"[ExperimentLogger] Failed to parse metadata: {e}")
            metadata = {}

        # Create timestamped directory
        timestamp_start = metadata.get("timestamp_start", time.time())
        dt = datetime.fromtimestamp(timestamp_start)
        timestamp_str = dt.strftime("%Y-%m-%d_%H-%M-%S")
        unix_timestamp = int(timestamp_start)

        # Extract model type(s)
        model_type_str = ""
        if "model_0" in metadata or "model_1" in metadata:
            # Multi-model workflow
            types = []
            if "model_0" in metadata and metadata["model_0"].get("model_type"):
                types.append(metadata["model_0"]["model_type"])
            if "model_1" in metadata and metadata["model_1"].get("model_type"):
                types.append(metadata["model_1"]["model_type"])
            if types:
                model_type_str = "_" + "+".join(types)
        elif metadata.get("model_type"):
            # Single model workflow
            model_type_str = "_" + metadata["model_type"]

        # Directory name: YYYY-MM-DD_HH-MM-SS_{unix_timestamp}_{model_type}
        dir_name = f"{timestamp_str}_{unix_timestamp}{model_type_str}"
        if is_baseline:
            dir_name += "_baseline"

        exp_dir = os.path.join(self.base_path, dir_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Capture device info
        device_info = self._get_device_info()

        # Add device info and prompts to metadata
        metadata["device_info"] = device_info
        metadata["positive_prompt"] = positive_prompt
        metadata["negative_prompt"] = negative_prompt
        metadata["is_baseline"] = is_baseline

        # Save output image
        output_image_path = os.path.join(exp_dir, "output_image.png")
        self._save_image(image_tensor, output_image_path)

        # Save baseline image if provided
        if baseline_image_tensor is not None:
            baseline_image_path = os.path.join(exp_dir, "baseline.png")
            self._save_image(baseline_image_tensor, baseline_image_path)

            # Compute metrics (SSIM, RMSE, MAE) against baseline
            try:
                from .metrics import compute_metrics
                metrics = compute_metrics(image_tensor, baseline_image_tensor)
                metadata["metrics"] = metrics
                print(f"[ExperimentLogger] Computed metrics: SSIM={metrics['ssim']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
            except Exception as e:
                print(f"[ExperimentLogger] Failed to compute metrics: {e}")
                metadata["metrics"] = None

        # Save metadata JSON
        metadata_path = os.path.join(exp_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save per-step data CSV
        per_step_data = metadata.get("per_step_data", [])
        if per_step_data:
            csv_path = os.path.join(exp_dir, "per_step_data.csv")
            self._save_per_step_csv(per_step_data, csv_path)

        # Save sigmas CSV (if available in per_step_data)
        if per_step_data:
            sigmas_csv_path = os.path.join(exp_dir, "sigmas.csv")
            self._save_sigmas_csv(per_step_data, sigmas_csv_path)

        # Save summary.txt
        summary_path = os.path.join(exp_dir, "summary.txt")
        self._save_summary(metadata, summary_path)

        print(f"[ExperimentLogger] Saved experiment to: {exp_dir}")
        return exp_dir

    def _get_device_info(self):
        """Capture device information (GPU, memory, etc.)."""
        device_info = {}

        try:
            if torch.cuda.is_available():
                device_info["device"] = f"cuda:{torch.cuda.current_device()}"
                device_info["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
                device_info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device_info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                device_info["reserved_memory_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            else:
                device_info["device"] = "cpu"
        except Exception as e:
            device_info["error"] = str(e)

        return device_info

    def _save_image(self, image_data, path):
        """
        Save image tensor or PIL Image to disk.

        Args:
            image_data: Can be torch.Tensor (from VAE) or PIL.Image
            path: Output path for PNG file
        """
        try:
            if isinstance(image_data, torch.Tensor):
                # Convert tensor to PIL Image
                # Assume tensor is in shape [B, C, H, W] or [C, H, W]
                if image_data.dim() == 4:
                    # Batch dimension, take first image
                    image_data = image_data[0]

                # Convert to numpy
                img_np = image_data.cpu().numpy()

                # If channels first, convert to channels last
                if img_np.shape[0] in [1, 3, 4]:
                    img_np = np.transpose(img_np, (1, 2, 0))

                # Clip and convert to uint8
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

                # Convert to PIL
                if img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np[:, :, 0], mode='L')
                elif img_np.shape[2] == 3:
                    img_pil = Image.fromarray(img_np, mode='RGB')
                elif img_np.shape[2] == 4:
                    img_pil = Image.fromarray(img_np, mode='RGBA')
                else:
                    raise ValueError(f"Unexpected number of channels: {img_np.shape[2]}")

                img_pil.save(path)
            elif isinstance(image_data, Image.Image):
                # Already PIL Image
                image_data.save(path)
            else:
                print(f"[ExperimentLogger] Unknown image type: {type(image_data)}")
        except Exception as e:
            print(f"[ExperimentLogger] Failed to save image to {path}: {e}")

    def _save_per_step_csv(self, per_step_data, path):
        """Save per-step data to CSV."""
        if not per_step_data:
            return

        try:
            with open(path, 'w', newline='') as f:
                # Get all unique keys from all step dicts
                all_keys = set()
                for step in per_step_data:
                    all_keys.update(step.keys())

                fieldnames = sorted(all_keys)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_step_data)
        except Exception as e:
            print(f"[ExperimentLogger] Failed to save per-step CSV: {e}")

    def _save_sigmas_csv(self, per_step_data, path):
        """Save sigma schedule to CSV."""
        if not per_step_data:
            return

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step_index', 'sigma_current', 'sigma_next'])
                for step in per_step_data:
                    writer.writerow([
                        step.get('step_index', ''),
                        step.get('sigma_current', ''),
                        step.get('sigma_next', '')
                    ])
        except Exception as e:
            print(f"[ExperimentLogger] Failed to save sigmas CSV: {e}")

    def _save_summary(self, metadata, path):
        """Save human-readable summary.txt."""
        try:
            with open(path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FSampler Experiment Run Summary\n")
                f.write("=" * 80 + "\n\n")

                # Check if this is a multi-model workflow
                is_multi_model = "model_0" in metadata or "model_1" in metadata

                if is_multi_model:
                    # Multi-model workflow
                    f.write("Multi-Model Workflow Detected\n")
                    f.write("=" * 80 + "\n\n")

                    for model_key in ["model_0", "model_1"]:
                        if model_key in metadata:
                            model_meta = metadata[model_key]
                            f.write(f"\n{'='*80}\n")
                            f.write(f"{model_key.upper()} (Base Model)\n" if model_key == "model_0" else f"{model_key.upper()} (Refiner Model)\n")
                            f.write(f"{'='*80}\n\n")
                            self._write_single_model_summary(f, model_meta)

                    # Global info at the bottom
                    f.write("\n" + "="*80 + "\n")
                    f.write("Global Information\n")
                    f.write("="*80 + "\n")
                    f.write(f"Is Baseline: {metadata.get('is_baseline', False)}\n")

                    # Device info
                    device_info = metadata.get("device_info", {})
                    if device_info:
                        f.write("\nDevice Information:\n")
                        f.write("-" * 80 + "\n")
                        for key, value in device_info.items():
                            f.write(f"{key}: {value}\n")

                    # Prompts
                    f.write("\nPrompts:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Positive: {metadata.get('positive_prompt', 'N/A')}\n")
                    f.write(f"Negative: {metadata.get('negative_prompt', 'N/A')}\n")

                    # Metrics
                    if "metrics" in metadata and metadata["metrics"]:
                        f.write("\nImage Quality Metrics (vs Baseline):\n")
                        f.write("-" * 80 + "\n")
                        m = metadata["metrics"]
                        f.write(f"SSIM: {m['ssim']:.6f} (structural similarity, higher=better)\n")
                        f.write(f"RMSE: {m['rmse']:.6f} (root mean square error, lower=better)\n")
                        f.write(f"MAE:  {m['mae']:.6f} (mean absolute error, lower=better)\n")

                else:
                    # Single model workflow
                    self._write_single_model_summary(f, metadata)

                    # Device info
                    device_info = metadata.get("device_info", {})
                    if device_info:
                        f.write("Device Information:\n")
                        f.write("-" * 80 + "\n")
                        for key, value in device_info.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")

                    # Prompts
                    f.write("Prompts:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Positive: {metadata.get('positive_prompt', 'N/A')}\n")
                    f.write(f"Negative: {metadata.get('negative_prompt', 'N/A')}\n")
                    f.write("\n")

                    # Metrics
                    if "metrics" in metadata and metadata["metrics"]:
                        f.write("Image Quality Metrics (vs Baseline):\n")
                        f.write("-" * 80 + "\n")
                        m = metadata["metrics"]
                        f.write(f"SSIM: {m['ssim']:.6f} (structural similarity, higher=better)\n")
                        f.write(f"RMSE: {m['rmse']:.6f} (root mean square error, lower=better)\n")
                        f.write(f"MAE:  {m['mae']:.6f} (mean absolute error, lower=better)\n")
                        f.write("\n")

                f.write("=" * 80 + "\n")
        except Exception as e:
            print(f"[ExperimentLogger] Failed to save summary: {e}")

    def _write_single_model_summary(self, f, metadata):
        """Write summary for a single model's metadata."""
        # Basic info
        f.write("Run Information:\n")
        f.write("-" * 80 + "\n")
        if "timestamp_start" in metadata:
            dt = datetime.fromtimestamp(metadata["timestamp_start"])
            f.write(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {metadata.get('model_type', 'N/A')}\n")
        f.write(f"Seed: {metadata.get('seed', 'N/A')}\n")
        if "is_baseline" in metadata:
            f.write(f"Is Baseline: {metadata.get('is_baseline', False)}\n")
        f.write("\n")

        # Sampler settings
        f.write("Sampler Settings:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Sampler: {metadata.get('sampler', 'N/A')}\n")
        f.write(f"Scheduler: {metadata.get('scheduler', 'N/A')}\n")
        f.write(f"Skip Mode: {metadata.get('skip_mode', 'N/A')}\n")
        f.write(f"Adaptive Mode: {metadata.get('adaptive_mode', 'N/A')}\n")
        f.write(f"Smoothing Beta: {metadata.get('smoothing_beta', 'N/A')}\n")
        f.write(f"Protect First Steps: {metadata.get('protect_first_steps', 'N/A')}\n")
        f.write(f"Protect Last Steps: {metadata.get('protect_last_steps', 'N/A')}\n")
        if metadata.get('anchor_interval') is not None:
            f.write(f"Anchor Interval: {metadata.get('anchor_interval')}\n")
        if metadata.get('max_consecutive_skips') is not None:
            f.write(f"Max Consecutive Skips: {metadata.get('max_consecutive_skips')}\n")
        f.write("\n")

        # Performance metrics
        f.write("Performance Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Steps: {metadata.get('total_steps', 'N/A')}\n")
        f.write(f"Model Calls: {metadata.get('model_calls', 'N/A')}\n")
        f.write(f"Skipped: {metadata.get('skipped', 'N/A')}\n")
        f.write(f"Reduction: {metadata.get('reduction_percent', 0.0):.2f}%\n")
        f.write(f"Total Time: {metadata.get('total_time_seconds', 0.0):.2f}s\n")
        f.write("\n")

        # Learning stabilizer stats
        f.write("Learning Stabilizer (L) Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"L Final: {metadata.get('l_final', 1.0):.4f}\n")
        f.write(f"L Mean: {metadata.get('l_mean', 1.0):.4f}\n")
        f.write(f"L Min: {metadata.get('l_min', 1.0):.4f}\n")
        f.write(f"L Max: {metadata.get('l_max', 1.0):.4f}\n")
        f.write("\n")
