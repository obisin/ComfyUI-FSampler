# Utility functions and CFGGuider copied from ComfyUI
# Original source: ComfyUI/comfy/k_diffusion/sampling.py and ComfyUI/comfy/samplers.py

import torch
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks


def append_zero(x):
    """Appends a zero to the end of a tensor."""
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded


# Import necessary helper functions from reference
def preprocess_conds_hooks(conds):
    """Preprocess conditioning hooks - imported from comfy.samplers"""
    # This is handled by comfy.samplers.preprocess_conds_hooks
    from comfy.samplers import preprocess_conds_hooks as comfy_preprocess
    return comfy_preprocess(conds)


def filter_registered_hooks_on_conds(conds, model_options):
    """Filter registered hooks on conds - imported from comfy.samplers"""
    from comfy.samplers import filter_registered_hooks_on_conds as comfy_filter
    return comfy_filter(conds, model_options)


def get_total_hook_groups_in_conds(conds):
    """Get total hook groups in conds - imported from comfy.samplers"""
    from comfy.samplers import get_total_hook_groups_in_conds as comfy_get_total
    return comfy_get_total(conds)


def cast_to_load_options(model_options, device=None, dtype=None):
    """Cast to load options - imported from comfy.samplers"""
    from comfy.samplers import cast_to_load_options as comfy_cast
    return comfy_cast(model_options, device, dtype)


def process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None):
    """Process conditioning - imported from comfy.samplers"""
    from comfy.samplers import process_conds as comfy_process
    return comfy_process(model, noise, conds, device, latent_image, denoise_mask, seed)


def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    """Main sampling function - imported from comfy.samplers"""
    from comfy.samplers import sampling_function as comfy_sampling
    return comfy_sampling(model, x, timestep, uncond, cond, cond_scale, model_options, seed)


class CFGGuider:
    """
    CFGGuider class copied from ComfyUI/comfy/samplers.py
    Handles Classifier-Free Guidance and model preparation for sampling.
    """
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.outer_predict_noise(*args, **kwargs)

    def outer_predict_noise(self, x, timestep, model_options={}, seed=None):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self.predict_noise,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, self.model_options, is_model_options=True)
        ).execute(x, timestep, model_options, seed)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0:
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds
        return output