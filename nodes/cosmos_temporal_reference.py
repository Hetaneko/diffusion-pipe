import torch


class ApplyCosmosReferenceLatent:
    """Attach optional Cosmos reference latents and role IDs to model kwargs."""

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/cosmos"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "layout_latent": ("LATENT",),
                "character_latent": ("LATENT",),
                "background_latent": ("LATENT",),
            },
        }

    @staticmethod
    def _samples(latent):
        return latent["samples"] if isinstance(latent, dict) else latent

    @staticmethod
    def _as_reference_tensor(latent):
        samples = ApplyCosmosReferenceLatent._samples(latent)
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        if samples.ndim != 5:
            raise ValueError(f"Cosmos reference latents must be 4D or 5D tensors, got shape {tuple(samples.shape)}")
        return samples

    def apply(self, latent, layout_latent=None, character_latent=None, background_latent=None):
        latent_out = latent.copy() if isinstance(latent, dict) else {"samples": latent}
        model_kwargs = dict(latent_out.get("model_kwargs", {}))

        active_refs = []
        if layout_latent is not None:
            active_refs.append((1, self._as_reference_tensor(layout_latent)))
        if character_latent is not None:
            active_refs.append((2, self._as_reference_tensor(character_latent)))
        if background_latent is not None:
            active_refs.append((3, self._as_reference_tensor(background_latent)))

        if active_refs:
            reference_latents = torch.cat([ref_latent for _, ref_latent in active_refs], dim=2)
            active_control_ref_ids = torch.tensor(
                [ref_id for ref_id, _ in active_refs],
                dtype=torch.long,
                device=reference_latents.device,
            )
            model_kwargs["control_latents"] = reference_latents
            model_kwargs["active_control_ref_ids"] = active_control_ref_ids
        else:
            model_kwargs.pop("control_latents", None)
            model_kwargs["active_control_ref_ids"] = torch.empty(0, dtype=torch.long)

        latent_out["model_kwargs"] = model_kwargs
        return (latent_out,)


NODE_CLASS_MAPPINGS = {
    "ApplyCosmosReferenceLatent": ApplyCosmosReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCosmosReferenceLatent": "Apply Cosmos Reference Latent",
}
