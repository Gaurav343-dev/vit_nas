def get_parameters_size(model, unit="M"):
    """Utility function to compute total number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if unit == "K":
        return f"{total_params / 1e3:.2f}K"
    elif unit == "M":
        return f"{total_params / 1e6:.2f}M"
    else:
        return total_params

def get_macs(model) -> int:
    """Return MACs for the model's currently active subnet.
    Call model.set_active_subnet(config) before this to measure a specific subnet.

    Args:
        model: SuperNet instance.
    Returns:        
        MACs (multiply-accumulate operations) for the active subnet.
    """
    return model.get_macs()


def get_peak_memory(model, img_size: int = 32, batch_size: int = 1,
                    unit: str = "MB", method: str = "analytical") -> float:
    """Estimate peak activation memory during a forward pass.

    Args:
        model: SuperNet (call set_active_subnet first) or SubNet
        img_size: spatial size of input image
        batch_size: batch size
        unit: "B", "KB", "MB", or "GB"
        method: "analytical" — formula-based, no forward pass needed
                "empirical"  — measures actual GPU allocation (requires CUDA)
                "both"       — returns dict with both estimates

    Returns:
        float (peak memory in requested unit) for analytical/empirical,
        or dict {"analytical_MB": ..., "empirical_MB": ...} for "both".
    """
    import torch

    divisors = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    scale = divisors.get(unit, 1e6)

    def _analytical():
        assert hasattr(model, "active_embed_dim"), \
            "analytical method requires a SuperNet with active_embed_dim set"
        B = batch_size
        E = model.active_embed_dim
        L = model.active_num_layers
        num_patches = (model.patch_embed.img_size // model.patch_embed.patch_size) ** 2
        T = num_patches + 1  # +1 for CLS token
        bytes_per_elem = 4   # float32

        # patch embed output baseline
        peak_elems = B * T * E

        for i in range(L):
            H = model.active_num_heads[i]
            M = model.active_mlp_dim[i]

            # peak candidates within this block:
            #   QKV output   : B × T × 3E
            #   attn scores  : B × H × T × T  ← quadratic, usually the bottleneck
            #   attn output  : B × T × E
            #   MLP hidden   : B × T × M
            block_peak = max(
                B * T * 3 * E,
                B * H * T * T,
                B * T * E,
                B * T * M,
            )
            peak_elems = max(peak_elems, block_peak)

        return (peak_elems * bytes_per_elem) / scale

    def _empirical():
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(
                "empirical peak memory requires a CUDA device. "
                "Move the model to GPU first, or use method='analytical'."
            )
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / scale

    if method == "analytical":
        return round(_analytical(), 4)
    elif method == "empirical":
        return round(_empirical(), 4)
    elif method == "both":
        return {
            f"analytical_{unit}": round(_analytical(), 4),
            f"empirical_{unit}":  round(_empirical(), 4),
        }
    else:
        raise ValueError(f"method must be 'analytical', 'empirical', or 'both', got {method!r}")