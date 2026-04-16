def get_parameters_size(model, unit="M"):
    """Utility function to compute total number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if unit == "K":
        return f"{total_params / 1e3:.2f}K"
    elif unit == "M":
        return f"{total_params / 1e6:.2f}M"
    else:
        return total_params

def get_flops(model, input_size) -> float:
    pass  # Placeholder for FLOPs calculation, which can be complex and may require additional libraries like ptflops or fvcore.

def get_peak_memory(model, input_size) -> float:
    pass  # Placeholder for peak memory calculation, which can be complex and may require additional libraries or custom hooks to measure during a forward pass.