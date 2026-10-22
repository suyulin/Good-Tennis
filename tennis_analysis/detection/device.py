"""Device selection shared by the PyTorch YOLO detectors."""


def select_yolo_device(device="auto"):
    """Prefer CUDA, then Apple MPS, then CPU; keep explicit selections."""
    if device in (None, "auto"):
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except (ImportError, RuntimeError, AttributeError):
            device = "cpu"

    # Keep MPS in FP32, as on CPU. Existing CUDA inference retains FP16.
    half = str(device).split(":")[0] not in ("cpu", "mps")
    return device, half
