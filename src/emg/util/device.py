import torch


def get_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if "cuda" in device_str and torch.cuda.is_available():
        return torch.device(device_str)
    elif "mps" in device_str and torch.backends.mps.is_available():
        return torch.device(device_str)
    return torch.device("cpu")
