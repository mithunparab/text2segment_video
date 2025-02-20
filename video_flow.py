import os
import torch
import cv2
import numpy as np

def load_raft_model(device, raft_path):
    """
    Loads the RAFT model dynamically without modifying sys.path.

    Args:
        device (str): The device to load the model on ('cuda' or 'cpu').
        raft_path (str): Path to the RAFT model directory.

    Returns:
        RAFT model instance.
    """
    try:
        raft_core = __import__("raft.core.raft", fromlist=["RAFT"])
        raft_utils = __import__("raft.core.utils.utils", fromlist=["InputPadder"])
        raft_config = __import__("raft.config", fromlist=["RAFTConfig"])
    except ImportError as e:
        raise ImportError("RAFT module not found. Ensure that RAFT is installed and accessible.") from e

    RAFT = raft_core.RAFT
    InputPadder = raft_utils.InputPadder
    RAFTConfig = raft_config.RAFTConfig

    config = RAFTConfig(dropout=0, alternate_corr=False, small=False, mixed_precision=False)
    model = RAFT(config)
    
    model_weights = os.path.join(raft_path, 'raft-sintel.pth')
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"RAFT weights not found at {model_weights}. Please download the weights.")

    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device).eval()

    return model, InputPadder

def compute_flow_and_foreground(image1: torch.Tensor, image2: torch.Tensor, model, padder_class, device: str, threshold: float = 2.0):
    """
    Computes optical flow between two frames using RAFT and extracts foreground based on motion magnitude.

    Args:
        image1 (torch.Tensor): Previous frame.
        image2 (torch.Tensor): Current frame.
        model: RAFT model instance.
        padder_class: The InputPadder class for handling padding.
        device (str): Device (CPU/GPU) for computation.
        threshold (float): Motion magnitude threshold.

    Returns:
        tuple: (Optical flow, Foreground mask, Foreground image)
    """
    padder = padder_class(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        _, flow_up = model(image1, image2, iters=20, test_mode=True)

    # Compute flow magnitude
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
    magnitude = np.linalg.norm(flow, axis=2)

    # Threshold to create motion mask
    mask = (magnitude > threshold).astype(np.uint8) * 255

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Convert mask to 3 channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Extract foreground
    img1_np = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    foreground = cv2.bitwise_and(img1_np, mask_3ch)

    return flow_up, mask, foreground
