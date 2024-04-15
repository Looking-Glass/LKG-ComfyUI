import torch
import torch.nn.functional as F
import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview

class GenerateMasks:
    """
    Takes a pair of images (color and depth) and generates masks for 4 infill regions,
    for a composite 3D effect. Assumes the input images are tensors in the format [B, H, W, C] with B=1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_image": ("IMAGE",),  # Color image
                "depth_image": ("IMAGE",),  # Depth image
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Background Infill Area", "Middle 0 Infill Area", "Middle 1 Infill Area", "Highlight Infill Area")
    FUNCTION = "generate_masks"
    CATEGORY = "Null Nodes"

    def generate_masks(self, color_image, depth_image):
        if color_image is None or depth_image is None:
            print("One or both images are None.")
            return None, None, None, None

        # Normalize and resize depth image to match color image dimensions
        target_height, target_width = color_image.shape[1:3]
        depth_image = depth_image.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]
        depth_resized = F.interpolate(depth_image, size=(target_height, target_width), mode='bilinear', align_corners=False)
        depth_resized = depth_resized.permute(0, 2, 3, 1)  # Back to [B, H, W, C]
        depth_norm = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min())

        # Calculate dynamic thresholds based on depth distribution
        depth_flat = depth_norm.reshape(-1).cpu().numpy()
        thresholds_values = np.percentile(depth_flat, [25, 50, 75, 90])
        thresholds = torch.tensor(thresholds_values, device=depth_norm.device, dtype=depth_norm.dtype)

        masks = []
        for i, threshold in enumerate(thresholds):
            mask = (depth_norm > threshold).float()

            # Expand the mask to match the number of color channels, typically 3 (RGB)
            if color_image.shape[3] == 3:  # Assuming color channels are last
                mask = mask.expand(-1, -1, -1, 3)

            # Convert the mask to the same dtype as the input image (typically uint8)
            mask = (mask * 255).type(torch.uint8)

            masks.append(mask)

        return tuple(masks)

NODE_CLASS_MAPPINGS = {
    "GenerateMasks": GenerateMasks
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateMasks": "Generate Masks Node"
}
