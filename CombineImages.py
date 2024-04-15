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

class CombineImagesSideBySide:
    """
    Correctly combines two images side by side into one image with double the width.
    Assumes the input images are tensors in the format [B, H, W, C].
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # color image to be placed on the left
                "image2": ("IMAGE",),  # depth image to be placed on the right
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "Null Nodes"

    def combine(self, image1, image2):
        # Validate inputs
        if image1 is None:
            return (image2,)
        if image2 is None:
            return (image1,)
        
        print(f"Image 1 Shape: {image1.shape}")
        print(f"Image 2 Shape: {image2.shape}")

        # Resize image2 to match the height of image1
        target_height = image1.shape[1]
        aspect_ratio_image2 = image2.shape[2] / image2.shape[1]
        new_width_image2 = int(target_height * aspect_ratio_image2)
        
        # Ensure image2 is resized to have the same height as image1
        image2_resized = torch.nn.functional.interpolate(image2.permute(0, 3, 1, 2), size=(target_height, new_width_image2), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # Combine images side by side along the width dimension
        result = torch.cat((image1, image2_resized), 2)  # Concatenate along the width dimension, assuming the format is [B, H, W, C]
        
        return (result,)



NODE_CLASS_MAPPINGS = {
    "SideBySide": CombineImagesSideBySide
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SideBySide": "Side by Side Node"
}