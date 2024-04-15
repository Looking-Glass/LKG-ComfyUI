import torch
import subprocess
import os
from PIL import Image
import numpy as np
import folder_paths 

class ProcessRGBDMask:
    """
    Takes an RGBD image, processes it through a CLI program, and saves the resulting image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgbd_image": ("IMAGE",),  # Combined RGB and Depth image
                "filename_prefix": ("STRING", {"default": "ProcessedRGBD"})  # Default filename prefix
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process_rgbd_image"
    CATEGORY = "Null Nodes"

    def process_rgbd_image(self, rgbd_image, filename_prefix="ProcessedRGBD"):
        if rgbd_image is None:
            print("The RGBD image is None.")
            return

        # Determine the aspect ratio of the input image
        aspect_ratio = rgbd_image.shape[2] / 2 / rgbd_image.shape[1]

        # Save the image to a temporary file
        input_path = os.path.abspath("temp_rgbd.jpg")
        self.save_image(rgbd_image, input_path, format='JPEG')

        # Get the correct output path and filename
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, rgbd_image.shape[2], rgbd_image.shape[1])
        
        # Format filename with batch number (0), counter, and required postfix
        columns = 10
        rows = 6
        output_filename = f"{filename.replace('%batch_num%', '0')}_{counter:05}_qs{columns}x{rows}a{aspect_ratio:.2f}.jpg"
        output_path = os.path.join(full_output_folder, output_filename)

        # Call the CLI program with the correct output path
        self.quiltify_rgbd(input_path, output_path, aspect_ratio)

        return (None,)

    def save_image(self, tensor_image, file_path, format='JPEG'):
        # Convert tensor image to uint8 format and save as specified format
        if tensor_image.dim() == 4:
            tensor_image = tensor_image.squeeze(0)
        if tensor_image.dtype != torch.uint8:
            tensor_image = (tensor_image * 255).byte()
        image = Image.fromarray(tensor_image.cpu().numpy())
        image.save(file_path, format=format)

    def quiltify_rgbd(self, input_filename, output_filename, aspect_ratio):
        cli_executable_path = os.path.abspath("quiltify/cli.exe")
        command = [
            cli_executable_path,
            "-operation", "quiltify_RGBD",
            "-output_width", "8192",
            "-output_height", "8192",
            "-input", input_filename,
            "-output", output_filename,
            "-aspect", str(aspect_ratio),
            "-columns", "10",
            "-rows", "6",
            "-views", "60",
            "-zoom", "1",
            "-cam_dist", "2",
            "-fov", "40",
            "-crop_pos_x", "0",
            "-crop_pos_y", "0",
            "-depth_inversion", "false",
            "-chroma_depth", "false",
            "-depth_loc", "right",
            "-depthiness", "2.7",
            "-depth_cutoff", "1.0",
            "-focus", "0"
        ]


        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Quiltify RGBD operation completed successfully.")
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running quiltify RGBD: {e}")
            print("Output:", e.stdout)
            print("Errors:", e.stderr)
        except FileNotFoundError as e:
            print(f"Failed to find the executable or one of the input/output files: {e}")



NODE_CLASS_MAPPINGS = {
    "ProcessRGBDMask": ProcessRGBDMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProcessRGBDMask": "Process RGBD Image Node"
}
