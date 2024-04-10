import os
from PIL import Image, ImageOps
import torch
import numpy as np

class LoadFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Correctly returning IMAGE as per framework's convention
    FUNCTION = "load_folder"
    CATEGORY = "Null Nodes"
    OUTPUT_NODE = False

    def __init__(self) -> None:
        print("LoadFolder Init")

    def load_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist.")
            return None

        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            print(f"No image files found in {folder_path}.")
            return None

        output_images = []
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB") if img.mode != 'RGB' else img
            img_np = np.array(img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(img_np).permute(2, 0, 1)[None, :])

        if output_images:
            output_batch = torch.cat(output_images, dim=0)
            return (output_batch,)
        else:
            print("No valid images were loaded.")
            return None

# Registration for LoadFolder
NODE_CLASS_MAPPINGS = {
    "LoadFolder": LoadFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFolder": "Load Folder",
}


# class LoadImage:
#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = folder_paths.get_input_directory()
#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
#         return {"required":
#                     {"image": (sorted(files), {"image_upload": True})},
#                 }

#     CATEGORY = "image"

#     RETURN_TYPES = ("IMAGE", "MASK")
#     FUNCTION = "load_image"
#     def load_image(self, image):
#         image_path = folder_paths.get_annotated_filepath(image)
#         img = Image.open(image_path)
#         output_images = []
#         output_masks = []
#         for i in ImageSequence.Iterator(img):
#             i = ImageOps.exif_transpose(i)
#             if i.mode == 'I':
#                 i = i.point(lambda i: i * (1 / 255))
#             image = i.convert("RGB")
#             image = np.array(image).astype(np.float32) / 255.0
#             image = torch.from_numpy(image)[None,]
#             if 'A' in i.getbands():
#                 mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
#                 mask = 1. - torch.from_numpy(mask)
#             else:
#                 mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
#             output_images.append(image)
#             output_masks.append(mask.unsqueeze(0))

#         if len(output_images) > 1:
#             output_image = torch.cat(output_images, dim=0)
#             output_mask = torch.cat(output_masks, dim=0)
#         else:
#             output_image = output_images[0]
#             output_mask = output_masks[0]

#         return (output_image, output_mask)
