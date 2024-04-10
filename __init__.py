import os
import importlib.util
from typing import Dict, Any

# Initialize empty dictionaries to hold merged mappings
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Directory of the current file (__init__.py)
current_directory = os.path.dirname(os.path.realpath(__file__))

# Iterate over all files in the current directory
for filename in os.listdir(current_directory):
    # Filter to only Python files (excluding __init__.py itself)
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        module_path = os.path.join(current_directory, filename)
        
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Attempt to merge NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            # Print the names of nodes loaded from this module
            for node_name in module.NODE_CLASS_MAPPINGS.keys():
                print(f"Loaded node class: {node_name}")
        
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            # Optionally, print the display names as well
            for display_name in module.NODE_DISPLAY_NAME_MAPPINGS.values():
                print(f"Loaded node display name: {display_name}")

# Export merged mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
