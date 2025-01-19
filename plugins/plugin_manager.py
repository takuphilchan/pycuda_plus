"""
Manager for dynamically loading and managing PyCUDA plugins.
"""
import importlib
import os

class PluginManager:
    def __init__(self, plugin_folder):
        """Initialize the plugin manager with a folder path."""
        self.plugin_folder = plugin_folder
        self.plugins = {}

    def load_plugin(self, plugin_name):
        """Dynamically load a plugin by name."""
        try:
            module_path = f"pycuda_plus.plugins.{plugin_name}"
            module = importlib.import_module(module_path)
            self.plugins[plugin_name] = module
            return module
        except ModuleNotFoundError as e:
            raise ImportError(f"Plugin {plugin_name} could not be loaded.") from e

    def get_plugin(self, plugin_name):
        """Retrieve a loaded plugin by name."""
        return self.plugins.get(plugin_name, None)
