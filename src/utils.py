# Utility functions
import yaml

def load_config(config_path="config/config.yaml"): 
    """Loads configuration settings from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) 
    return config