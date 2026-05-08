### config_utils.py ###
import importlib
import yaml
import os
from .dist_utils import is_main_process

def load_json(pth):
    with open(pth, 'r') as file:
        data_dict = json.load(file)
    return data_dict

def load_yaml_config(path):

    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml_config(config, path):
    
    # Check if the file already exists, if so, check if the content is the same

    if os.path.exists(path):
        existing_config = load_yaml_config(path)
        if existing_config == config:
            return
        else:
            raise ValueError(f"Config at {path} already exists and is different. Please check the file or choose a different path.")
    else:
        if is_main_process():
            with open(path, "w") as f:
                yaml.dump(config, f)

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(obj_or_str, reload=False):
    if not isinstance(obj_or_str, str):
        return obj_or_str

    module, name = obj_or_str.rsplit(".", 1)

    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)

    return getattr(module_imp, name)
