### finder.py ###
import os

def get_image_seq_relpath(image_folder, image_root_list):
    """
    Convert an image folder path into a relative subfolder under whichever image_root it belongs to.
    Example:
        image_root = /data/selected_nbg_zoo1010
        image_folder = /data/selected_nbg_zoo1010/Fox#run_y60
        return -> Fox#run_y60
    """
    image_folder_abs = os.path.abspath(image_folder)

    for image_root in image_root_list:
        image_root_abs = os.path.abspath(image_root)
        try:
            common = os.path.commonpath([image_folder_abs, image_root_abs])
        except ValueError:
            continue

        if common == image_root_abs:
            return os.path.relpath(image_folder_abs, image_root_abs)

    return os.path.basename(image_folder_abs)

def resolve_npz_info(seq_name, base_dir):
    npz_path = os.path.join(base_dir, "npz_train", f"{seq_name.replace('_', '/')}.npz")
    if os.path.exists(npz_path):
        return npz_path, False  # GT exists
    else:
        species = seq_name.split("#")[0]
        species_npz_dir = os.path.join(base_dir, "npz_train")
        for name in sorted(os.listdir(species_npz_dir)):
            if name.startswith(species + "#"):
                subdir = os.path.join(species_npz_dir, name)
                if os.path.isdir(subdir):
                    files = sorted(os.listdir(subdir))
                    if files:
                        return os.path.join(subdir, files[0]), True
        return None, True
    

def find_all_valid_glb_sequences(glb_root_dir):
    """
    Traverse all subfolders of glb_root_dir and find folders containing multiple .glb files.
    Returns: List[(seq_name, full_glb_folder_path)]
    """
    valid_seq_folders = []
    for root, dirs, files in os.walk(glb_root_dir):
        glb_files = [f for f in files if f.endswith('.glb')]
        if len(glb_files) > 1:
            rel_path = os.path.relpath(root, glb_root_dir)
            top_folder = rel_path.split(os.sep)[0]
            valid_seq_folders.append((top_folder, root))
    return valid_seq_folders



def find_image_folder(seq_name, image_root_list):
    for image_root in image_root_list:
        for folder in os.listdir(image_root):
            full_path = os.path.join(image_root, folder)
            if folder.startswith(seq_name) and os.path.isdir(full_path):
                return full_path
    return None


def find_all_valid_image_sequences(image_root_list, min_images=2):
    """
    Traverse all image roots and find folders containing at least `min_images` image files.
    Returns:
        List[(seq_name, full_image_folder_path)]
    Notes:
        - seq_name is taken as the folder name
        - if the same seq_name appears in multiple roots, the first one found is kept
    """
    valid_seq_folders = []
    seen = set()

    for image_root in image_root_list:
        if not os.path.isdir(image_root):
            continue

        for folder in sorted(os.listdir(image_root)):
            full_path = os.path.join(image_root, folder)
            if not os.path.isdir(full_path):
                continue

            img_files = [
                f for f in os.listdir(full_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(img_files) >= min_images and folder not in seen:
                valid_seq_folders.append((folder, full_path))
                seen.add(folder)

    return valid_seq_folders