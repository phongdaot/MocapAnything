import os
import sys
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mesh import blender_visualize_character_motion


character_folder = 'zoo/characters_fix_facezplus'
blender_pth = '/home/ma-user/work/g00826954/projects/proj_4d/TripoSG/dataset/blender-3.6.14-linux-x64/blender'
hdri_pth = 'transparent'

bvh_list = sorted(glob('zoo/bvh/*/*.bvh'))

for motion_pth in bvh_list:
    view = motion_pth.split('/')[-1][:-4]
    motion_name = motion_pth.split('/')[-2]
    out_dir = f'zoo/video/{motion_name}'
    print(motion_name)
    character = os.path.join(character_folder, motion_name.split('#')[0])
    blender_visualize_character_motion(
        out_dir,
        motion_pth,
        character_folder=character,
        auto_scale='bvh',
        blender_path=blender_pth,
        hdri_path=hdri_pth,
        view_scale=1.3,
        object_position=1.0,
        camera_trace=False,
        fps=30,
    )
