# conda activate mocapanything

export PYTHONPATH=$PYTHONPATH:./TripoSG

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m inference.video2mesh --config configs/inference/inference_video2mesh.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m inference.mesh2pose --config configs/inference/inference_mesh2pose.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m inference.video2pose2rot --config configs/inference/inference_video2pose2rot.yaml

# mocap
source ~/.bashrc
conda activate tripoSG
export PYTHONPATH=$PYTHONPATH:./TripoSG
CUDA_VISIBLE_DEVICES=1 python3 -m inference.video2pose2rot --config configs/inference/inference_video2pose2rot_v2_obj.yaml


source ~/.bashrc
conda activate tripoSG
export PYTHONPATH=$PYTHONPATH:./TripoSG
CUDA_VISIBLE_DEVICES=2 python3 -m inference.video2pose2rot --config configs/inference/inference_video2pose2rot_v2_zoo_wild.yaml


source ~/.bashrc
conda activate tripoSG
export PYTHONPATH=$PYTHONPATH:./TripoSG
CUDA_VISIBLE_DEVICES=3 python3 -m inference.video2pose2rot --config configs/inference/inference_video2pose2rot_v2_zoo.yaml