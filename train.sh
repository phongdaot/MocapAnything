# conda activate mocapanything
export PYTHONPATH=$PYTHONPATH:./TripoSG

torchrun --nproc_per_node=8 --master_port=12345 -m train.pose2rot \
    --config configs/train/train_pose2rot.yaml

torchrun --nproc_per_node=8 --master_port=12345 -m train.video2pose \
    --config configs/train/train_video2pose.yaml
    
torchrun --nproc_per_node=8 --master_port=12345 -m train.video2pose2rot \
    --config configs/train/train_video2pose2rot.yaml

torchrun --nproc_per_node=8 --master_port=29506 -m train.video2mesh \
--config configs/train/train_video2mesh.yaml