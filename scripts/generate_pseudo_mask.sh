# NEED TO SET
# DATASET_ROOT=PATH/TO/DATASET
DATASET_ROOT=/data/coco_2014
GPU=0,1,2,3

# 2.1. train an attribute manipulation network
# CUDA_VISIBLE_DEVICES=0 python3 run_sample_coco.py \
#     --coco14_root ${DATASET_ROOT} \
#     --num_workers 8 \
#     --train_amn_pass True

# 2.2. generate activation maps and refined seed for boundary refinement
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_coco.py \
    --coco14_root ${DATASET_ROOT} \
    --num_workers 8 \
    --make_amn_cam_pass True \
    --eval_amn_cam_pass True \
    --amn_cam_to_ir_label_pass True \
    --conf_fg_thres 0.45 \
    --conf_bg_thres 0.15

# 3.1. train a boundary refinement network (IRN)
# CUDA_VISIBLE_DEVICES=0 python3 run_sample_coco.py \
#     --coco14_root ${DATASET_ROOT} \
#     --num_workers 8 \
#     --train_irn_pass True

# 3.2. generate the final pseudo-masks
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_coco.py \
    --coco14_root ${DATASET_ROOT} \
    --num_workers 8 \
    --make_sem_seg_pass True \
    --eval_sem_seg_pass True
