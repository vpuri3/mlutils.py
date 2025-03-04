#
# export CUDA_VISIBLE_DEVICES="0,1,2"

# # restart from checkpoint
# EXP_NAME="exp"
# torchrun --nproc-per-node gpu -m am \
#     --config out/am/${EXP_NAME}/config.yaml \
#     --restart_file out/am/${EXP_NAME}/ckpt02/model.pt

EXP_NAME="exp"
torchrun --nproc-per-node gpu -m project --exp_name ${EXP_NAME} --train true \
    --epochs 100 --model_type 0 --width 128 --num_layers 8 \
    --num_heads 8 --mlp_ratio 2 --num_slices 32 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-3
