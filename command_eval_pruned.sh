python evaluate_pruned.py \
    --checkpoint_path ./logs/log_rs/final_pruned_model.pth \
    --dataset_root ../graspnet_dataset \
    --camera realsense \
    --num_point 20000 \
    --num_view 300
