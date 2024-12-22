# apply pruning 
CUDA_VISIBLE_DEVICES=0 python pruning_train.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root ../graspnet_dataset --max_epoch 2 --prune_amount 0.2

