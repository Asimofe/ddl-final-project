CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint-rs.tar --camera realsense --dataset_root ../graspnet_dataset
# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet
