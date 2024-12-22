import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from graspnet import GraspNet, get_loss
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--dataset_root', required=True, help='Dataset root')
    parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during evaluation [default: 1]')
    return parser.parse_args()

def move_to_device(data, device):
    """
    Recursively move all tensors in data structure to specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data

def convert_pruned_state_dict(state_dict):
    """
    Convert pruned state dict to regular state dict by combining weight_orig and weight_mask
    """
    new_state_dict = {}
    keys_to_remove = []
    
    for key in state_dict.keys():
        if key.endswith('.weight'):
            new_state_dict[key] = state_dict[key]
        elif key.endswith('.weight_orig'):
            base_key = key[:-5]
            mask_key = base_key + '_mask'
            
            if mask_key in state_dict:
                weight = state_dict[key] * state_dict[mask_key]
                new_key = base_key
                new_state_dict[new_key] = weight
                keys_to_remove.extend([key, mask_key])
        elif not key.endswith('.weight_mask'):
            new_state_dict[key] = state_dict[key]
    
    return new_state_dict

def load_model(checkpoint_path, num_view, device):
    # Initialize model
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])
    net = net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    try:
        net.load_state_dict(checkpoint)
    except RuntimeError:
        print("Converting pruned state dict to regular state dict...")
        converted_state_dict = convert_pruned_state_dict(checkpoint)
        missing, unexpected = net.load_state_dict(converted_state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
    
    net.eval()
    return net

def evaluate(net, dataloader, device):
    total_loss = 0
    score_list = []
    
    for batch_idx, batch_data_label in enumerate(dataloader):
        print(f'Evaluating batch {batch_idx + 1}/{len(dataloader)}')
        
        # Move all data to device
        batch_data_label = move_to_device(batch_data_label, device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points)
            
            # Get scores
            grasp_scores = end_points['grasp_score_pred'].cpu().numpy()
            score_list.append(np.mean(grasp_scores))
            
            total_loss += loss.item()
            
        if batch_idx % 10 == 0:
            print(f'Current Avg Loss: {total_loss/(batch_idx+1):.4f}')
            print(f'Current Avg Score: {np.mean(score_list):.4f}')
    
    # Calculate final metrics
    final_loss = total_loss / len(dataloader)
    final_score = np.mean(score_list)
    
    return final_loss, final_score

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    net = load_model(args.checkpoint_path, args.num_view, device)
    
    # Initialize dataset and dataloader
    print("Preparing dataset...")
    valid_obj_idxs, grasp_labels = load_grasp_labels(args.dataset_root)
    test_dataset = GraspNetDataset(
        args.dataset_root, 
        valid_obj_idxs, 
        grasp_labels,
        camera=args.camera, 
        split='test_seen',
        num_points=args.num_point,
        remove_outlier=True,
        augment=False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    print("Starting evaluation...")
    avg_loss, avg_score = evaluate(net, test_dataloader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Grasp Score: {avg_score:.4f}")

if __name__ == "__main__":
    main()