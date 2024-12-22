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

def load_model(checkpoint_path, num_view, device):
    """
    Load regular (non-pruned) model from checkpoint
    """
    # Initialize model
    net = GraspNet(
        input_feature_dim=0, 
        num_view=num_view, 
        num_angle=12, 
        num_depth=4,
        cylinder_radius=0.05, 
        hmin=-0.02, 
        hmax_list=[0.01,0.02,0.03,0.04]
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # First load to CPU
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove any pruning-related keys from state_dict
    clean_state_dict = {}
    for k, v in state_dict.items():
        if not any(x in k for x in ['_orig', '_mask']):
            clean_state_dict[k] = v
    
    # Load the cleaned state dict
    net.load_state_dict(clean_state_dict, strict=False)
    
    # Move model to device after loading weights
    net = net.to(device)
    
    # Double check all parameters are on the correct device
    for param in net.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("Loaded checkpoint weights")
    
    net.eval()
    return net

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoint-rs.tar', help='Model checkpoint path')
    parser.add_argument('--dataset_root', required=True, help='Dataset root')
    parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during evaluation [default: 1]')
    args = parser.parse_args()
    
    # Explicitly set to use GPU 1
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
    print(f"Using device: {device} (GPU 1)")
    
    # Load model
    print("Loading model from checkpoint:", args.checkpoint_path)
    net = load_model(args.checkpoint_path, args.num_view, device)
    
    # Verify model is on correct device
    print(f"Verifying model device...")
    for name, param in net.named_parameters():
        if param.device != device:
            print(f"Warning: Parameter {name} is on {param.device}, moving to {device}")
            param.data = param.data.to(device)
    
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
    
    # Save results to file
    result_file = 'evaluation_results_regular.txt'
    with open(result_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Average Grasp Score: {avg_score:.4f}\n")
    print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    main()