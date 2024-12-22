""" Training routine for GraspNet baseline model with Pruning support. """

import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune  # [ADDED] Pruning 라이브러리
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from graspnet import GraspNet, get_loss
from pytorch_utils import BNMomentumScheduler
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels

import wandb

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--prune_amount', type=float, default=0.0, help='Amount of weights to prune [default: 0.0]')
cfgs = parser.parse_args()

# wandb initialize
wandb.init(
    project="GraspNet-Training",
    name=f"GraspNet_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "learning_rate": cfgs.learning_rate,
        "batch_size": cfgs.batch_size,
        "max_epoch": cfgs.max_epoch,
        "weight_decay": cfgs.weight_decay,
        "dataset_root": cfgs.dataset_root,
        "camera": cfgs.camera,
        "prune_amount": cfgs.prune_amount
    }
)

# Global configurations
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Dataset and Dataloader
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, remove_outlier=True, augment=True)
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
               cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04]).to(device)
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Pruning functions
def apply_pruning(model, prune_amount):
    """
    Apply pruning to all Conv and Linear layers in the model.
    """
    if prune_amount > 0:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=prune_amount)
                log_string(f"Pruning applied to {name} with amount {prune_amount:.2f}")


def remove_pruning(model):
    """
    Remove pruning mask and make pruning permanent.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')
            log_string(f"Pruning mask removed from {name}")

def move_to_device(data, device):
    """
    Recursively move data (tensors or nested lists of tensors) to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data  # 그대로 반환 (처리가 필요 없는 경우)

# Training and evaluation routines
def train_one_epoch():
    global EPOCH_CNT
    net.train()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        optimizer.zero_grad()

        # 데이터 전체를 재귀적으로 GPU로 이동
        batch_data_label = move_to_device(batch_data_label, device)

        # Forward pass
        end_points = net(batch_data_label)

        # Compute loss and backward pass
        loss, _ = get_loss(end_points)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            log_string(f"Epoch {EPOCH_CNT} - Batch {batch_idx + 1}: Loss {loss.item():.4f}")


def evaluate_one_epoch():
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            # 데이터 전체를 재귀적으로 GPU로 이동
            batch_data_label = move_to_device(batch_data_label, device)

            # Forward pass
            end_points = net(batch_data_label)
            loss, _ = get_loss(end_points)
            total_loss += loss.item()
    avg_loss = total_loss / len(TEST_DATALOADER)
    log_string(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss



def train():
    global EPOCH_CNT
    # Apply pruning
    apply_pruning(net, cfgs.prune_amount)

    # Train and evaluate
    for epoch in range(cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string(f"**** Epoch {epoch + 1}/{cfgs.max_epoch} ****")
        train_one_epoch()
        val_loss = evaluate_one_epoch()

    # Remove pruning mask after training
    remove_pruning(net)

    # Save final model
    final_model_path = os.path.join(cfgs.log_dir, 'final_pruned_model.pth')
    torch.save(net.state_dict(), final_model_path)
    log_string(f"Final pruned model saved to {final_model_path}")


if __name__ == "__main__":
    train()
