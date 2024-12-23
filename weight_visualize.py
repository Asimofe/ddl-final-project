import os
import sys
import math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

import matplotlib.pyplot as plt
import torch
import numpy as np
from graspnet import GraspNet
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

def count_plotting_layers(model):
    """모델에서 플롯팅이 필요한 레이어의 수를 계산"""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
            count += 1
    return count

def create_subplot_layout(total_layers):
    """레이어 수에 따른 최적의 subplot 레이아웃 계산"""
    if total_layers <= 1:
        return 1, 1
    
    # 최적의 행/열 수 계산
    cols = math.ceil(math.sqrt(total_layers))
    rows = math.ceil(total_layers / cols)
    return rows, cols

def plot_weight_distribution(model, title, output_path, bins=256, count_nonzero_only=False):
    # 총 레이어 수 계산
    total_layers = count_plotting_layers(model)
    rows, cols = create_subplot_layout(total_layers)
    
    # 적절한 figure 크기 계산 (레이어 수에 비례)
    fig_width = min(20, cols * 4)  # 최대 너비 제한
    fig_height = min(20, rows * 3)  # 최대 높이 제한
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if total_layers == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    plot_index = 0
    layer_stats = {}  # 레이어별 통계 정보 저장
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
            ax = axes[plot_index]
            
            # 가중치 추출
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                    weight = module.weight_orig * module.weight_mask
                else:
                    weight = module.weight
                
                # BatchNorm의 경우 running_mean과 running_var도 표시
                running_mean = module.running_mean
                running_var = module.running_var
            else:
                if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                    weight = module.weight_orig * module.weight_mask
                else:
                    weight = module.weight
            
            # 가중치 데이터 준비
            weight_cpu = weight.detach().cpu()
            if count_nonzero_only:
                weight_cpu = weight_cpu[weight_cpu != 0]
            
            # 히스토그램 플롯
            ax.hist(weight_cpu.view(-1).numpy(), bins=bins, density=True,
                   color='blue', alpha=0.5, label='Weights')
            
            # BatchNorm 레이어의 경우 추가 정보 표시
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                ax.hist(running_mean.cpu().numpy(), bins=bins//4, density=True,
                       color='red', alpha=0.5, label='Running Mean')
                ax.hist(running_var.cpu().numpy(), bins=bins//4, density=True,
                       color='green', alpha=0.5, label='Running Var')
                ax.legend(fontsize='x-small')
            
            # 레이어 정보 표시
            ax.set_title(f'{name}\n{module.__class__.__name__}', fontsize=8)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            
            # 통계 정보 수집
            layer_stats[name] = {
                'mean': float(weight_cpu.mean()),
                'std': float(weight_cpu.std()),
                'min': float(weight_cpu.min()),
                'max': float(weight_cpu.max()),
                'zeros_pct': float((weight_cpu == 0).sum().float() / weight_cpu.numel() * 100)
            }
            
            # 통계 정보를 그래프에 추가
            stats_text = f'μ={layer_stats[name]["mean"]:.2e}\nσ={layer_stats[name]["std"]:.2e}\n'
            stats_text += f'zeros={layer_stats[name]["zeros_pct"]:.1f}%'
            ax.text(0.95, 0.95, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=6,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plot_index += 1
    
    # 사용하지 않는 subplot 제거
    for idx in range(plot_index, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f'Histogram of Weights - {title}')
    fig.tight_layout()
    
    # 이미지 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    stats_file = output_path.rsplit('.', 1)[0] + '_stats.txt'
    with open(stats_file, 'w') as f:
        f.write(f"Weight Distribution Statistics - {title}\n")
        f.write("=" * 50 + "\n\n")
        for name, stats in layer_stats.items():
            f.write(f"Layer: {name}\n")
            f.write("-" * 30 + "\n")
            for stat_name, value in stats.items():
                f.write(f"{stat_name}: {value:.6f}\n")
            f.write("\n")
    
    print(f"그래프가 {output_path}에 저장되었습니다.")
    print(f"통계 정보가 {stats_file}에 저장되었습니다.")

def main():
    # 체크포인트 경로
    original_checkpoint_path = './logs/log_rs/checkpoint-rs.tar'
    pruned_checkpoint_path = './logs/log_rs/final_pruned_model.pth'
    
    # 결과 저장할 디렉토리 생성
    os.makedirs('weight_plots', exist_ok=True)
    
    # 원본 모델 가중치 분포 시각화
    if os.path.exists(original_checkpoint_path):
        model = GraspNet(input_feature_dim=0)
        checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("원본 모델 가중치 로드 완료")
        
        plot_weight_distribution(model, "Original Model", 
                               "weight_plots/original_weights1.png")
        plot_weight_distribution(model, "Original Model (Non-zero weights only)", 
                               "weight_plots/original_nonzero_weights1.png", 
                               count_nonzero_only=True)
    else:
        print("원본 모델 체크포인트를 찾을 수 없습니다.")
    
    # Pruned 모델 가중치 분포 시각화
    if os.path.exists(pruned_checkpoint_path):
        model = GraspNet(input_feature_dim=0)
        checkpoint = torch.load(pruned_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("Pruned 모델 가중치 로드 완료")
        
        plot_weight_distribution(model, "Pruned Model", 
                               "weight_plots/pruned_weights1.png")
        plot_weight_distribution(model, "Pruned Model (Non-zero weights only)", 
                               "weight_plots/pruned_nonzero_weights1.png", 
                               count_nonzero_only=True)
    else:
        print("Pruned 모델 체크포인트를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()