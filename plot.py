import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

import matplotlib.pyplot as plt
import torch
from graspnet import GraspNet
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 필수: 디스플레이 없이 이미지 저장 가능하게 함

def plot_weight_distribution(model, title, output_path, bins=256, count_nonzero_only=False):
   fig, axes = plt.subplots(3,3, figsize=(10, 6))
   axes = axes.ravel()
   plot_index = 0
   
   for name, module in model.named_modules():
       if isinstance(module, (nn.Conv2d, nn.Linear)):
           ax = axes[plot_index]
           # pruning이 적용된 경우와 아닌 경우 처리
           if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
               weight = module.weight_orig * module.weight_mask
           else:
               weight = module.weight
               
           if count_nonzero_only:
               weight_cpu = weight.detach().view(-1).cpu()
               weight_cpu = weight_cpu[weight_cpu != 0].view(-1)
               ax.hist(weight_cpu, bins=bins, density=True,
                       color = 'blue', alpha = 0.5)
           else:
               ax.hist(weight.detach().view(-1).cpu(), bins=bins, density=True,
                       color = 'blue', alpha = 0.5)
           ax.set_xlabel(name)
           ax.set_ylabel('density')
           plot_index += 1
           if plot_index >= 9:  # 최대 9개의 subplot만 표시
               break
               
   fig.suptitle(f'Histogram of Weights - {title}')
   fig.tight_layout()
   fig.subplots_adjust(top=0.925)
   
   # 이미지 파일로 저장
   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   plt.close()
   print(f"그래프가 {output_path}에 저장되었습니다.")

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
       print("✅ 원본 모델 가중치 로드 완료")
       plot_weight_distribution(model, "Original Model", 
                              "weight_plots/original_weights.png")
       plot_weight_distribution(model, "Original Model (Non-zero weights only)", 
                              "weight_plots/original_nonzero_weights.png", 
                              count_nonzero_only=True)
   else:
       print("❌ 원본 모델 체크포인트를 찾을 수 없습니다.")
   
   # Pruned 모델 가중치 분포 시각화
   if os.path.exists(pruned_checkpoint_path):
       model = GraspNet(input_feature_dim=0)
       checkpoint = torch.load(pruned_checkpoint_path, map_location='cpu')
       model.load_state_dict(checkpoint, strict=False)
       print("✅ Pruned 모델 가중치 로드 완료")
       plot_weight_distribution(model, "Pruned Model", 
                              "weight_plots/pruned_weights.png")
       plot_weight_distribution(model, "Pruned Model (Non-zero weights only)", 
                              "weight_plots/pruned_nonzero_weights.png", 
                              count_nonzero_only=True)
   else:
       print("❌ Pruned 모델 체크포인트를 찾을 수 없습니다.")

if __name__ == "__main__":
   main()