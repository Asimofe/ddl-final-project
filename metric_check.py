import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
import torch.nn as nn
import torch.nn.utils.prune as prune

# GraspNet 모델 임포트 (경로에 따라 수정 필요)
from graspnet import GraspNet

def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
    sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()

def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
    sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # pruning이 적용된 경우
            if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                weight = module.weight_orig * module.weight_mask
            # pruning이 적용되지 않은 경우
            else:
                weight = module.weight
            num_nonzeros += (weight != 0).sum()
            num_elements += weight.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # pruning이 적용된 경우
            if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                weight = module.weight_orig * module.weight_mask
            # pruning이 적용되지 않은 경우
            else:
                weight = module.weight
            
            if count_nonzero_only:
                num_counted_elements += (weight != 0).sum()
            else:
                num_counted_elements += weight.numel()
            
            # bias도 계산에 포함
            if hasattr(module, 'bias') and module.bias is not None:
                if count_nonzero_only:
                    num_counted_elements += (module.bias != 0).sum()
                else:
                    num_counted_elements += module.bias.numel()
    return num_counted_elements

def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element (default: 32 for float32)
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

if __name__ == "__main__":
    # Pruned 모델과 일반 모델의 체크포인트 경로
    pruned_checkpoint_path = './logs/log_rs/final_pruned_model.pth'
    original_checkpoint_path = './logs/log_rs/checkpoint_2epoch.tar'
    
    # GraspNet 모델 인스턴스 생성
    model = GraspNet(input_feature_dim=0)
    
    # 원본 모델 분석
    print("\n=== 원본 모델 분석 ===")
    if os.path.exists(original_checkpoint_path):
        checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("원본 모델 가중치 로드 완료")
        
        sparsity = get_model_sparsity(model)
        total_params = get_num_parameters(model, count_nonzero_only=False)
        nonzero_params = get_num_parameters(model, count_nonzero_only=True)
        model_size_bytes = get_model_size(model, data_width=32) / 8
        
        print(f"모델 희소성: {sparsity * 100:.2f}%")
        print(f"총 파라미터: {total_params:,}")
        print(f"0이 아닌 파라미터: {nonzero_params:,}")
        print(f"모델 크기: {model_size_bytes:.0f} bytes")
    else:
        print("원본 모델 체크포인트를 찾을 수 없습니다.")
    
    # Pruned 모델 분석
    print("\n=== Pruned 모델 분석 ===")
    if os.path.exists(pruned_checkpoint_path):
        model = GraspNet(input_feature_dim=0)  # 새 모델 인스턴스
        checkpoint = torch.load(pruned_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("Pruned 모델 가중치 로드 완료")
        
        sparsity = get_model_sparsity(model)
        total_params = get_num_parameters(model, count_nonzero_only=False)
        nonzero_params = get_num_parameters(model, count_nonzero_only=True)
        model_size_bytes = get_model_size(model, data_width=32) / 8
        
        print(f"모델 희소성: {sparsity * 100:.2f}%")
        print(f"총 파라미터: {total_params:,}")
        print(f"0이 아닌 파라미터: {nonzero_params:,}")
        print(f"모델 크기: {model_size_bytes:.0f} bytes")
        
        # 압축률 계산
        print(f"\n메모리 절감: {(1 - torch.true_divide(nonzero_params, total_params)) * 100:.2f}%")
    else:
        print("Pruned 모델 체크포인트를 찾을 수 없습니다.")