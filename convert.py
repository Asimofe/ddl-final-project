import torch

def convert_pruned_checkpoint(checkpoint_path, output_path):
    """
    Convert a pruned checkpoint by replacing 'weight_orig' with 'weight' and removing 'weight_mask'.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    converted_checkpoint = {}

    for key, value in checkpoint.items():
        if "weight_orig" in key:
            # Replace "weight_orig" with "weight"
            new_key = key.replace("weight_orig", "weight")
            converted_checkpoint[new_key] = value
        elif "weight_mask" in key:
            # Skip "weight_mask"
            continue
        else:
            # Copy other keys as-is
            converted_checkpoint[key] = value

    # Save the converted checkpoint
    torch.save(converted_checkpoint, output_path)
    print(f"Converted checkpoint saved to {output_path}")


if __name__ == "__main__":
    # 원본 체크포인트와 변환된 체크포인트 경로
    original_checkpoint = "./logs/log_rs/final_pruned_model.pth"
    converted_checkpoint = "./logs/log_rs/final_pruned_model_converted.pth"

    # 체크포인트 변환
    convert_pruned_checkpoint(original_checkpoint, converted_checkpoint)
