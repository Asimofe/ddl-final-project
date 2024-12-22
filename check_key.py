import torch

checkpoint = torch.load("./logs/log_rs/final_pruned_model.pth")
print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(key)

