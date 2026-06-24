import torch

MODEL_PATH = "/home/indra/eq_team/earthquake-classification/results/runs/20260617_201329_CNN5s/checkpoints/best_cnn_5s_dynamic_quantized.pt"

ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

print("Checkpoint keys:")
print(ckpt.keys())

print("\nClasses:")
print(ckpt["classes"])

print("\nState dict keys:")
for k in list(ckpt["model_state_dict"].keys())[:80]:
    print(k, ckpt["model_state_dict"][k].shape if hasattr(ckpt["model_state_dict"][k], "shape") else type(ckpt["model_state_dict"][k]))