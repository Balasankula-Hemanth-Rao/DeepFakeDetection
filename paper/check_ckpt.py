import torch, sys
ckpt = torch.load(
    r'e:\major project\DeepFakeDetection\model-service\checkpoints\best_model.pth',
    map_location='cpu', weights_only=False
)
sd = ckpt['model_state_dict']
layers = list(sd.keys())
print("=== FIRST 20 LAYERS ===")
for l in layers[:20]: print(" ", l, sd[l].shape)
print("\n=== LAST 10 LAYERS ===")
for l in layers[-10:]: print(" ", l, sd[l].shape)
print("\n=== METADATA ===")
print(ckpt['metadata'])
