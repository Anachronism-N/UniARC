import sys
sys.path.append('/commondocument/group2/ASRCompare/code')
from model.model_llama2_continus import IS
import torch

ckpt_path="/commondocument/group2/ASRCompare/code/model/ckpt/epoch=19-train_loss=4.02-val_loss=3.97.ckpt"

print(f"Inspecting checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("Keys in state_dict:")
    audio_model_keys = [key for key in state_dict.keys() if key.startswith('audio_model.')]
    if audio_model_keys:
        print("Found audio_model keys in checkpoint:")
        for key in audio_model_keys:
            print(key)
    else:
        print("No audio_model keys found in checkpoint.")
else:
    print("No 'state_dict' found in checkpoint.")