# Modified for the UniARC unified source release; see README.md.

import os
import torch
import torch.nn as nn
import numpy as np
import faiss
from transformers import AutoProcessor, HubertModel

class HubertKmeansEncoder(nn.Module):
    def __init__(self, 
                 model_path: str = "facebook/hubert-large-ls960-ft", 
                 kmeans_path: str | None = None,
                 sampling_rate: int = 16000):
        super().__init__()
        kmeans_path = kmeans_path or os.environ.get("HUBERT_KMEANS_PATH")
        if not kmeans_path or not os.path.isfile(kmeans_path):
            raise FileNotFoundError("Provide the paper centroid .npy via kmeans_path or HUBERT_KMEANS_PATH.")
        
        # Load Hubert Model
        print(f"Loading Hubert from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.sampling_rate = sampling_rate
        
        # Model config
        self.output_dim = 1024 # Centroids are 1024-dim usually, need to check if we return centroids or codes.
        # User asked for "Hubert+Kmeans", implies discrete codes or centroids. 
        # WavLM+Kmeans implementation returned *centroids* (float vectors).
        # We will assume same behavior: return centroids.
        
        # Load K-means centroids
        print(f"Loading K-means centroids from {kmeans_path}...")
        self.centroids = np.load(kmeans_path) # (K, D)
        self.k, self.d = self.centroids.shape
        print(f"Centroids shape: {self.centroids.shape}")

        if self.d != self.model.config.hidden_size:
             # Typically Hubert Large is 1024. Centroids should be 1024.
             # If mismatch, we warn.
             print(f"Warning: Model hidden size {self.model.config.hidden_size} != Centroids dim {self.d}")

        # Initialize Faiss index
        print(f"Initializing Faiss index (K={self.k}, D={self.d})...")
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.centroids)
        
        # Move centroids to torch buffer if we want to return them as tensors, 
        # BUT Faiss is CPU only (usually). 
        # So we process chunks on CPU, get indices, then lookup centroids.
        self.register_buffer("centroids_tensor", torch.from_numpy(self.centroids).float())

    def forward(self, audio: torch.Tensor, audio_attention_mask=None):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        audio_list = [a.detach().cpu().float().numpy() for a in audio]
        
        features = self.processor(
            audio_list,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        
        if audio_attention_mask is not None:
             features["attention_mask"] = audio_attention_mask

        device = next(self.model.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            outputs = self.model(**features)
        
        # Hubert output: (B, T, D)
        hidden = outputs.last_hidden_state
        
        # Quantization needs to happen on CPU via Faiss
        B, T, D = hidden.shape
        hidden_flat = hidden.view(-1, D).cpu().numpy()
        
        # Search nearest centroids
        _, indices = self.index.search(hidden_flat, 1) # indices: (B*T, 1)
        indices = indices.flatten()
        
        # Flattened indices -> Lookup centroids
        # We return the centroids as "features"
        quantized = self.centroids_tensor[indices].to(device)
        quantized = quantized.view(B, T, D)
        
        # Handle attention mask if present
        # Processor returns mask for raw audio? No, for features usually if padded.
        # But here we used simple padding.
        # If we have attention_mask in features, we should pass it.
        # Hubert downsampling is 320x (20ms stride -> 50Hz).
        # We verify attention_mask existence.
        return_mask = features.get("attention_mask", None)
        
        return quantized, return_mask

if __name__ == "__main__":
    model = HubertKmeansEncoder()
    x = torch.randn(1, 16000)
    y, mask = model(x)
    print(y.shape)
