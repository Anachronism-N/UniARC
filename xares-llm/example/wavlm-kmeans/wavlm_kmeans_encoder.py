# Modified for the UniARC unified source release; see README.md.
import torch
import numpy as np
import faiss
from transformers import WavLMModel, AutoProcessor
import os

class WavLMKmeansEncoder(torch.nn.Module):
    def __init__(self, 
                 model_path: str = "microsoft/wavlm-large",
                 kmeans_path: str | None = None,
                 sampling_rate: int = 16000,
                 layer_idx: int = 24): # Last layer (24 for Large)
        super().__init__()
        kmeans_path = kmeans_path or os.environ.get("WAVLM_KMEANS_PATH")
        if not kmeans_path or not os.path.isfile(kmeans_path):
            raise FileNotFoundError("Provide the paper centroid .npy via kmeans_path or WAVLM_KMEANS_PATH.")
        
        # Load WavLM Model
        print(f"Loading WavLM from {model_path}...")
        self.model = WavLMModel.from_pretrained(model_path)
        self.model.eval()
        
        # Load Processor (optional, mostly for feature extraction rules if needed, but WavLM usually takes raw waveform)
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            # Fallback for models without processor config, though WavLM usually has one
            print(f"Warning: Could not load processor: {e}")
            self.processor = None

        self.sampling_rate = sampling_rate
        
        # Load K-means centroids
        print(f"Loading K-means centroids from {kmeans_path}...")
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"Centroids file not found: {kmeans_path}")
            
        self.centroids = np.load(kmeans_path) # Shape: (K, D)
        self.k = self.centroids.shape[0]
        self.d = self.centroids.shape[1]
        
        # Initialize Faiss Index
        print(f"Initializing Faiss index (K={self.k}, D={self.d})...")
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.centroids)
        
        # Output dimension matches centroid dimension
        self.output_dim = self.d
        self.layer_idx = layer_idx

    def forward(self, audio: torch.Tensor, audio_attention_mask=None):
        """
        Args:
            audio (torch.Tensor): (B, T) raw waveform at 16kHz
            audio_attention_mask (torch.Tensor): (B, T) boolean or 0/1 mask
        Returns:
            quantized_features (torch.Tensor): (B, Frames, D) - features replaced by nearest centroids
            None
        """
        device = next(self.model.parameters()).device
        
        # Ensure input is on correct device and shape
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        # WavLM expects input_values. 
        # If we have processor, use it? Or just pass audio directly as input_values?
        # Usually standard WavLM takes 'input_values' (B, T).
        
        input_values = audio.to(device)
        if audio_attention_mask is not None:
            attention_mask = audio_attention_mask.to(device)
        else:
            attention_mask = None
            
        with torch.no_grad():
            outputs = self.model(
                input_values, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Extract features from specific layer
            # hidden_states is a tuple. For WavLM-Large (24 layers), len is 25 (0=embeddings, 1-24=trans).
            # If layer_idx is 24, we want the last one (-1).
            if self.layer_idx is None or self.layer_idx >= len(outputs.hidden_states):
                 features = outputs.last_hidden_state
            else:
                 features = outputs.hidden_states[self.layer_idx]
            
            # features: (B, Frames, D)
            B, T, D = features.shape
            
            # Flatten for Faiss: (B*T, D)
            features_flat = features.reshape(-1, D).cpu().numpy()
            
            # Search nearest centroids
            # D, I = self.index.search(features_flat, 1) # I is (B*T, 1) indices
            # But we want the *centroids vector* itself as output, or the discrete codes?
            # User request said "wavlm+kmeans encoder", likely implying quantized continuous representation 
            # OR discrete codes depending on downstream. 
            # Looking at previous context (encoders for LLM), usually we output continuous features 
            # (Projector adapts D -> LLM).
            # If inputs are "quantized features" (semantic tokens), we might return the centroids.
            # Standard "Kmeans features" usually means the centroids corresponding to cluster assignment.
            
            _, indices = self.index.search(features_flat, 1) # indices: (N, 1)
            indices = indices.flatten() # (N,)
            
            # Lookup centroids
            quantized_flat = self.centroids[indices] # (N, D)
            
            # Convert back to tensor
            quantized_features = torch.from_numpy(quantized_flat).to(device)
            quantized_features = quantized_features.view(B, T, D)
            
            return quantized_features, None

if __name__ == "__main__":
    try:
        encoder = WavLMKmeansEncoder()
        
        # Create dummy audio (1 sec at 16kHz)
        audio = torch.randn(1, 16000)
        
        if torch.cuda.is_available():
            encoder = encoder.cuda()
            audio = audio.cuda()
            
        print("Running forward pass...")
        output, _ = encoder(audio)
        print(f"Output shape: {output.shape}")
        print(f"Output device: {output.device}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Identify if it's just missing paths (likely in this environment)
        if "No such file" in str(e) or "not found" in str(e):
             print("\nNote: Paths need to be valid for this to run. Please check model_path/kmeans_path.")
