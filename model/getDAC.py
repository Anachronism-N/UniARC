import dac
from audiotools import AudioSignal
import torch
import librosa

# 1. 加载预训练模型 (会自动下载权重)
# 选项有: '44khz', '24khz', '16khz'
model_path = dac.utils.download(model_type="16khz") 
model = dac.DAC.load(model_path)
model.to('cuda')
model.eval()

# # 2. 加载音频
# signal = AudioSignal("/commondocument/group2/ASRCompare/data/ASR_data/19_198_000000_000000.wav")
# print("wav_signal:",signal)

# if signal.sample_rate != model.sample_rate:
#     signal.resample(model.sample_rate)

path = "/commondocument/group2/ASRCompare/data/rough_data/Ses01F_impro01_F000.wav"
target_sr = 16000

# 1. 加载并直接重采样 (librosa 默认会将音频归一化到 -1.0 到 1.0 之间的 float32)
# sr=None 表示保持原样，sr=16000 表示自动重采样
wav_data, _ = librosa.load(path, sr=target_sr)


# 2. 转换为 PyTorch Tensor，并增加维度 [Batch, Channel, Time] -> [1, 1, T]
w = torch.from_numpy(wav_data).unsqueeze(0).unsqueeze(0)

print("w:",w.shape,w)

signal = AudioSignal(w, sample_rate=target_sr)

signal.to(model.device)
print("signal:",signal)
print("signal.audio_data:",signal.audio_data)
# 3. 预处理 (重采样和归一化)
signal = model.preprocess(signal.audio_data, signal.sample_rate)

print("signal_after:",signal)
print("signal.data:",signal.data)
# 4. 编码 (将音频变为离散的 Codec Tokens)
z, codes, latents, commitment_loss, codebook_indices = model.encode(signal)
# codebook_indices 就是你可能需要的离散 Token

print("feature:",z,z.shape)  #[batchsize,1024,T]
# print("codes:",codes,codes.shape)
# print("latents:",latents,latents.shape)

# 5. 解码 (从 Tokens 还原波形)
# y = model.decode(z)

# # 6. 保存结果
# y.write("reconstructed.wav")