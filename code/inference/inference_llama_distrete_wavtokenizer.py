import sys
sys.path.append('/commondocument/group2/ASRCompare/code')

from dataset.dataloader_continus import ASRDataset, collate_fn

from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_llama2_wavtokenizer_prompt import IS
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch.optim as optim
import math

import os
import argparse
import logging

layer=24
wavtokenizer_ckpt_path = "/commondocument/group2/ASRCompare/model/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt"
wavtokenizer_config_path = "/commondocument/group2/ASRCompare/model/WavTokenizer-large-speech-75token/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"
llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"

pl.seed_everything(3407)
# print("0")
# model=IS(hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)
model = IS.load_from_checkpoint(
    "/commondocument/group2/ASRCompare/code/model/ckpt/wavtokenizer/epoch=99-train_loss=0.26-val_loss=0.24-ASR_wavtokenizer-3407.ckpt",
    wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,
    wavtokenizer_config_path=wavtokenizer_config_path,
    hubert_ckpt_path=hubert_ckpt_path,
    llama_ckpt_path=llama_ckpt_path,
    layer=layer
)

model = model.float()
# model = IS.load_from_checkpoint(
#     "/commondocument/group2/ASRCompare/code/model/ckpt/epoch=45-train_loss=0.006-val_loss=0.001.ckpt")
model=model.to("cuda:0")
batchsize=1
# print("0")
test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/before_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/before_data_ER/text.txt")
# test_set_other=ASRDataset("/commondocument/group2/ASRCompare/data/mrk.scp", "/commondocument/group2/ASRCompare/data/text.txt")
dataloader_clean = DataLoader(test_set_clean, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
# dataloader_other = DataLoader(test_set_other, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

torch.cuda.empty_cache()
# print("1")
# state_dict = torch.load("/commondocument/group2/ASRCompare/code/model/ckpt/epoch=997-train_loss=0.00-val_loss=0.00.ckpt",map_location="cpu")
# print("2")
# model.load_state_dict(state_dict,strict=False)  # 改回False，因为audio_model参数可能被冻结未保存
# print("3")
model.eval()
# model.llama.eval()
# torch.cuda.empty_cache()


import tqdm

fdir="/commondocument/group2/ASRCompare/code/inference/test"
if not os.path.exists(fdir):
    os.makedirs(fdir)
for i,batch in tqdm.tqdm(enumerate(dataloader_clean)):
    print("\nchoice 1: test_asr\n")
    model.test_asr(batch,fdir)
    
    # print("choice 2: inference")
    # outputs = model.inference(batch)
    # # print("outputs", outputs)
    # print("y_pre:")
    # for output in outputs:
    #     print(output)
    # print("target:")
    # print(batch[1])
    
# fdir="/commondocument/group2/ASRCompare/code/inference/test"
# if not os.path.exists(fdir):
#     os.makedirs(fdir)

# for i,batch in tqdm.tqdm(enumerate(dataloader_other)):
#     model.test_asr(batch,fdir)


# print("推理模型参数:")
# for n, _ in model.named_parameters():
#     print(n)
