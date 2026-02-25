import sys
sys.path.append('/commondocument/group2/ASRCompare/code')

from dataset.dataloader_continus_wavtokenizer import ASRDataset, collate_fn

from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_llama2_speechtokenizer_prompt import IS
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer=24
speechtokenizer_ckpt_path = "/commondocument/group2/ASRCompare/model/speechtokenizer/ckpt.dev"
speechtokenizer_config_path = "/commondocument/group2/ASRCompare/model/speechtokenizer/config.json"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"
# llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
llama_ckpt_path = "/commondocument/group2/ASRCompare/model/Meta-Llama-3.1-8B"

pl.seed_everything(3407)
# print("0")
# model=IS(hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)
model = IS.load_from_checkpoint(
    "/commondocument/group2/ASRCompare/code/model/8B_ckpt/speechtokenizer_us/epoch=05-train_loss=0.075-val_loss=0.299-speechtokenizer.ckpt",
    speechtokenizer_ckpt_path=speechtokenizer_ckpt_path,
    speechtokenizer_config_path=speechtokenizer_config_path,
    hubert_ckpt_path=hubert_ckpt_path,
    llama_ckpt_path=llama_ckpt_path,
    map_location=device,
    strict=False  # 忽略缺少的参数
)

model = model.float()
# model = IS.load_from_checkpoint(
#     "/commondocument/group2/ASRCompare/code/model/ckpt/epoch=45-train_loss=0.006-val_loss=0.001.ckpt")
# model=model.to("cuda:0")
batchsize=4
# print("0")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/test_clean/test_clean.scp", "/commondocument/group2/ASRCompare/data/Librispeech/test_clean/test_clean.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/test_other/test_other.scp", "/commondocument/group2/ASRCompare/data/Librispeech/test_other/test_other.txt")
# test_set_other=ASRDataset("/commondocument/group2/ASRCompare/data/mrk.scp", "/commondocument/group2/ASRCompare/data/text.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/test.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/test_labels.txt")
test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/us8k/test.scp", "/commondocument/group2/ASRCompare/data/us8k/test.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/SLURP_intent/test/test.scp", "/commondocument/group2/ASRCompare/data/SLURP_intent/test/test.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/cremad_final_split/test.scp", "/commondocument/group2/ASRCompare/data/cremad_final_split/test.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/sdd_final_split/test.scp", "/commondocument/group2/ASRCompare/data/sdd_final_split/test.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/test.scp", "/commondocument/group2/ASRCompare/data/GTZAN/test.txt")
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_test.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_test_all.txt")

dataloader_clean = DataLoader(test_set_clean, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)
# dataloader_other = DataLoader(test_set_other, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

# print("1")
# state_dict = torch.load("/commondocument/group2/ASRCompare/code/model/ckpt/epoch=997-train_loss=0.00-val_loss=0.00.ckpt",map_location="cpu")
# print("2")
# model.load_state_dict(state_dict,strict=False)  # 改回False，因为audio_model参数可能被冻结未保存
# print("3")
model.eval()
# model.llama.eval()
# torch.cuda.empty_cache()


import tqdm

fdir="/commondocument/group2/ASRCompare/code/inference/8B/speechtokenizer_US"
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
