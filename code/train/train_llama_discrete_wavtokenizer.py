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

# layer=24

# whisper_ckpt_path="path/to/whisper_ckpt"

wavtokenizer_ckpt_path = "/commondocument/group2/ASRCompare/model/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt"
wavtokenizer_config_path = "/commondocument/group2/ASRCompare/model/WavTokenizer-large-speech-75token/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"

pl.seed_everything(3407)

# model=IS(wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,wavtokenizer_config_path=wavtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)

model=IS(wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,wavtokenizer_config_path=wavtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path)
model = model.float()

batchsize=2
wavpth="/commondocument/group2/ASRCompare/data/before_data_ER/mrk.scp"
textpth="/commondocument/group2/ASRCompare/data/before_data_ER/text.txt"

valwavpth="/commondocument/group2/ASRCompare/data/before_data_ER/mrk.scp"
valtextpth="/commondocument/group2/ASRCompare/data/before_data_ER/text.txt"


ckpt_path = "/commondocument/group2/ASRCompare/code/model/ckpt/wavtokenizer/epoch=99-train_loss=0.26-val_loss=0.24-ASR_wavtokenizer-3407.ckpt"
trainset = ASRDataset(wavpth,textpth)
valset = ASRDataset(valwavpth,valtextpth)

train_sampler = RandomSampler(trainset)

train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)

checkpoint_callback = ModelCheckpoint(
        dirpath='/commondocument/group2/ASRCompare/code/model/ckpt/wavtokenizer',
        filename='{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}-ASR_wavtokenizer-3407',
        save_top_k=2,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

trainer = pl.Trainer(
    max_epochs=100,
    profiler=None,  # "simple"
    logger=TensorBoardLogger(name='wavtokenizer',save_dir='/commondocument/group2/ASRCompare/code/train/log'),
    accelerator='gpu',
    num_nodes=1,
    devices=1,
    log_every_n_steps=50,
    precision="32",
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=4,
    strategy="ddp"
    )

# trainer.fit(model, train_loader, val_loader)
trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path)