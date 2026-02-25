import sys
sys.path.append('/commondocument/group2/ASRCompare/code')

from dataset.dataloader_continus_wavtokenizer import ASRDataset, collate_fn

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

# 添加这一行来启用 Tensor Cores 加速
torch.set_float32_matmul_precision('high') 

# layer=24


wavtokenizer_ckpt_path = "/commondocument/group2/ASRCompare/model/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_v2.ckpt"
wavtokenizer_config_path = "/commondocument/group2/ASRCompare/model/WavTokenizer-large-speech-75token/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"

pl.seed_everything(3407)

# model=IS(wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,wavtokenizer_config_path=wavtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)

model=IS(wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,wavtokenizer_config_path=wavtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path)
# model = model.float()

batchsize=14
trainset = ASRDataset("/commondocument/group2/ASRCompare/data/MELD/meld_train.scp", "/commondocument/group2/ASRCompare/data/MELD/meld_train.txt")
valset = ASRDataset("/commondocument/group2/ASRCompare/data/MELD/meld_dev.scp", "/commondocument/group2/ASRCompare/data/MELD/meld_dev.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/train/train-clean-100.scp", "/commondocument/group2/ASRCompare/data/train/train-clean-100.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.scp", "/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/train.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/train_labels.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/val.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/valid_labels.txt")

train_sampler = RandomSampler(trainset)

train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn, num_workers=3)
val_loader = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn, num_workers=3)

checkpoint_callback = ModelCheckpoint(
        dirpath='/commondocument/group2/ASRCompare/code/model/ckpt/wavtokenizer',
        filename='{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-ER_wavtokenizer-MELD',
        save_top_k=2,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

trainer = pl.Trainer(
    max_epochs=200,
    profiler=None,  # "simple"
    logger=TensorBoardLogger(name='wavtokenizer',save_dir='/commondocument/group2/ASRCompare/code/train/log'),
    accelerator='gpu',
    num_nodes=1,
    devices=[0,1,2,3,5,6,7],
    log_every_n_steps=20,
    precision="32",
    callbacks=[checkpoint_callback],
    # accumulate_grad_batches=4,
    strategy="ddp"
    )

trainer.fit(model, train_loader, val_loader)
# trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path)