import sys
sys.path.append('/commondocument/group2/ASRCompare/code')

from dataset.dataloader_continus import ASRDataset, collate_fn

from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_llama2_continus_prompt import IS   # 注意要实现不同功能需要从我改后的代码引入
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

import tqdm

layer=24

hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"
llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
ckpt_path="/commondocument/group2/ASRCompare/code/model/ckpt/hubert_ASR/epoch=12-train_loss=3956.290-val_loss=6936.900-linear_prompt_ASR-3407.ckpt"
pl.seed_everything(3407)

model=IS(hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)
model = model.float()

# print(model)


batchsize=36
trainset = ASRDataset("/commondocument/group2/ASRCompare/data/train/train-clean-100.scp", "/commondocument/group2/ASRCompare/data/train/train-clean-100.txt")
valset = ASRDataset("/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.scp", "/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/ASR_data/mrk.scp", "/commondocument/group2/ASRCompare/data/ASR_data/text.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/ASR_data/mrk.scp", "/commondocument/group2/ASRCompare/data/ASR_data/text.txt")
# We use a random sampler to shuffle the indices
train_sampler = RandomSampler(trainset)


train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)


checkpoint_callback = ModelCheckpoint(
        dirpath='/commondocument/group2/ASRCompare/code/model/ckpt/hubert_ASR',
        filename='{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-linear_prompt_ASR-3407',
        save_top_k=2,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

trainer = pl.Trainer(
    max_epochs=1,
    profiler=None,   # "simple"
    logger=TensorBoardLogger(name='hubert_ASR',save_dir='/commondocument/group2/ASRCompare/code/train/log'),
    accelerator='gpu',
    num_nodes=1,
    devices=8,
    log_every_n_steps=50,
    precision="32",
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=8,
    strategy="ddp",
    # enable_progress_bar=False,  # 禁用进度条
    # strategy='ddp_find_unused_parameters_true', # <-- 添加或修改这一行
    )
# 
# trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path)   #沿着上次训练的结果继续训练
trainer.fit(model, train_loader, val_loader)

# print("训练模型参数:")
# for n, _ in model.named_parameters():
#     print(n)


# 训练后直接推理一次看结果与训练时有无差别

# model.eval()

# # 准备测试数据
# test_set_clean=ASRDataset("/commondocument/group2/ASRCompare/data/before_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/before_data_ER/text.txt")
# dataloader_clean = DataLoader(test_set_clean, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

# # 进行推理
# fdir="/commondocument/group2/ASRCompare/code/inference/test"
# if not os.path.exists(fdir):
#     os.makedirs(fdir)
# for i,batch in tqdm.tqdm(enumerate(dataloader_clean)):
#     print("\nchoice 1: test_asr\n")
#     model.test_asr(batch,fdir)