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
from lightning.pytorch.callbacks import Timer, ModelCheckpoint, EarlyStopping
import os
import argparse
import logging

# 添加这一行来启用 Tensor Cores 加速
torch.set_float32_matmul_precision('high') 

# layer=24


xcodec_ckpt_path="/commondocument/group2/ASRCompare/model/Xcodec2/epoch3D1400000.ckpt"
llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"

pl.seed_everything(3407)

# model=IS(wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,wavtokenizer_config_path=wavtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)

model=IS(xcodec_ckpt_path=xcodec_ckpt_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path)
# model = model.float()

batchsize=8
# wavpth="/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp"
# textpth="/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt"

# valwavpth="/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp"
# valtextpth="/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt"

# wavpth="/commondocument/group2/ASRCompare/data/train/train-clean-100.scp"
# textpth="/commondocument/group2/ASRCompare/data/train/train-clean-100.txt"

# valwavpth="/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.scp"
# valtextpth="/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.txt"

# wavpth="/commondocument/group2/ASRCompare/data/iemocap_4class_data/train.scp"
# textpth="/commondocument/group2/ASRCompare/data/iemocap_4class_data/train_labels.txt"

# valwavpth="/commondocument/group2/ASRCompare/data/iemocap_4class_data/val.scp"
# valtextpth="/commondocument/group2/ASRCompare/data/iemocap_4class_data/valid_labels.txt"

# ckpt_path = "/commondocument/group2/ASRCompare/code/model/ckpt/wavtokenizer/epoch=99-train_loss=0.26-val_loss=0.24-ASR_wavtokenizer-3407.ckpt"
# trainset = ASRDataset(wavpth,textpth)
# valset = ASRDataset(valwavpth,valtextpth)

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/train/train-all-960.scp", "/commondocument/group2/ASRCompare/data/Librispeech/train/train-all-960.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_all.scp", "/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_all.txt")

trainset = ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/train.scp", "/commondocument/group2/ASRCompare/data/GTZAN/train.txt")
valset = ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/valid.scp", "/commondocument/group2/ASRCompare/data/GTZAN/valid.txt")

train_sampler = RandomSampler(trainset)

train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn, num_workers=3)
val_loader = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn, num_workers=3)

checkpoint_callback = ModelCheckpoint(
        dirpath='/commondocument/group2/ASRCompare/code/model/ckpt/xcodec_ASR',
        filename='{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-xcodec_music',
        save_top_k=2,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

early_stop_callback = EarlyStopping(
   monitor='val_loss',   # 监控验证集损失
   patience=5,          # 如果 val_loss 连续 5 个 epoch 没有改善，则停止训练
   verbose=True,         # 打印早停信息
   mode='min'            # 'min' 表示监控的指标越小越好
)

trainer = pl.Trainer(
    max_epochs=150,
    profiler=None,  # "simple"
    logger=TensorBoardLogger(name='xcodec_music',save_dir='/commondocument/group2/ASRCompare/code/train/log'),
    accelerator='gpu',
    num_nodes=1,
    devices=[0],
    log_every_n_steps=20,
    precision="32",
    # callbacks=[checkpoint_callback],
    callbacks=[checkpoint_callback, early_stop_callback],
    accumulate_grad_batches=4,
    strategy="ddp"
    )

trainer.fit(model, train_loader, val_loader)
# trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path)