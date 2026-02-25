import sys
sys.path.append('/commondocument/group2/ASRCompare/code')

from dataset.dataloader_continus import ASRDataset, collate_fn

from torch.utils.data import Dataset, DataLoader, RandomSampler

from model.model_llama2_wavlm_prompt import IS   # 注意要实现不同功能需要从我改后的代码引入
# from model.model_llama2_wavtokenizer_prompt import IS   # 注意要实现不同功能需要从我改后的代码引入
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger 
import lightning.pytorch as pl
import torch.optim as optim
import math

import os
# import argparses
import logging

import tqdm

# from pytorch_lightning.callbacks import Timer
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks import Timer, ModelCheckpoint, EarlyStopping

layer=24

wavlm_ckpt_path="/commondocument/group2/ASRCompare/model/wavlm/wavlm-large"
# llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"
llama_ckpt_path = "/commondocument/group2/ASRCompare/model/Meta-Llama-3.1-8B"
# ckpt_path="/commondocument/group2/ASRCompare/code/model/ckpt/hubert_ASR/epoch=49-train_loss=0.011-val_loss=0.006-linear_ASR-3407_reckpt.ckpt"
pl.seed_everything(3407)

model=IS(hubert_ckpt_path=hubert_ckpt_path, wavlm_ckpt_path=wavlm_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)
# model = model.float()

# ------------------------------    从已有ckpt继续训练    ---------------------------------
# # 加载 checkpoint 文件到 CPU 以避免设备冲突
# checkpoint = torch.load(ckpt_path, map_location='cpu') 

# # 获取 state_dict
# loaded_state_dict = checkpoint['state_dict']

# # 使用 strict=False 将权重加载到模型中
# # 这会加载 loaded_state_dict 中所有与 model 匹配的键，并忽略不匹配的键
# model.load_state_dict(loaded_state_dict, strict=False)

# ------------------------------    从已有ckpt继续训练    ---------------------------------

batchsize=8

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/MELD/meld_train.scp", "/commondocument/group2/ASRCompare/data/MELD/meld_train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/MELD/meld_dev.scp", "/commondocument/group2/ASRCompare/data/MELD/meld_dev.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/train/train-clean-100.scp", "/commondocument/group2/ASRCompare/data/train/train-clean-100.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.scp", "/commondocument/group2/ASRCompare/data/dev_clean/dev_clean.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/3label_data_ER/mrk.scp", "/commondocument/group2/ASRCompare/data/3label_data_ER/text.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/train.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/train_labels.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/val.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/valid_labels.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/train.scp", "/commondocument/group2/ASRCompare/data/GTZAN/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/valid.scp", "/commondocument/group2/ASRCompare/data/GTZAN/valid.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_train.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_dev.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_dev.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/train/train-all-960.scp", "/commondocument/group2/ASRCompare/data/Librispeech/train/train-all-960.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_all.scp", "/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_all.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/us8k/train.scp", "/commondocument/group2/ASRCompare/data/us8k/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/us8k/dev.scp", "/commondocument/group2/ASRCompare/data/us8k/dev.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/SLURP_intent/train/train.scp", "/commondocument/group2/ASRCompare/data/SLURP_intent/train/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/SLURP_intent/devel/devel.scp", "/commondocument/group2/ASRCompare/data/SLURP_intent/devel/devel.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_train.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_dev.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_dev.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/sdd_final_split/train.scp", "/commondocument/group2/ASRCompare/data/sdd_final_split/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/sdd_final_split/val.scp", "/commondocument/group2/ASRCompare/data/sdd_final_split/val.txt")

trainset = ASRDataset("/commondocument/group2/ASRCompare/data/cremad_final_split/train.scp", "/commondocument/group2/ASRCompare/data/cremad_final_split/train.txt")
valset = ASRDataset("/commondocument/group2/ASRCompare/data/cremad_final_split/val.scp", "/commondocument/group2/ASRCompare/data/cremad_final_split/val.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/train/train-clean-100.scp", "/commondocument/group2/ASRCompare/data/Librispeech/train/train-clean-100.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_clean.scp", "/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_clean.txt")

# We use a random sampler to shuffle the indices
train_sampler = RandomSampler(trainset)


train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)


checkpoint_callback = ModelCheckpoint(
        dirpath='/commondocument/group2/ASRCompare/code/model/8B_ckpt/wavlm_cremad',
        filename='{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}_continuous',
        save_top_k=2,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )
# --- 2. 创建 EarlyStopping 回调 ---
# 核心是告诉它监控哪个指标，以及“耐心”是多少
early_stop_callback = EarlyStopping(
   monitor='val_loss',   # 监控验证集损失
   patience=30,          # 如果 val_loss 连续 30 个 epoch 没有改善，则停止训练
   verbose=True,         # 打印早停信息
   mode='min'            # 'min' 表示监控的指标越小越好
)

trainer = pl.Trainer(
    max_epochs=150,
    profiler=None,   # "simple"
    logger=TensorBoardLogger(name='wavlm_cremad',save_dir='/commondocument/group2/ASRCompare/code/train/8B_log'),
    accelerator='gpu',
    num_nodes=1,
    devices=[0,1,2,3],
    log_every_n_steps=20,
    precision="bf16-mixed", 
    callbacks=[checkpoint_callback, early_stop_callback],
    # callbacks=[checkpoint_callback],
    accumulate_grad_batches=4,
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