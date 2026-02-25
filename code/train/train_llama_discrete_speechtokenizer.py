import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
import argparse
import logging

# 添加这一行来启用 Tensor Cores 加速
torch.set_float32_matmul_precision('high') 

# layer=24


speechtokenizer_ckpt_path = "/commondocument/group2/ASRCompare/model/speechtokenizer/ckpt.dev"
speechtokenizer_config_path = "/commondocument/group2/ASRCompare/model/speechtokenizer/config.json"
# llama_ckpt_path="/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
llama_ckpt_path = "/commondocument/group2/ASRCompare/model/Meta-Llama-3.1-8B"
hubert_ckpt_path="/commondocument/group2/ASRCompare/model/hubert-large-ls960-ft"

pl.seed_everything(3407)

# model=IS(wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,wavtokenizer_config_path=wavtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path,layer=layer)

model=IS(speechtokenizer_ckpt_path=speechtokenizer_ckpt_path,speechtokenizer_config_path=speechtokenizer_config_path,hubert_ckpt_path=hubert_ckpt_path,llama_ckpt_path=llama_ckpt_path)
# model = model.float()

batchsize=8

# ckpt_path = "/commondocument/group2/ASRCompare/code/model/ckpt/wavtokenizer/epoch=99-train_loss=0.26-val_loss=0.24-ASR_wavtokenizer-3407.ckpt"
# trainset = ASRDataset(wavpth,textpth)
# valset = ASRDataset(valwavpth,valtextpth)

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/train/train-all-960.scp", "/commondocument/group2/ASRCompare/data/Librispeech/train/train-all-960.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_all.scp", "/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_all.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/train/train-clean-100.scp", "/commondocument/group2/ASRCompare/data/Librispeech/train/train-clean-100.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_clean.scp", "/commondocument/group2/ASRCompare/data/Librispeech/dev_clean/dev_clean.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/train.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/train_labels.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/iemocap_4class_data/val.scp", "/commondocument/group2/ASRCompare/data/iemocap_4class_data/valid_labels.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/train.scp", "/commondocument/group2/ASRCompare/data/GTZAN/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/GTZAN/valid.scp", "/commondocument/group2/ASRCompare/data/GTZAN/valid.txt")

trainset = ASRDataset("/commondocument/group2/ASRCompare/data/us8k/train.scp", "/commondocument/group2/ASRCompare/data/us8k/train.txt")
valset = ASRDataset("/commondocument/group2/ASRCompare/data/us8k/dev.scp", "/commondocument/group2/ASRCompare/data/us8k/dev.txt")


# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/SLURP_intent/train/train.scp", "/commondocument/group2/ASRCompare/data/SLURP_intent/train/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/SLURP_intent/devel/devel.scp", "/commondocument/group2/ASRCompare/data/SLURP_intent/devel/devel.txt")


# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_train.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/clotho/clotho_dev.scp", "/commondocument/group2/ASRCompare/data/clotho/clotho_dev.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/sdd_final_split/train.scp", "/commondocument/group2/ASRCompare/data/sdd_final_split/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/sdd_final_split/val.scp", "/commondocument/group2/ASRCompare/data/sdd_final_split/val.txt")

# trainset = ASRDataset("/commondocument/group2/ASRCompare/data/cremad_final_split/train.scp", "/commondocument/group2/ASRCompare/data/cremad_final_split/train.txt")
# valset = ASRDataset("/commondocument/group2/ASRCompare/data/cremad_final_split/val.scp", "/commondocument/group2/ASRCompare/data/cremad_final_split/val.txt")

train_sampler = RandomSampler(trainset)

train_loader = DataLoader(trainset, batch_size=batchsize, sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn, num_workers=4)

checkpoint_callback = ModelCheckpoint(
        dirpath='/commondocument/group2/ASRCompare/code/model/8B_ckpt/speechtokenizer_us8k',
        filename='{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-speechtokenizer',
        save_top_k=2,
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

early_stop_callback = EarlyStopping(
   monitor='val_loss',   # 监控验证集损失
   patience=30,          # 如果 val_loss 连续 5 个 epoch 没有改善，则停止训练
   verbose=True,         # 打印早停信息
   mode='min'            # 'min' 表示监控的指标越小越好
)

trainer = pl.Trainer(
    max_epochs=150,
    profiler=None,  # "simple"
    logger=TensorBoardLogger(name='speechtokenizer_us8k',save_dir='/commondocument/group2/ASRCompare/code/train/8B_log'),
    accelerator='gpu',
    num_nodes=1,
    devices=[4,5,6,7],
    log_every_n_steps=20,
    precision="32",
    # callbacks=[checkpoint_callback],
    callbacks=[checkpoint_callback, early_stop_callback],
    accumulate_grad_batches=4,
    strategy="ddp"
    )

trainer.fit(model, train_loader, val_loader)
# trainer.fit(model, train_loader, val_loader,ckpt_path=ckpt_path)