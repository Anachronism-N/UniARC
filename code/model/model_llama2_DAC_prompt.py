import lightning.pytorch as pl
from typing import Tuple, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModelForCausalLM
from transformers import LlamaForCausalLM
from module.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from module.layers import ConvNorm, LinearNorm
from module.embedding import TextEmbedding,AudioEmbedding,TokenEmbedding
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from module.utils import make_pad_mask_number, Transpose

from transformers import HubertModel, Wav2Vec2Processor

from torch.nn import CrossEntropyLoss
import os
import numpy as np

import dac
from audiotools import AudioSignal


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, tokenizer_id,vocab_size,ignore_index=-100, eos_penalty=0.1):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eos_penalty = eos_penalty
        weights = torch.ones(vocab_size)
        weights[tokenizer_id] = 1  # 给 EOS 赋予 3 倍的权重（具体数值需调整）
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=weights)
        self.tokenizer_id=tokenizer_id
        self.ignore_index = ignore_index

        # ----------------------------------训练过程中解码代码------------------------------------------------
        # llama_ckpt_path = "/commondocument/group2/ASRCompare/model/Llama-3.2-1B"
        # self.text_tokenizer = AutoTokenizer.from_pretrained(llama_ckpt_path)  #只是为了查看输出，查看之后就注释掉
        # ----------------------------------训练过程中解码代码------------------------------------------------

    def forward(self, logits, labels):
        # # ----------------------------------训练过程中解码代码------------------------------------------------
        # # 尝试转成文本
        # predicted_tokens = torch.argmax(logits, dim=-1)
        # predicted_tokens_list = predicted_tokens.tolist()  # 转成 list
        # # print("predicted_tokens_list:",predicted_tokens_list)
        # predicted_text = self.text_tokenizer.decode(
        #     predicted_tokens_list,
        #     # skip_special_tokens=True  # 去掉 BOS/EOS/PAD 等特殊 token
        # )
        # print("----------------------------------")
        # print("predicted_text:", predicted_text)
        # print(labels)
        # # ----------------------------------训练过程中解码代码------------------------------------------------
        base_loss = self.loss_fct(logits, labels)

        return base_loss

class IS(pl.LightningModule):
    def __init__(self,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 10,
        DAC_path: str = None,
        llama_ckpt_path: str = None,
        # layer: int = 24,
        ):
        super(IS, self).__init__()
        self.dac = dac.DAC.load(DAC_path)
        self.text_tokenizer = AutoTokenizer.from_pretrained(llama_ckpt_path)
        # tokenizer.pad_token = "<|finetune_right_pad_id|>"  在special_tokens.map.json中找到的
        # 确保设置了pad_token
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = "<|finetune_right_pad_id|>"
        
        # self.llama=LlamaForCausalLM.from_pretrained(llama_ckpt_path,torch_dtype=torch.bfloat16)
        self.llama=LlamaForCausalLM.from_pretrained(llama_ckpt_path) # 换成32,省得后面转换
        # 检查是否需要更新嵌入层大小（如果添加了新的特殊标记）
        if len(self.text_tokenizer) > self.llama.config.vocab_size:
            self.llama.resize_token_embeddings(len(self.text_tokenizer))
            
        self.bos_emb=self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.bos_token_id))
        self.eos_emb=self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.eos_token_id))

        # ---------------------------------训练参数设定-----------------------------------
        
        # 定义并确保这个层的参数是可训练的
        self.audio_embedding_last_Linear = nn.Sequential(
            nn.Linear(10240, 8192),  #这个参数尚存疑512
            nn.ReLU(),
            # nn.Linear(8192, 2048)
            nn.Linear(8192, 4096)
        )
        # 确保自定义层参数可训练

        for param in self.parameters():
            param.requires_grad = False

        for param in self.audio_embedding_last_Linear.parameters():
            param.requires_grad = True

        # for param in self.audio_model.parameters():
        #     param.requires_grad = False

        # for param in self.llama.parameters():
        #     param.requires_grad = True   # 想要冻结应该用False，此处仅为测试！
        
        self.wrong_number=0
        vocab_size = len(self.text_tokenizer)
        self.loss_fn=CustomCrossEntropyLoss(self.text_tokenizer.eos_token_id,vocab_size)

        # ==============================测试=======================================
        # print("="*50)
        # print("Running stride test to find the correct total_stride...")
        # self.wavtokenizer.eval() # 切换到评估模式
        # with torch.no_grad():
        #     # 创建一个假的音频输入：batch_size=1, 长度为3秒的音频 (3 * 16000 = 48000个采样点)
        #     # 使用一个较长的音频可以避免边界效应带来的误差
        #     dummy_input = torch.randn(1, 48000) 
            
        #     # 将它通过您在forward函数中使用的相同预处理流程
        #     dummy_input = dummy_input.to(self.device) # 假设模型已在GPU上
        #     dummy_input = convert_audio(dummy_input, 16000, 24000, 1)
        #     bandwidth_id = torch.tensor([0], device=self.device)
            
        #     # 获取特征输出
        #     features, _ = self.wavtokenizer.encode_infer(dummy_input, bandwidth_id=bandwidth_id)
            
        #     # features 的形状是 [batch_size, time_steps, feature_dim]
        #     # 我们需要的就是中间的 time_steps
        #     input_length = 48000
        #     output_feature_length = features.shape[1]
            
        #     print("features.shape: ", features.shape)
        #     calculated_stride = input_length / output_feature_length
            
        #     print(f"Input audio length: {input_length}")
        #     print(f"Output feature length: {output_feature_length}")
        #     print(f"Calculated Stride (Input / Output): {calculated_stride}")
        #     print("Please use the integer part of this value as your `total_stride`.")
        # print("="*50)

        # print("Initial sum of audio_model.encoder.layers[0].attention.k_proj.weight:", self.audio_model.encoder.layers[0].attention.k_proj.weight.sum())

    def on_load_checkpoint(self, checkpoint):
        # 加载checkpoint时，只更新可训练的参数
        loaded_state_dict = checkpoint['state_dict']
        
        # 打印检查点中包含的参数数量
        print(f"Loaded checkpoint contains {len(loaded_state_dict)} parameters")
        
        # 只加载模型中存在且可训练的参数
        model_state_dict = self.state_dict()
        trainable_params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        print(f"Model has {len(trainable_params)} trainable parameters")
        
        # 过滤要加载的参数
        filtered_state_dict = {}
        for name, param in loaded_state_dict.items():
            if name in model_state_dict and model_state_dict[name].requires_grad:
                filtered_state_dict[name] = param
                # print(f"Loading parameter: {name}")
        
        print(f"Will load {len(filtered_state_dict)} parameters from checkpoint")
        
        # 使用strict=False允许只加载部分参数
        self.load_state_dict(filtered_state_dict, strict=False)
        
        print("Checkpoint loaded successfully!")

    def on_save_checkpoint(self, checkpoint):
        # 1. 获取模型中所有可训练参数的名称
        trainable_param_names = {name for name, param in self.named_parameters() if param.requires_grad}
        # print("Trainable parameter names:", trainable_param_names)
        # print("Trainable parameters count:", len(trainable_param_names))
        
        # 2. 基于这些名称过滤checkpoint中的参数
        trainable_params = {name: param for name, param in checkpoint['state_dict'].items() if name in trainable_param_names}
        # print("trainable_params:", list(trainable_params.keys()))
        # print("trainable_params count:", len(trainable_params))
        
        # 3. 更新checkpoint只保留可训练参数
        checkpoint['state_dict'] = trainable_params
        # 打印保存检查点时一个特定权重的总和
        # print("On saving checkpoint, sum of audio_model.encoder.layers[0].attention.k_proj.weight:", self.audio_model.encoder.layers[0].attention.k_proj.weight.sum())

    def calculate_feature_lengths(self, audio_sample_lengths):
        """
        根据音频的原始采样点数，计算输出的特征序列长度。
        """
        total_stride = 320

        feat_lengths = audio_sample_lengths // total_stride

        # 确保返回的是整数张量
        return feat_lengths.to(torch.long)

    def forward(self, inputs):
        # 步骤 1: 接收批处理好的数据
        padded_audios, output_text, audio_sample_lengths = inputs
        batchsize = padded_audios.shape[0]
        downsample_factor = 10 
        # print(padded_audios.abs().max())

        # 步骤 2: 批处理特征提取
        with torch.no_grad():   
            w = padded_audios.to(self.device) / 32768.0
            # 这里的 '1' 就代表了单声道 (mono channel)
            w = w.unsqueeze(1)
            # w = w.float()
            # print("w:",w.shape,w)
            signal = AudioSignal(w,sample_rate=16000)
            # print("signal:",signal)
            # print("signal.audio_data:",signal.audio_data)
            # signal = self.dac.preprocess(signal.audio_data, signal.sample_rate)
            # print("signal_after:",signal)
            features,_, _, _, _ = self.dac.encode(signal.audio_data)
            # print("feature:",features,features.shape)

            features = features.transpose(1, 2).contiguous()
            # print("feature_trans:",features,features.shape)
            # 使用辅助函数计算每个样本的真实特征长度 (结果是一个张量)
            feat_lengths = self.calculate_feature_lengths(audio_sample_lengths)
            # print("feat_lengths:",feat_lengths)

        # 步骤 3: 特征拼接与降采样 (完全向量化)
        max_feat_len = features.shape[1]
        feat_dim = features.shape[2]
        
        # 3.1 确保时间维度长度能被 downsample_factor 整除
        padded_time_len = (max_feat_len + downsample_factor - 1) // downsample_factor * downsample_factor
        padding_size = padded_time_len - max_feat_len
        if padding_size > 0:
            padded_features = F.pad(features, (0, 0, 0, padding_size))
        else:
            padded_features = features

        # 3.2 使用 view() 进行拼接
        stacked_features = padded_features.view(
            batchsize, 
            padded_features.size(1) // downsample_factor, 
            feat_dim * downsample_factor
        )

        # 步骤 4: 高效计算下采样后的新长度
        audio_lengths = (feat_lengths + downsample_factor - 1) // downsample_factor
        # print("audio_lengths:",audio_lengths)
        # 步骤 5: 通过线性层
        audio_inputs = self.audio_embedding_last_Linear(stacked_features)

        # 步骤 6: 为LLM准备输入序列 (此处的循环是处理变长序列的正确方法)
        x = []
        bos_emb = self.bos_emb.unsqueeze(0).to(self.device).detach()

        for i in range(batchsize):
            # 使用下采样后的 audio_lengths 进行切片，得到每个样本的有效部分
            audio_input = audio_inputs[i, :audio_lengths[i], :]
            input_x = torch.cat((bos_emb, audio_input), dim=0)
            x.append(input_x)

        # 将变长序列的列表重新填充为规整的批处理张量
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = padded_x 

        # 步骤 7: 后续流程 (保持不变，完全正确)
        texts = [text + self.text_tokenizer.eos_token for text in output_text]
        texts = self.text_tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, add_special_tokens=False).to(self.device)

        # prompt = "Identify the text corresponding to the speech: "
        # prompt = "Identify the emotion corresponding to the speech: "    # ER
        # prompt = "Identify the music genre corresponding to the audio: "     # music
        # prompt = "Identify the description corresponding to the audio:"       #clotho
        # prompt = "Identify the urban sound category corresponding to the audio:"     #urbansound
        # prompt = "Identify the intent corresponding to the speech: "               #IC
        # prompt = "Identify the description corresponding to the audio:"       #clotho
        prompt = "Generate a caption for the music: "     #song describer
        
        prompt = self.text_tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True, add_special_tokens=False).to(self.device)

        targets = texts["input_ids"].masked_fill(
            texts.input_ids == self.text_tokenizer.pad_token_id, -100
        )
        
        with torch.no_grad():
            texts_embes = self.llama.get_input_embeddings()(texts["input_ids"])
            prompt_embes = self.llama.get_input_embeddings()(prompt["input_ids"])

        prompt_embes = prompt_embes.repeat_interleave(batchsize, dim=0)

        inputs_embeds = torch.cat((x, prompt_embes, texts_embes), dim=1)

        attns_text = texts.attention_mask
        attns_prompt = prompt.attention_mask.repeat_interleave(batchsize, dim=0)
        
        # 使用正确的、最终计算出的 audio_lengths 创建 attention mask
        # +1 是因为我们在前面加了一个 bos_emb
        attns_audio = make_pad_mask_number(audio_lengths + 1).to(x.device)
        
        attns = torch.cat((attns_audio, attns_prompt, attns_text), dim=1)

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attns,
            return_dict=True
        )
        
        hidden = outputs[0]
        len_target = targets.shape[1]
        
        shift_logits = hidden[..., -len_target-1:-1, :].contiguous()
        shift_labels = targets.contiguous()
        
        loss_fct = self.loss_fn
        feature_dim = hidden.size(-1)
        shift_logits = shift_logits.view(-1, feature_dim)
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)
        
        return loss
    


    def training_step(self,batch,batch_idx):
        # print("train")
        loss=self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer

    def validation_step(self,batch,batch_idx):
        # print("validation")
        loss=self(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True,batch_size=len(batch[0]),sync_dist=True)
        return loss

    def test_asr(self, inputs, filedir):
        # 推理
        # y 是一个列表，例如 ['prediction_text_1', 'prediction_text_2']
        y = self.inference(inputs)
        
        # target 也是一个列表，例如 ['target_text_1', 'target_text_2']
        # 从你的日志看，这里被错误地加载成了情感标签 ['happy', 'neutral']
        targets = inputs[1]

        # 打印调试，这能帮你发现问题
        print("y_pre:", y)
        print("target:", targets)

        target_dir = os.path.join(filedir, "target.txt")
        output_dir = os.path.join(filedir, "output.txt")

        try:
            with open(target_dir, "a", encoding='utf-8') as f_target, \
                open(output_dir, "a", encoding='utf-8') as f_output:
                
                # 使用 zip 同时遍历 预测值(pred) 和 真实值(targ)
                for pred, targ in zip(y, targets):
                    # 【核心修改】
                    # 1. 将 pred 内部的换行符替换为空格，确保一个预测只占一行
                    pred_cleaned = str(pred).replace('\n', ' ').replace('\r', ' ')
                    
                    # 2. 写入文件，并在末尾手动添加一个换行符作为行分隔
                    f_target.write(str(targ) + "\n")
                    f_output.write(pred_cleaned + "\n")

        except Exception as e:
            print(f"Error writing to file: {e}")
            # 最好也记录一下 self.wrong_number += 1
            self.wrong_number += 1

    def topk_sampling(self,logits, top_k=10, top_p=1.0, temperature=1.0):
        # temperature: (`optional`) float
        #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        # top_k: (`optional`) int
        #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
        # top_p: (`optional`) float
        #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            logits = logits / temperature
        # Top-p/top-k filtering
        logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        # Sample
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return token
        
    def top_k_top_p_filtering(
        self,logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(
                max(top_k, min_tokens_to_keep), logits.size(-1)
            )  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        # pred_ids = logits.argmax(-1) 
        # print("pred_ids:",tokenizer.batch_decode(pred_ids))

        return logits


    def inference(self, inputs):
        self.eval()
        
        # 1. 正确地解包来自 DataLoader 的三个元素
        padded_audios, output_text, audio_sample_lengths = inputs
        batchsize = padded_audios.shape[0]
        
        # ----------------------------------------------------------------------
        # 2. 【核心重构】复用和 forward 函数完全相同的批处理前端逻辑
        # ----------------------------------------------------------------------
        with torch.no_grad():
            w = padded_audios.to(self.device) / 32768.0
            
            # --- “三明治”包装法，确保函数安全调用 ---
            w = w.unsqueeze(1)
            signal = AudioSignal(w,sample_rate=16000)
            signal = self.dac.preprocess(signal.audio_data, signal.sample_rate)

            features,_, _, _, _ = self.dac.encode(signal)
            # print("feature:",features,features.shape)

            features = features.transpose(1, 2).contiguous()
            
            # 使用辅助函数计算每个样本的真实特征长度 (结果是一个张量)
            feat_lengths = self.calculate_feature_lengths(audio_sample_lengths)


            downsample_factor = 10
            
            # 特征拼接与降采样 (完全向量化)
            max_feat_len = features.shape[1]
            feat_dim = features.shape[2]
            padded_time_len = (max_feat_len + downsample_factor - 1) // downsample_factor * downsample_factor
            padding_size = padded_time_len - max_feat_len
            if padding_size > 0:
                padded_features = F.pad(features, (0, 0, 0, padding_size))
            else:
                padded_features = features

            stacked_features = padded_features.view(
                batchsize, 
                padded_features.size(1) // downsample_factor, 
                feat_dim * downsample_factor
            )

            audio_lengths = (feat_lengths + downsample_factor - 1) // downsample_factor
            audio_inputs = self.audio_embedding_last_Linear(stacked_features)
            
            # ----------------------------------------------------------------------
            # 3. 后续为 Llama.generate 准备输入的逻辑 (这部分保持不变)
            # ----------------------------------------------------------------------
            # prompt = "Identify the text corresponding to the speech: "
            # prompt = "Identify the emotion corresponding to the speech: "    # ER
            # prompt = "Identify the music genre corresponding to the audio: "     # music
            # prompt = "Identify the description corresponding to the audio:"       #clotho
            # prompt = "Identify the urban sound category corresponding to the audio:"     #urbansound
            # prompt = "Identify the intent corresponding to the speech: "               #IC
            # prompt = "Identify the description corresponding to the audio:"       #clotho
            prompt = "Generate a caption for the music: "     #song describer

            prompt = self.text_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

            with torch.no_grad():
                prompt_embes = self.llama.get_input_embeddings()(prompt["input_ids"])
            prompt_embes = prompt_embes.squeeze(0)

            x = []
            len_x1 = []
            bos_emb = self.bos_emb.unsqueeze(0).to(self.device).detach()

            for i in range(batchsize):
                audio_input = audio_inputs[i, :audio_lengths[i], :]
                input_x = torch.cat((bos_emb, audio_input, prompt_embes), dim=0)
                x.append(input_x)
                len_x1.append(input_x.shape[0])

            max_len = max(len_x1)
            embed_dim = x[0].shape[-1]

            padded_x = torch.zeros((batchsize, max_len, embed_dim), dtype=x[0].dtype, device=self.device)
            attention_mask = torch.zeros((batchsize, max_len), dtype=torch.long, device=self.device)

            for i, seq in enumerate(x):
                length = len_x1[i]
                padded_x[i, max_len - length:, :] = seq
                attention_mask[i, max_len - length:] = 1

            outputs = self.llama.generate(              
                inputs_embeds=padded_x,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=5,
                max_new_tokens=200,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4
            )
        
            outputs = self.text_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        return outputs