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

from transformers import WavLMModel, Wav2Vec2Processor

from torch.nn import CrossEntropyLoss
import os


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
        hubert_ckpt_path: str = None,
        wavlm_ckpt_path: str = None,
        llama_ckpt_path: str = None,
        layer: int = 24,
        ):
        super(IS, self).__init__()
        if not hasattr(self, 'audio_model'):
            # print("here!")
            self.audio_model = WavLMModel.from_pretrained(wavlm_ckpt_path)
        self.processor = Wav2Vec2Processor.from_pretrained(hubert_ckpt_path)
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
            nn.Linear(10240, 8192),
            nn.ReLU(),
            # nn.Linear(8192, 2048)      #1B
            nn.Linear(8192, 4096)
        )
        # 确保自定义层参数可训练

        for param in self.parameters():
            param.requires_grad = False

        for param in self.audio_embedding_last_Linear.parameters():
            param.requires_grad = True
        
        self.wrong_number=0
        vocab_size = len(self.text_tokenizer)
        # print("vocab_size：", vocab_size)
        self.loss_fn=CustomCrossEntropyLoss(self.text_tokenizer.eos_token_id,vocab_size)

        
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
                print(f"Loading parameter: {name}")
        
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

    def forward(self,inputs):
        audio,output_text=inputs
        # print("output_text:",output_text)
        batchsize=len(audio)
        #print(audio[0].shape[0])
        audio_len=[audio[i].shape[0]//320-1 for i in range(batchsize)]
        audio_len=[(i+9)//10 for i in audio_len]
        # # print(audio.shape)
        # for i in range(batchsize):
        #     audio[i]=audio[i][:audio_len[i]*10]

        # print("audio_len:",audio) 一致

        with torch.no_grad():
            input_values = self.processor(audio,  return_tensors="pt", sampling_rate=16000, padding=True).input_values.to(self.device)
            input_values=input_values.float()  #这里被我改成float32了
            # print("input_values",input_values)
            outputs = self.audio_model(input_values).last_hidden_state
            # print("outputs:",outputs)
            # print("outputs.shape:",outputs.shape)
            time_len=outputs.shape[1]
            padded_time_len=(time_len+9)//10*10
            # print("padded_time_len",padded_time_len)
            outputs=F.pad(outputs,(0,0,0,padded_time_len-time_len))
            # print("outputs.shape",outputs.shape)
            outputs=outputs.view(batchsize,-1,10240)

        # print("final_outputs:",outputs)
        # audio_inputs=padded_audio_input_values.contiguous().view(batchsize,padded_time_len//10,-1)
        audio_inputs=outputs.contiguous()
        # print(audio_inputs.shape)
        audio_inputs=self.audio_embedding_last_Linear(audio_inputs)
        # print("audio_input:",audio_inputs)
        audio_lengths = torch.tensor(audio_len)

        x=[]
        len_x1=[]
        bos_emb=self.bos_emb.unsqueeze(0).to(self.device).detach()

        for i in range(batchsize):
            audio_input=audio_inputs[i,:audio_lengths[i],:]
            #print(text_input_pre.shape,audio_input.shape,text_input_post.shape)
            # eos_emb=self.eos_emb.unsqueeze(0).to(self.device).detach()
            input_x=torch.cat((bos_emb,audio_input),dim=0)
            x.append(input_x)
            # print("text_lengths[i],audio_lengths[i]",text_lengths[i],audio_lengths[i])
            # print("input_x.shape[0]",input_x.shape[0])
            len_x1.append(input_x.shape[0])
        #x为一个list，保存的是每个batch的input，每一个元素代表一个text和一个audio拼接的input
        #len_x为一个list，保存的是每个batch的input的长度
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = torch.stack(list(padded_x), dim=0)
        
        texts=[text+self.text_tokenizer.eos_token for text in output_text]
        texts=self.text_tokenizer(texts,return_tensors="pt",padding="longest",truncation=True,add_special_tokens=False).to(self.device)

        # print("Texts_token",texts)
        # ---------------------------------------------PROMPT-------------------------------------------
        # prompt = "Identify the text corresponding to the speech: "   # ASR
        prompt = "Identify the emotion corresponding to the speech: "    # ER
        # prompt = "Identify the music genre corresponding to the audio: "     # music
        # prompt = "Identify the urban sound category corresponding to the audio:"     #urbansound
        # prompt = "Identify the intent corresponding to the speech: "               #IC
        # prompt = "Identify the description corresponding to the audio:"       #clotho
        # prompt = "Generate a caption for the music: "     #song describer
        
        prompt = self.text_tokenizer(prompt,return_tensors="pt",padding="longest",truncation=True,add_special_tokens=False).to(self.device)
        # ---------------------------------------------PROMPT-------------------------------------------

        # print("prompt:",prompt)
        # print("pad_token_id:", self.text_tokenizer.pad_token_id)

        targets=texts["input_ids"].masked_fill(
            texts.input_ids == self.text_tokenizer.pad_token_id, -100
        )
        # print("targets:",targets)
        with torch.no_grad():
            # print("texts[input_ids]:",texts["input_ids"])
            texts_embes=self.llama.get_input_embeddings()(texts["input_ids"])
            prompt_embes = self.llama.get_input_embeddings()(prompt["input_ids"])
        # --------------------------------------PROMPT---------------------------------------------------
        batch_size = texts_embes.shape[0]
        prompt_embes = prompt_embes.repeat_interleave(batch_size, dim=0)

        # print("shape:",x.shape,prompt_embes.shape,texts_embes.shape)
        # --------------------------------------ATTENTION------------------------------------------------
        inputs_embeds=torch.cat((x,prompt_embes,texts_embes),dim=1)  # 使用教师强制
        # inputs_embeds=x  #注意这里，去掉了教师强制
        # print(x.shape,texts_embes.shape,inputs_embeds.shape)

        attns_text=texts.attention_mask
        attns_prompt = prompt.attention_mask
        attns_prompt = prompt.attention_mask.repeat_interleave(batch_size, dim=0)
        # print("attns_text",attns_text)
        attns_audio=make_pad_mask_number(audio_lengths+1).to(x.device)
        #加2是因为bos和eos
        # --------------------------------------ATTENTION------------------------------------------------
        attns=torch.cat((attns_audio,attns_prompt,attns_text),dim=1)  # 使用教师强制
        # attns=attns_audio  # #注意这里，去掉了教师强制

        # print(attns_audio.shape,attns_text.shape,attns.shape)
        # print("attns_audio:",attns_audio)
        # print("attns_text:",attns_text)
        # print("attns:",attns)
        # print(inputs_embeds.shape,attns.shape,targets.shape)
        # print("inputs_embeds:",inputs_embeds.shape,inputs_embeds)
        outputs=self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attns,
            return_dict=True
            )
        # print(outputs.keys())

        # print("output:",outputs.shape,outputs)
        hidden=outputs[0]
        len_target=targets.shape[1]
        # 去掉教师强制
        
        # shift_logits和lable的对应:
        # 对应方式1:对齐除了bos以外的文本token
        # shift_logits = hidden[..., -len_target:-1, :].contiguous()  # 对齐文本长度

        # print("shift_logits:",shift_logits.shape,shift_logits)
        # shift_labels = targets[...,1:].contiguous()

        # print("shift_labels:",shift_labels.shape,shift_labels)
        # 对应方式2:
        shift_logits = hidden[..., -len_target-1:-1, :].contiguous()  # 从bos开始对齐
        shift_labels = targets.contiguous()
        
        # 对应方式3:
        # shift_logits = hidden.contiguous()  # 完整token列表尝试--报错,不行,其长度与target不一致

        loss_fct = self.loss_fn
        # # 使用hidden的实际特征维度而不是直接使用vocab_size
        feature_dim = hidden.size(-1)
        # # print("feature_dim: ",feature_dim)
        shift_logits = shift_logits.view(-1, feature_dim)
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)    # 这里是计算loss的输入
        
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

    def inference(self,inputs):
        self.eval()
        audio,output_text=inputs
        batchsize=len(audio)
        #print(audio[0].shape[0])
        audio_len=[audio[i].shape[0]//320-1 for i in range(batchsize)]
        audio_len=[(i+9)//10 for i in audio_len]
        # self.audio_model.eval()

        with torch.no_grad():
            input_values = self.processor(audio,  return_tensors="pt", sampling_rate=16000, padding=True).input_values.to(self.device)
            input_values=input_values.float()
            # print("inference_input_values",input_values)
            outputs=self.audio_model(input_values,output_hidden_states=True)['hidden_states']
            outputs=outputs[-1]
            # print("inference_outputs:",outputs)
            time_len=outputs.shape[1]
            padded_time_len=(time_len+9)//10*10
            # print(padded_time_len)
            outputs=F.pad(outputs,(0,0,0,padded_time_len-time_len))
            # print(outputs.shape)
            outputs=outputs.view(batchsize,-1,10240)

        # print("final_outputs:",outputs)
        # audio_inputs=padded_audio_input_values.contiguous().view(batchsize,padded_time_len//10,-1)
        audio_inputs=outputs.contiguous()

        audio_inputs=self.audio_embedding_last_Linear(audio_inputs)
        # print("audio_input:",audio_inputs)
        # print(audio_len)
        audio_lengths = torch.tensor(audio_len).to(self.device)
        
         # ---------------------------------------------PROMPT-------------------------------------------
        # prompt = "Identify the text corresponding to the speech: "
        prompt = "Identify the emotion corresponding to the speech: "    # ER
        # prompt = "Identify the music genre corresponding to the audio: "     # music
        # prompt = "Identify the urban sound category corresponding to the audio:"     #urbansound
        # prompt = "Identify the intent corresponding to the speech: "               #IC
        # prompt = "Identify the description corresponding to the audio:"       #clotho
        # prompt = "Generate a caption for the music: "     #song describer

        prompt = self.text_tokenizer(prompt,return_tensors="pt",padding="longest",truncation=True,add_special_tokens=False).to(self.device)

        with torch.no_grad():
            prompt_embes = self.llama.get_input_embeddings()(prompt["input_ids"])
        prompt_embes = prompt_embes.squeeze(0)
        # --------------------------------------PROMPT---------------------------------------------------

        x=[]
        len_x1=[]

        bos_emb=self.bos_emb.unsqueeze(0).to(self.device).detach()

        for i in range(batchsize):

            audio_input=audio_inputs[i,:audio_lengths[i],:]
            #print(text_input_pre.shape,audio_input.shape,text_input_post.shape)
            
            # print("bos,audio,prompt:",bos_emb.shape,audio_input.shape,prompt_embes.shape)
            input_x=torch.cat((bos_emb,audio_input,prompt_embes),dim=0)  # 修改点，我这里加了bos
            x.append(input_x)
            #print(text_lengths[i],audio_lengths[i])
            len_x1.append(input_x.shape[0])

        # --- 【修改重点】改为手动左填充 (Left Padding) ---
        
        # 1. 获取最大长度
        max_len = max(len_x1)
        embed_dim = x[0].shape[-1]

        # 2. 初始化全0的 Embeddings 和 Attention Mask
        # padding_value=0 对于 embedding 是合适的，前提是 mask 设置正确
        padded_x = torch.zeros((batchsize, max_len, embed_dim), dtype=x[0].dtype, device=self.device)
        attention_mask = torch.zeros((batchsize, max_len), dtype=torch.long, device=self.device)

        # 3. 填入数据（右对齐 = 左填充）
        for i, seq in enumerate(x):
            length = len_x1[i]
            # 核心逻辑：数据放在每一行的 [max_len - length : ]
            padded_x[i, max_len - length:, :] = seq
            # Mask 对应位置设为 1
            attention_mask[i, max_len - length:] = 1

        # print("padded_x (left padded):", padded_x.shape,padded_x)
        # print("attention_mask (left padded):", attention_mask)

        # --- Generate 调用 ---
        with torch.no_grad():
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
        
        # print("outputbase:", outputs)

        outputs = self.text_tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return outputs
    
