# Modified for the UniARC unified source release; see README.md.
# Source: Anachronism-N/UniARC, code/model/model_llama2_DAC_prompt.py
import lightning.pytorch as pl
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from uniarc.module.utils import make_pad_mask_number
import os
import dac
from audiotools import AudioSignal

class CustomCrossEntropyLoss(nn.Module):

    def __init__(self, tokenizer_id, vocab_size, ignore_index=-100, eos_penalty=0.1):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eos_penalty = eos_penalty
        weights = torch.ones(vocab_size)
        weights[tokenizer_id] = 1
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)
        self.tokenizer_id = tokenizer_id
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        base_loss = self.loss_fct(logits, labels)
        return base_loss

class IS(pl.LightningModule):

    def __init__(self, DAC_path: str=None, llama_ckpt_path: str=None, task_prompt: str='Identify the text corresponding to the speech: ', learning_rate: float=1e-05):
        super(IS, self).__init__()
        self.task_prompt = task_prompt
        self.learning_rate = learning_rate
        self.strict_loading = False
        self.dac = dac.DAC.load(DAC_path)
        self.text_tokenizer = AutoTokenizer.from_pretrained(llama_ckpt_path)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = '<|finetune_right_pad_id|>'
        self.llama = LlamaForCausalLM.from_pretrained(llama_ckpt_path)
        if len(self.text_tokenizer) > self.llama.config.vocab_size:
            self.llama.resize_token_embeddings(len(self.text_tokenizer))
        self.bos_emb = self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.bos_token_id))
        self.eos_emb = self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.eos_token_id))
        self.audio_embedding_last_Linear = nn.Sequential(nn.Linear(10240, 8192), nn.ReLU(), nn.Linear(8192, self.llama.config.hidden_size))
        for param in self.parameters():
            param.requires_grad = False
        for param in self.audio_embedding_last_Linear.parameters():
            param.requires_grad = True
        self.wrong_number = 0
        vocab_size = len(self.text_tokenizer)
        self.loss_fn = CustomCrossEntropyLoss(self.text_tokenizer.eos_token_id, vocab_size)

    def train(self, mode=True):
        # Frozen backbones must not activate dropout, masking, or codebook updates.
        super(IS, self).train(mode)
        self.llama.eval()
        self.dac.eval()
        return self

    def on_load_checkpoint(self, checkpoint):
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        state = checkpoint['state_dict']
        missing = trainable - state.keys()
        if missing:
            raise ValueError(f'Checkpoint is missing trainable adapter parameters: {sorted(missing)}')
        checkpoint['state_dict'] = {name: value for name, value in state.items() if name in trainable}

    def on_save_checkpoint(self, checkpoint):
        trainable_param_names = {name for name, param in self.named_parameters() if param.requires_grad}
        trainable_params = {name: param for name, param in checkpoint['state_dict'].items() if name in trainable_param_names}
        checkpoint['state_dict'] = trainable_params

    def calculate_feature_lengths(self, audio_sample_lengths):
        """
        根据音频的原始采样点数，计算输出的特征序列长度。
        """
        total_stride = 320
        feat_lengths = audio_sample_lengths // total_stride
        return feat_lengths.to(torch.long)

    def forward(self, inputs):
        padded_audios, output_text, audio_sample_lengths = inputs
        batchsize = padded_audios.shape[0]
        downsample_factor = 10
        with torch.no_grad():
            w = padded_audios.to(self.device) / 32768.0
            w = w.unsqueeze(1)
            signal = AudioSignal(w, sample_rate=16000)
            features, _, _, _, _ = self.dac.encode(signal.audio_data)
            features = features.transpose(1, 2).contiguous()
            feat_lengths = self.calculate_feature_lengths(audio_sample_lengths)
        max_feat_len = features.shape[1]
        feat_dim = features.shape[2]
        padded_time_len = (max_feat_len + downsample_factor - 1) // downsample_factor * downsample_factor
        padding_size = padded_time_len - max_feat_len
        if padding_size > 0:
            padded_features = F.pad(features, (0, 0, 0, padding_size))
        else:
            padded_features = features
        stacked_features = padded_features.view(batchsize, padded_features.size(1) // downsample_factor, feat_dim * downsample_factor)
        audio_lengths = (feat_lengths + downsample_factor - 1) // downsample_factor
        audio_inputs = self.audio_embedding_last_Linear(stacked_features)
        x = []
        bos_emb = self.bos_emb.unsqueeze(0).to(self.device).detach()
        for i in range(batchsize):
            audio_input = audio_inputs[i, :audio_lengths[i], :]
            input_x = torch.cat((bos_emb, audio_input), dim=0)
            x.append(input_x)
        padded_x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
        x = padded_x
        texts = [text + self.text_tokenizer.eos_token for text in output_text]
        texts = self.text_tokenizer(texts, return_tensors='pt', padding='longest', truncation=True, add_special_tokens=False).to(self.device)
        prompt = self.task_prompt
        prompt = self.text_tokenizer(prompt, return_tensors='pt', padding='longest', truncation=True, add_special_tokens=False).to(self.device)
        targets = texts['input_ids'].masked_fill(texts.input_ids == self.text_tokenizer.pad_token_id, -100)
        with torch.no_grad():
            texts_embes = self.llama.get_input_embeddings()(texts['input_ids'])
            prompt_embes = self.llama.get_input_embeddings()(prompt['input_ids'])
        prompt_embes = prompt_embes.repeat_interleave(batchsize, dim=0)
        inputs_embeds = torch.cat((x, prompt_embes, texts_embes), dim=1)
        attns_text = texts.attention_mask
        attns_prompt = prompt.attention_mask.repeat_interleave(batchsize, dim=0)
        attns_audio = make_pad_mask_number(audio_lengths + 1).to(x.device)
        attns = torch.cat((attns_audio, attns_prompt, attns_text), dim=1)
        # Match generate(): positions count valid tokens, including when audio
        # padding sits between the audio prefix and the text prompt.
        position_ids = attns.long().cumsum(-1) - 1
        position_ids.masked_fill_(attns == 0, 0)
        outputs = self.llama(inputs_embeds=inputs_embeds, attention_mask=attns, position_ids=position_ids, return_dict=True)
        hidden = outputs[0]
        len_target = targets.shape[1]
        shift_logits = hidden[..., -len_target - 1:-1, :].contiguous()
        shift_labels = targets.contiguous()
        loss_fct = self.loss_fn
        feature_dim = hidden.size(-1)
        shift_logits = shift_logits.view(-1, feature_dim)
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch[0]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-06)
        return optimizer

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch[0]), sync_dist=True)
        return loss

    def test_asr(self, inputs, filedir):
        y = self.inference(inputs)
        targets = inputs[1]
        print('y_pre:', y)
        print('target:', targets)
        target_dir = os.path.join(filedir, 'target.txt')
        output_dir = os.path.join(filedir, 'output.txt')
        try:
            with open(target_dir, 'a', encoding='utf-8') as f_target, open(output_dir, 'a', encoding='utf-8') as f_output:
                for pred, targ in zip(y, targets):
                    pred_cleaned = str(pred).replace('\n', ' ').replace('\r', ' ')
                    f_target.write(str(targ) + '\n')
                    f_output.write(pred_cleaned + '\n')
        except Exception as e:
            print(f'Error writing to file: {e}')
            self.wrong_number += 1

    def topk_sampling(self, logits, top_k=10, top_p=1.0, temperature=1.0):
        if temperature != 1.0:
            logits = logits / temperature
        logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return token

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
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
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def inference(self, inputs):
        self.eval()
        padded_audios, output_text, audio_sample_lengths = inputs
        batchsize = padded_audios.shape[0]
        with torch.no_grad():
            w = padded_audios.to(self.device) / 32768.0
            w = w.unsqueeze(1)
            signal = AudioSignal(w, sample_rate=16000)
            signal = self.dac.preprocess(signal.audio_data, signal.sample_rate)
            features, _, _, _, _ = self.dac.encode(signal)
            features = features.transpose(1, 2).contiguous()
            feat_lengths = self.calculate_feature_lengths(audio_sample_lengths)
            downsample_factor = 10
            max_feat_len = features.shape[1]
            feat_dim = features.shape[2]
            padded_time_len = (max_feat_len + downsample_factor - 1) // downsample_factor * downsample_factor
            padding_size = padded_time_len - max_feat_len
            if padding_size > 0:
                padded_features = F.pad(features, (0, 0, 0, padding_size))
            else:
                padded_features = features
            stacked_features = padded_features.view(batchsize, padded_features.size(1) // downsample_factor, feat_dim * downsample_factor)
            audio_lengths = (feat_lengths + downsample_factor - 1) // downsample_factor
            audio_inputs = self.audio_embedding_last_Linear(stacked_features)
            prompt = self.task_prompt
            prompt = self.text_tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
            with torch.no_grad():
                prompt_embes = self.llama.get_input_embeddings()(prompt['input_ids'])
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
            outputs = self.llama.generate(inputs_embeds=padded_x, attention_mask=attention_mask, do_sample=False, num_beams=5, max_new_tokens=200, pad_token_id=self.text_tokenizer.pad_token_id, eos_token_id=self.text_tokenizer.eos_token_id, repetition_penalty=1.15, no_repeat_ngram_size=4)
            outputs = self.text_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return outputs
