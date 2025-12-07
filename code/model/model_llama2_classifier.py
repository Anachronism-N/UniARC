import lightning.pytorch as pl
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, LlamaForCausalLM, HubertModel, Wav2Vec2Processor

class ISClassifier(pl.LightningModule):
    def __init__(self, hubert_ckpt_path: str, llama_ckpt_path: str, layer: int = 24):
        super().__init__()
        self.audio_model = HubertModel.from_pretrained(hubert_ckpt_path)
        self.processor = Wav2Vec2Processor.from_pretrained(hubert_ckpt_path)
        self.text_tokenizer = AutoTokenizer.from_pretrained(llama_ckpt_path)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.llama = LlamaForCausalLM.from_pretrained(llama_ckpt_path)
        if len(self.text_tokenizer) > self.llama.config.vocab_size:
            self.llama.resize_token_embeddings(len(self.text_tokenizer))

        self.bos_emb = self.llama.get_input_embeddings()(torch.tensor(self.text_tokenizer.bos_token_id))
        self.audio_embedding_last_Linear = nn.Sequential(
            nn.Linear(10240, 8192),
            nn.ReLU(),
            nn.Linear(8192, 2048)
        )
        for p in self.parameters():
            p.requires_grad = False
        for p in self.audio_embedding_last_Linear.parameters():
            p.requires_grad = True

    def _encode_audio(self, audio_list: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            input_values = self.processor(audio_list, return_tensors="pt", sampling_rate=16000, padding=True).input_values.to(self.device)
            input_values = input_values.float()
            hs = self.audio_model(input_values, output_hidden_states=True)['hidden_states'][-1]
            t = hs.shape[1]
            t_pad = (t + 9) // 10 * 10
            hs = F.pad(hs, (0, 0, 0, t_pad - t))
            hs = hs.view(len(audio_list), -1, 10240)
        x = self.audio_embedding_last_Linear(hs.contiguous())
        lengths = []
        for i in range(len(audio_list)):
            l = audio_list[i].shape[0] // 320 - 1
            l = (l + 9) // 10
            lengths.append(l)
        lengths = torch.tensor(lengths, device=self.device)
        seqs = []
        for i in range(len(audio_list)):
            bos = self.bos_emb.unsqueeze(0).to(self.device)
            seqs.append(torch.cat((bos, x[i, :lengths[i], :]), dim=0))
        padded = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=0)
        return padded, lengths

    def inference(self, inputs):
        audio, output_text = inputs
        self.eval()
        x, audio_lengths = self._encode_audio(audio)
        preds = []
        pad_id = self.text_tokenizer.pad_token_id
        for i in range(len(audio)):
            candidates = list({t.strip() for t in output_text})
            tokens = self.text_tokenizer(
                [c + self.text_tokenizer.eos_token for c in candidates],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                add_special_tokens=True
            ).to(self.device)
            with torch.no_grad():
                cand_embes = self.llama.get_input_embeddings()(tokens["input_ids"])  # [N, L, D]
            x_i = x[i].unsqueeze(0).repeat(len(candidates), 1, 1)  # [N, T, D]
            attn_audio_i = torch.ones_like(x_i[:, :, 0], dtype=torch.long)  # [N, T]
            inputs_embeds = torch.cat((x_i, cand_embes), dim=1)  # [N, T+L, D]
            attn = torch.cat((attn_audio_i, tokens.attention_mask), dim=1)  # [N, T+L]
            with torch.no_grad():
                out = self.llama(inputs_embeds=inputs_embeds, attention_mask=attn, return_dict=True)
            logits = out.logits if hasattr(out, "logits") else out[0]
            len_target = tokens["input_ids"].shape[1]
            shift_logits = logits[..., -len_target-1:-1, :].contiguous()
            shift_labels = tokens["input_ids"].contiguous()
            mask = (shift_labels != pad_id)
            logprobs = F.log_softmax(shift_logits, dim=-1)
            token_lp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            token_lp = token_lp.masked_fill(~mask, 0.0)
            score = token_lp.sum(dim=1)
            best_idx = torch.argmax(score).item()
            preds.append(candidates[best_idx])
        return preds