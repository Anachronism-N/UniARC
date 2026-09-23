# Modified for the UniARC unified source release; see README.md.
# Copyright 2025 Horizon Team, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType

from xares_llm.audio_encoder_checker import check_audio_encoder
from xares_llm.modeling_audiollm.configuration_xaresllm import XaresLLMModelConfig
from xares_llm.utils import attr_from_module, attr_from_py_path


class XaresLLMMLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class XaresLLMQFormerProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_queries=64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, output_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.ln_q = nn.LayerNorm(output_dim)
        self.ln_k = nn.LayerNorm(output_dim)
        self.ln_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        B = x.shape[0]
        k = self.input_proj(x)
        k = self.ln_k(k)
        q = self.query.expand(B, -1, -1)
        q = self.ln_q(q)
        out, _ = self.cross_attn(query=q, key=k, value=k)
        return self.ln_out(out)

class XaresLLMModel(PreTrainedModel, nn.Module):
    config_class = XaresLLMModelConfig

    def __init__(self, config: XaresLLMModelConfig) -> None:
        super().__init__(config)
        self.config = config
        if Path(self.config.audio_encoder_name).is_file():
            audio_encoder = attr_from_py_path(self.config.audio_encoder_name, endswith="Encoder")(
                **self.config.audio_encoder_params
            )
        else:
            audio_encoder = attr_from_module(self.config.audio_encoder_name)(**self.config.audio_encoder_params)
        try:
            audio_encoder_parameters = list(audio_encoder.parameters())
            if len(audio_encoder_parameters) > 0:
                device_type = audio_encoder_parameters[0].device.type
                if device_type != "meta":  # When using .from_pretrained, device is meta
                    check_audio_encoder(audio_encoder)
        except Exception as e:
            logger.exception(e)
            raise
        self.audio_encoder = audio_encoder
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad_(False)

        decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_type,
            attn_implementation="sdpa",
        )

        # add to eliminate warnings: Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
        if decoder.config.pad_token_id is None:
            print('In XaresLLMModel: setting pad_token_id to eos_token_id for decoder')
            decoder.config.pad_token_id = decoder.config.eos_token_id
        if decoder.generation_config.pad_token_id is None:
            print('In XaresLLMModel: setting pad_token_id to eos_token_id for decoder.generation_config')
            decoder.generation_config.pad_token_id = decoder.config.eos_token_id


        if config.lora_enabled:
            peft_config = LoraConfig(
                target_modules=config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
            )
            self.decoder = get_peft_model(decoder, peft_config)
            self.decoder.print_trainable_parameters()
        else:
            self.decoder = decoder
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False

        if config.projector_type == "mlp":
            self.audio_projector = XaresLLMMLPProjector(self.audio_encoder.output_dim, self.decoder.config.hidden_size)
        elif config.projector_type == "qformer":
            self.audio_projector = XaresLLMQFormerProjector(self.audio_encoder.output_dim, self.decoder.config.hidden_size)
        else:
            self.audio_projector = nn.Linear(self.audio_encoder.output_dim, self.decoder.config.hidden_size)

    def merge_and_unload(self):
        self.decoder = self.decoder.merge_and_unload()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.decoder.gradient_checkpointing_disable()

    @property
    def keys_to_ignore_on_save(self):
        return [k for k in self.state_dict().keys() if k.startswith("audio_encoder")]

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration as e:
            logger.error("Rerun the script with 'accelerate launch -m xares_llm.run'")
            raise e

    def _prepare_multimodal_inputs(self, audio, audio_attention_mask, input_ids, attention_mask, labels=None):
        audio = audio.to(self.device)
        audio_attention_mask = audio_attention_mask.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        final_audio_attention_mask = None
        with torch.no_grad():
            audio_feature, final_audio_attention_mask = self.audio_encoder(audio, audio_attention_mask)
            audio_feature = audio_feature.to(self.device)  # returned tensor might be on cpu
        audio_feature = self.audio_projector(audio_feature)
        if final_audio_attention_mask is None:
            final_audio_attention_mask = torch.ones(*audio_feature.shape[:2], device=attention_mask.device)
        elif audio_feature.shape[1] != final_audio_attention_mask.shape[1]:
             # Length changed (e.g. Q-Former). Create new mask.
             final_audio_attention_mask = torch.ones(audio_feature.shape[0], audio_feature.shape[1], device=audio_feature.device)
        # An error occurs if .get_input_embeddings() is used with self.input_embeds = ...
        input_embeds = self.decoder.get_input_embeddings()(input_ids)  # Int -> Float

        # concatenate all data: [AUDIO, TEXT]
        input_embeds = torch.cat((audio_feature, input_embeds), dim=1)
        zero_audio_targets = torch.full(
            audio_feature.shape[:2], device=audio_feature.device, dtype=torch.int, fill_value=-100
        )
        if labels is not None:
            labels = torch.cat((zero_audio_targets, labels), dim=1)
        else:
            labels = None
        attention_mask = torch.cat(
            (final_audio_attention_mask, attention_mask),
            dim=1,
        )
        return input_embeds, attention_mask, labels

    def forward(self, audio, audio_attention_mask, input_ids, attention_mask, labels, **kwargs):
        input_embeds, attention_mask, labels = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels
        )
        return self.decoder(input_ids=None, inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask)

    @torch.no_grad()
    def generate(self, audio, audio_attention_mask, input_ids, attention_mask, **gen_kwargs):
        input_embeds, attention_mask, _ = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels=None
        )
        return self.decoder.generate(
            input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask, **gen_kwargs
        )
