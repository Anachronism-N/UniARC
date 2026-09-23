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

from transformers.configuration_utils import PretrainedConfig
from typing import Dict, Any


class XaresLLMModelConfig(PretrainedConfig):
    model_type = "xaresllmmodel"

    def __init__(
        self,
        audio_encoder_name: str | None = None,
        audio_encoder_params: Dict[str, Any] | None = None,
        decoder_type: str = "HuggingFaceTB/SmolLM2-135M",
        projector_type: str = "linear",
        lora_enabled: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: str = "all-linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_type = decoder_type
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_params = dict(audio_encoder_params or {})
        self.projector_type = projector_type
        self.lora_enabled = lora_enabled
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules


__all__ = ["XaresLLMModelConfig"]
