# Copyright 2026 SGLang Team
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

"""Compare native Gemma 4 audio positions and masks with Transformers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioRelPositionalEncoding,
    sliding_window_mask_function,
)

from sglang.srt.models import gemma4_audio
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestAudioPositionPrecision(CustomTestCase):
    def test_matches_hf_under_model_default_dtype(self):
        def linear(input_size, output_size, **kwargs):
            return torch.nn.Linear(input_size, output_size, bias=False)

        original_dtype = torch.get_default_dtype()
        for chunk, dtype in (
            (chunk, dtype)
            for chunk in (12, 32)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ):
            with self.subTest(chunk=chunk, dtype=dtype):
                config = Gemma4AudioConfig(
                    hidden_size=1024,
                    num_attention_heads=8,
                    attention_chunk_size=chunk,
                    attention_context_left=chunk + 1,
                    attention_context_right=0,
                )
                try:
                    # HF initializes frequencies before loading model dtype.
                    torch.set_default_dtype(torch.float32)
                    reference = Gemma4AudioRelPositionalEncoding(config).to(dtype)
                    torch.set_default_dtype(dtype)
                    with (
                        patch.object(
                            gemma4_audio,
                            "get_parallel",
                            return_value=SimpleNamespace(attn_tp_size=1),
                        ),
                        patch.object(
                            gemma4_audio, "ColumnParallelLinear", side_effect=linear
                        ),
                    ):
                        actual = gemma4_audio.Gemma4AudioRelativePositionEmbedding(
                            config
                        )
                finally:
                    torch.set_default_dtype(original_dtype)

                positions = torch.arange(chunk, -1, -1).unsqueeze(0)
                output = actual._get_timing_signal_1d_pos(positions, torch.float32).to(
                    dtype
                )
                # HF derives its position range from the configured context.
                expected = reference(torch.zeros(1, 17, 1024, dtype=dtype))
                torch.testing.assert_close(output, expected, atol=0, rtol=0)


class TestAudioWindow(CustomTestCase):
    def test_mask_matches_hf(self):
        for chunk, left, right in (
            (12, 13, 0),
            (12, 13, 3),
            (8, 8, 0),
            (4, 2, 1),
            (4, 1, 3),
        ):
            with self.subTest(chunk=chunk, left=left, right=right):
                config = Gemma4AudioConfig(
                    attention_chunk_size=chunk,
                    attention_context_left=left,
                    attention_context_right=right,
                )
                # Exercise the encoder constructor's actual mask computation.
                # Weight allocation and distributed linear setup are unrelated.
                with (
                    patch.object(
                        gemma4_audio,
                        "Gemma4AudioSubSampleConvProjection",
                        return_value=torch.nn.Identity(),
                    ),
                    patch.object(
                        gemma4_audio, "make_layers", return_value=torch.nn.ModuleList()
                    ),
                    patch.object(
                        gemma4_audio,
                        "RowParallelLinear",
                        return_value=torch.nn.Identity(),
                    ),
                ):
                    encoder = gemma4_audio.Gemma4AudioEncoder(config)
                past = left - 1
                query = past + torch.arange(chunk)[:, None]
                key = torch.arange(chunk + past + right)[None, :]
                expected = sliding_window_mask_function((past, right))(0, 0, query, key)
                torch.testing.assert_close(
                    encoder.causal_valid_mask, expected, atol=0, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
