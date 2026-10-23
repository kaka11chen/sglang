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

"""Video frame batching preserves frame order and valid projected tokens."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.models import embedding_gemma2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VisionTower(nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        self.batches = []

    def forward(self, pixels, positions):
        self.batches.append(pixels.shape[0])
        return pixels * 3, positions[..., 0] >= 0


class _Projection(nn.Module):
    def forward(self, inputs_embeds):
        return torch.cat((inputs_embeds, inputs_embeds * 2), dim=-1)


class _TextModel(nn.Module):
    device = torch.device("cpu")

    def dtype(self):
        return torch.float32


def _model(batch_size=4):
    model = embedding_gemma2.EmbeddingGemma2Model.__new__(
        embedding_gemma2.EmbeddingGemma2Model
    )
    nn.Module.__init__(model)
    model.vision_batch_size = batch_size
    model.vision_tower = _VisionTower()
    model.embed_vision = _Projection()
    model.text_config = SimpleNamespace(hidden_size=12)
    model.language_model = _TextModel()
    return model


def _frame(index, patches=6, valid=5):
    pixels = torch.arange(patches * 6, dtype=torch.float32).reshape(patches, 6)
    pixels += index * 100
    positions = torch.stack((torch.arange(patches), torch.zeros(patches)), dim=-1)
    positions[valid:] = -1
    return SimpleNamespace(feature=pixels, video_position_ids=positions)


def _expected(frames):
    tokens = [
        frame.feature[frame.video_position_ids[:, 0] >= 0] * 3 for frame in frames
    ]
    pixels = torch.cat(tokens, dim=0)
    return torch.cat((pixels, pixels * 2), dim=-1)


class TestVideoFrameBatching(CustomTestCase):
    def test_configured_cap_preserves_order_and_mask(self):
        frames = [_frame(i) for i in range(9)]
        for cap in (1, 2, 4, 8):
            with self.subTest(cap=cap):
                model = _model(cap)
                output = model.get_video_feature(frames)
                torch.testing.assert_close(output, _expected(frames), atol=0, rtol=0)
                self.assertEqual(
                    model.vision_tower.batches,
                    [min(cap, 9 - start) for start in range(0, 9, cap)],
                )

    def test_shape_change_flushes_pending_frames(self):
        frames = [_frame(i) for i in range(5)] + [
            _frame(i + 5, patches=4, valid=3) for i in range(4)
        ]
        model = _model()
        output = model.get_video_feature(frames)
        torch.testing.assert_close(output, _expected(frames), atol=0, rtol=0)
        self.assertEqual(model.vision_tower.batches, [4, 1, 4])

    def test_precomputed_features_keep_their_position(self):
        first, last = _frame(1), _frame(2)
        precomputed = torch.arange(36, dtype=torch.float32).reshape(3, 12)
        items = [first, SimpleNamespace(feature=precomputed), last]
        model = _model()
        output = model.get_video_feature(items)
        expected = torch.cat((_expected([first]), precomputed, _expected([last])))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        self.assertEqual(model.vision_tower.batches, [1, 1])

    def test_four_dimensional_input_obeys_cap(self):
        frames = [_frame(i) for i in range(6)]
        item = SimpleNamespace(
            feature=torch.stack([frame.feature for frame in frames]).reshape(
                2, 3, 6, 6
            ),
            video_position_ids=torch.stack(
                [frame.video_position_ids for frame in frames]
            ).reshape(2, 3, 6, 2),
        )
        model = _model()
        output = model.get_video_feature([item])
        torch.testing.assert_close(output, _expected(frames), atol=0, rtol=0)
        self.assertEqual(model.vision_tower.batches, [4, 2])

    def test_nested_frame_lists_preserve_order(self):
        frames = [_frame(i) for i in range(6)]
        item = SimpleNamespace(
            feature=[[frame.feature for frame in frames[:2]], frames[2].feature]
            + [frame.feature for frame in frames[3:]],
            video_position_ids=[
                [frame.video_position_ids for frame in frames[:2]],
                frames[2].video_position_ids,
            ]
            + [frame.video_position_ids for frame in frames[3:]],
        )
        model = _model()
        output = model.get_video_feature([item])
        torch.testing.assert_close(output, _expected(frames), atol=0, rtol=0)
        self.assertEqual(model.vision_tower.batches, [4, 2])

    def test_empty_and_padding_only_inputs(self):
        model = _model()
        for items in ([], [_frame(0, valid=0), _frame(1, valid=0)]):
            with self.subTest(empty=not items):
                output = model.get_video_feature(items)
                self.assertEqual(output.shape, (0, 12))
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.device, torch.device("cpu"))

    def test_missing_positions_raise(self):
        model = _model()
        with self.assertRaisesRegex(ValueError, "no matching video_position_ids"):
            model.get_video_feature([SimpleNamespace(feature=_frame(0).feature)])


class TestVideoFrameBatchConfiguration(CustomTestCase):
    def test_invalid_cap_fails_before_model_allocation(self):
        config = SimpleNamespace(text_config=SimpleNamespace())
        for cap in (0, -1):
            with (
                self.subTest(cap=cap),
                envs.SGLANG_EMBEDDING_GEMMA2_VISION_BATCH_SIZE.override(cap),
            ):
                with self.assertRaisesRegex(
                    ValueError, "VISION_BATCH_SIZE must be positive"
                ):
                    embedding_gemma2.EmbeddingGemma2Model(config)

    def test_constructor_reads_cap(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(), vision_config=None, audio_config=None
        )
        with (
            envs.SGLANG_EMBEDDING_GEMMA2_VISION_BATCH_SIZE.override(2),
            patch.object(
                embedding_gemma2, "EmbeddingGemma2TextModel", return_value=_TextModel()
            ),
        ):
            model = embedding_gemma2.EmbeddingGemma2Model(config)
        self.assertEqual(model.vision_batch_size, 2)


if __name__ == "__main__":
    unittest.main()
