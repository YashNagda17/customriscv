from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.code_generator as code_generator


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        prompt = messages[-1].content
        if "STEP 2 TASK: CREATE model.h" in prompt:
            return _FakeResponse(
                '```c model.h\n'
                '#pragma once\n'
                '#include "weights.h"\n'
                'void model_inference(const float* input, float* output);\n'
                '```'
            )
        if "STEP 3 TASK: IMPLEMENT model.c" in prompt:
            assert 'STEP 2 model.h CONTRACT TO IMPLEMENT' in prompt
            assert '#include "weights.h"' in prompt
            return _FakeResponse(
                '```c model.c\n'
                '#include "model.h"\n'
                'void model_inference(const float* input, float* output) {\n'
                '    output[0] = input[0];\n'
                '}\n'
                '```'
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:200]}")


def _minimal_state() -> dict:
    return {
        "ir_graph": {
            "model_name": "unit_model",
            "nodes": [
                {
                    "id": "input",
                    "op": "TENSOR_INPUT",
                    "inputs": [],
                    "shape": [1],
                },
                {
                    "id": "output",
                    "op": "TENSOR_OUTPUT",
                    "inputs": ["input"],
                    "shape": [],
                },
            ],
            "input_shapes": {"input": [1]},
            "weight_metadata": {},
        },
        "weights_metadata": {},
        "weight_precision": "f32",
        "weight_mode": "embedded",
        "verification_attempts": 0,
        "optimization_iteration": 0,
    }


def test_extract_c_artifact_requires_exactly_one_named_block():
    assert (
        code_generator._extract_c_artifact(
            '```c model.h\n#pragma once\n```', "model.h"
        )
        == "#pragma once"
    )

    with pytest.raises(ValueError, match="Expected exactly one fenced code block"):
        code_generator._extract_c_artifact("#pragma once", "model.h")

    with pytest.raises(ValueError, match="Expected exactly one fenced code block"):
        code_generator._extract_c_artifact(
            '```c model.h\n#pragma once\n```\n```c model.h\n#pragma once\n```',
            "model.h",
        )


def test_generate_code_writes_model_header_and_implementation(monkeypatch, tmp_path):
    monkeypatch.setattr(code_generator, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(code_generator, "ChatOpenAI", _FakeChatOpenAI)

    result = code_generator.generate_code(_minimal_state())

    model_h_path = Path(result["model_header_path"])
    model_c_path = Path(result["code_path"])
    weights_h_path = Path(result["header_path"])

    assert model_h_path == tmp_path / "model.h"
    assert model_c_path == tmp_path / "model.c"
    assert weights_h_path == tmp_path / "weights.h"
    assert model_h_path.read_text(encoding="utf-8").startswith("#pragma once")
    assert '#include "model.h"' in model_c_path.read_text(encoding="utf-8")
    assert result["generated_model_header"] == model_h_path.read_text(encoding="utf-8")
    assert result["generated_code"] == model_c_path.read_text(encoding="utf-8")
