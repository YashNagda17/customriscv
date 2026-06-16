"""Shared code-generation contract helpers.

These helpers define the C helper-function contract that both the code
prompts and verifier use. Keeping the required helper signatures in one
place prevents the LLM header prompt and deterministic verifier from
drifting apart.
"""

from __future__ import annotations

from ir import IRGraph, IROpType


HELPER_SIGNATURES_BY_OP: dict[str, tuple[str, ...]] = {
    IROpType.CONV2D: (
        "void conv2d(const float* in, const float* weight, const float* bias, float* out, int IC, int OC, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.LINEAR: (
        "void linear(const float* in, const float* weight, const float* bias, float* out, int in_features, int out_features);",
    ),
    IROpType.RELU: ("void relu(float* data, int size);",),
    IROpType.SILU: ("void silu(float* data, int size);",),
    IROpType.RMSNORM: (
        "void rmsnorm(const float* in, const float* weight, float* out, int size, float eps);",
    ),
    IROpType.LAYERNORM: (
        "void layernorm(const float* in, const float* weight, const float* bias, float* out, int size, float eps);",
    ),
    IROpType.SOFTMAX: ("void softmax(float* data, int size);",),
    IROpType.MATMUL: (
        "void matmul(const float* A, const float* B, float* C, int M, int K, int N);",
    ),
    IROpType.EMBEDDING: (
        "void embedding(const float* table, int token_id, float* out, int dim);",
    ),
    IROpType.ROTARY_EMBEDDING: (
        "void rope(float* q, float* k, int head_dim, int pos, int num_heads);",
    ),
    IROpType.ATTENTION: (
        "void attention(const float* x, const float* wq, const float* wk, const float* wv, const float* wo, float* out, int seq_len, int dim, int num_heads, int head_dim);",
    ),
    IROpType.BATCHNORM: (
        "void batchnorm2d(const float* in, const float* weight, const float* bias, float* out, int C, int H, int W, float eps);",
    ),
    IROpType.MAXPOOL2D: (
        "void maxpool2d(const float* in, float* out, int C, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.AVGPOOL2D: (
        "void avgpool2d(const float* in, float* out, int C, int H, int W, int KH, int KW, int stride, int pad);",
    ),
    IROpType.FLATTEN: (
        "void flatten(const float* in, float* out, int size);",
    ),
    IROpType.ADD: (
        "void add_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.SUB: (
        "void sub_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.MUL: (
        "void mul_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.DIV: (
        "void div_tensors(const float* a, const float* b, float* out, int size);",
    ),
    IROpType.SQRT: (
        "void sqrt_tensor(const float* in, float* out, int size);",
    ),
    IROpType.MEAN: (
        "void mean_tensor(const float* in, float* out, int outer, int reduce, int inner);",
    ),
    IROpType.RESHAPE: (
        "void reshape_copy(const float* in, float* out, int size);",
    ),
    IROpType.TRANSPOSE: (
        "void transpose2d(const float* in, float* out, int rows, int cols);",
    ),
    IROpType.SPLIT: (
        "void split_tensor(const float* in, float* out0, float* out1, float* out2, int part_size);",
    ),
    IROpType.CONCAT: (
        "void concat_tensors(const float* a, const float* b, float* out, int a_size, int b_size);",
    ),
    IROpType.QUANTIZE: (
        "void quantize_f32_to_i8(const float* in, signed char* out, int size, float scale);",
    ),
    IROpType.DEQUANTIZE: (
        "void dequantize_i8_to_f32(const signed char* in, float* out, int size, float scale);",
    ),
}


def helper_name_from_signature(signature: str) -> str:
    """Return the C function name from a simple helper prototype."""
    before_paren = signature.split("(", 1)[0].strip()
    return before_paren.split()[-1]


def required_helper_signatures(ir_dict: dict) -> list[str]:
    """Return required helper prototypes for all non-I/O operations in IR order."""
    ir_graph = IRGraph.from_dict(ir_dict)
    seen: set[str] = set()
    signatures: list[str] = []
    for node in ir_graph.topological_order():
        for signature in HELPER_SIGNATURES_BY_OP.get(node.op, ()):  # no helper for I/O/dropout
            if signature not in seen:
                seen.add(signature)
                signatures.append(signature)
    return signatures
