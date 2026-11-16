// src/poly/hy_v3.hpp - Tencent's HY V3 with routed experts (llama.cpp architecture "hy_v3"): Hy-MT2-30B-A3B.
//
// One residual stream and attention in every layer, over the keys and values of every token so far: 32 query heads
// over 4 key heads of 128 values each, every head's queries and keys normed and then turned by the position.  The
// first layer's FFN is dense; the others' are routed experts with sigmoid gates and a selection bias, beside an
// ungated shared expert.  The picked experts' weights are made to sum to one and then scaled.  The output matrix is
// a tensor of its own.  The files so far have no block that guesses ahead.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class HyV3 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void save_layer(int, int64_t, uint8_t*) override {}   // no recurrent state: a layer has only its rows
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }
    const Names* tensor_alias(const char* name) const override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        Dev wq, wk, wv, wo, q_norm, k_norm;
        uint16_t* keys = nullptr;   ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
        // --kv q8_0 keeps those rows as ggml Q8_0 instead: n_kv * head_dim int8 codes and an fp16 scale a
        // 32-value block, in two arrays a side (store_q8)
        int8_t* k_codes = nullptr;
        uint16_t* k_scales = nullptr;
        int8_t* v_codes = nullptr;
        uint16_t* v_scales = nullptr;
        // a dense layer's FFN
        Dev ffn_gate, ffn_up, ffn_down;
        // routed experts: the bias their pick adds to the gates
        std::vector<float> probs_b;
        Dev probs_b_dev;
    };

    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void dense_ffn(Block& B, int il, int n);
    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0; }

    std::vector<Block> blocks_;    ///< one a layer of layers_
    std::vector<uint8_t> dense_;   ///< per layer: its FFN is dense (the file has no router for it)
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0;
    int n_run_ = 0;           ///< the layers this engine runs: their rows the card's rows hold
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    bool w_norm_ = true;      ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;       ///< and then scaled by this, when it is not 0
    int chunk_ = 1;           ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    size_t score_n_ = 0;       ///< the floats of score_: room for attend()'s blocks of rows too
    float* q8w_ = nullptr;     ///< --kv q8_0: a prompt batch's K and V, dequantized to fp16 (kwork | vwork)
    size_t work_halves_ = 0;   ///< the halves of each of q8w_'s two halves
};

}  // namespace strata::poly
