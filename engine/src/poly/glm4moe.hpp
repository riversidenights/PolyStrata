// src/poly/glm4moe.hpp - Z.ai's GLM-4.5 family with routed experts (llama.cpp architecture "glm4moe"): GLM-4.5-Air
// is 106B-A12B.
//
// One residual stream and attention in every layer, over the keys and values of every token so far: 96 query heads
// over 8 key heads of 128 values each in Air.  The queries, keys and values have a bias each; GLM-4.5 (not Air)
// norms the queries and keys of a head too.  Half of a head's values turn by the position (NeoX pairing).  The
// first layer's FFN is dense; the others' are 128 routed experts with sigmoid gates, a selection bias and weights
// made to sum to one, beside an ungated shared expert.  The output matrix is a tensor of its own.  The block after
// the model's that guesses ahead has a token table and an output of its own in Unsloth's files, which this engine
// does not run (load_shape()).
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

class Glm4Moe final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return n_head_ * head_dim_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void save_layer(int, int64_t, uint8_t*) override {}   // no state beside the caches' rows
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        Dev wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm;
        uint16_t* keys = nullptr;   ///< rows of n_kv * head_dim, fp16: max_context of them
        uint16_t* vals = nullptr;
        int8_t* k_codes = nullptr;  ///< --kv q8_0: a row is n_kv * head_dim codes and n_kv * head_dim / 32 scales
        uint16_t* k_scales = nullptr;
        int8_t* v_codes = nullptr;
        uint16_t* v_scales = nullptr;
        Dev ffn_gate, ffn_up, ffn_down;   ///< a dense layer's FFN
        std::vector<float> probs_b;       ///< the bias the routed experts' pick adds to the gates (zeros without one)
        Dev probs_b_dev;
    };

    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void turn(const Block& B, const float* x, const Dev& bias, const Dev& norm, float* y, int heads, int64_t pos,
              int rows, const long long* at);
    void store(const float* x, const Block& B, int64_t pos, int n, bool value);
    void dense_ffn(Block& B, int il, int n);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_dense_ = 0, n_rot_ = 0;
    float theta_ = 0;         ///< a rotary pair turns by position * theta^pair
    bool w_norm_ = false;     ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;       ///< and then scaled by this, when it is not 0
    int chunk_ = 1;           ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    size_t score_n_ = 0;       ///< the floats of score_
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    float* q8w_ = nullptr;     ///< a --kv q8_0 prompt batch's K and V, dequantized to fp16 (kwork | vwork)
    size_t work_halves_ = 0;   ///< the halves of each of q8w_'s two halves
};

}  // namespace strata::poly
