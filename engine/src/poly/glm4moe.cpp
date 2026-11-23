// src/poly/glm4moe.cpp - the layers of GLM-4.5 and GLM-4.5-Air.  The reference for the math is llama.cpp's glm4moe
// graph (src/models/glm4-moe.cpp): biased q, k and v projections, optional q and k norms, ggml_rope_ext in the NeoX
// pairing on rope.dimension_count values of a head, the attention scaled by 1 / sqrt(head_dim), a post-attention norm
// before the FFN, and build_moe_ffn with sigmoid gates, the selection bias and the weights' norm and scale, plus an
// ungated shared expert.  The caches and the MoE are Laguna's (laguna.cpp) without its window layers and gate.
#include "glm4moe.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr size_t kQ8WorkRows = 2048;               // the K or V blocks a q8 prompt batch dequantizes at a time
constexpr int64_t kGatingSigmoid = 2;              // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID

}  // namespace

void Glm4Moe::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void Glm4Moe::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto num = [&](const char* key, double otherwise) {
        const MetaValue* v = find(key);
        return v && v->is_num() ? v->num() : otherwise;
    };

    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the keys and values have different widths");
    need(head_dim_ % 32 == 0 && head_dim_ <= 256, "bad head size");
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count(_kv)");

    n_rot_ = shape_i("rope.dimension_count");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");
    theta_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);
    // GLM-4.5's files turn every position as trained (131,072); scaled positions would need YaRN here (laguna.cpp)
    const MetaValue* scaling = find("rope.scaling.type");
    const std::string kind = scaling && scaling->type == MetaType::STRING ? scaling->s : std::string("none");
    need(kind == "none" || num("rope.scaling.factor", 1.0) == 1.0, "rope.scaling is not supported for glm4moe");
    const MetaValue* sections = find("rope.dimension_sections");
    need(!sections || sections->type != MetaType::ARRAY ||
             std::all_of(sections->items.begin(), sections->items.end(), [](const MetaValue& v) { return v.u == 0; }),
         "rope.dimension_sections (a GLM-4.5V file) is not supported");

    const MetaValue* gating = find("expert_gating_func");
    need(!gating || (int64_t) gating->u == kGatingSigmoid, "the expert gate is not a sigmoid");
    const MetaValue* shared = find("expert_shared_count");
    need(!shared || shared->u == 1, "expert_shared_count is not 1");
    const MetaValue* groups = find("expert_group_count");
    need(!groups || groups->u <= 1, "routing by expert groups is not supported");
    const MetaValue* dense = find("leading_dense_block_count");
    n_dense_ = dense ? (int) dense->u : 0;
    need(n_dense_ >= 0 && n_dense_ < n_layer_, "bad leading_dense_block_count");
    n_ff_ = n_dense_ > 0 ? shape_i("feed_forward_length") : 0;
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : false;   // llama.cpp's default
    w_scale_ = (float) num("expert_weights_scale", 0.0);

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    // a prompt batch's products guard their half sums until a run shows GLM-4.5's stay in range
    half_sum_guard_ = true;
    std::fprintf(stderr, "strata-poly: glm4moe: %d layers (%d dense), %d query heads over %d key heads of %d, %d of "
                 "them turned; %d experts, %d a token\n",
                 n_layer_, n_dense_, n_head_, n_kv_, head_dim_, n_rot_, n_expert_, n_used_);
}

void Glm4Moe::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    auto opt = [&](const char* n, int64_t a) { return tensor(il, n) ? f32(n, a) : Dev{}; };

    const int64_t q_width = (int64_t) n_head_ * head_dim_, kv_width = (int64_t) n_kv_ * head_dim_;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("post_attention_norm.weight", n_embd_);
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    B.wq = mat("attn_q.weight", n_embd_, q_width);
    B.wk = mat("attn_k.weight", n_embd_, kv_width);
    B.wv = mat("attn_v.weight", n_embd_, kv_width);
    B.wo = mat("attn_output.weight", q_width, n_embd_);
    // every GLM-4.5 file has the biases (the model's attention_bias)
    B.bq = f32("attn_q.bias", q_width);
    B.bk = f32("attn_k.bias", kv_width);
    B.bv = f32("attn_v.bias", kv_width);
    B.q_norm = opt("attn_q_norm.weight", head_dim_);   // GLM-4.5's, not Air's
    B.k_norm = opt("attn_k_norm.weight", head_dim_);
    if (il < n_dense_) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, true);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

// The caches: every layer keeps a row a position.
void Glm4Moe::alloc_caches() {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t row = (size_t) n_kv_ * head_dim_;
    const size_t scale_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);   // a Q8_0 block's scale
    if (q8) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row + scale_row);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (q8) {
            B.k_codes = (int8_t*) kv_alloc(row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scale_row, 1);
            B.v_codes = (int8_t*) kv_alloc(row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scale_row, 1);
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Glm4Moe::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_n_ = (size_t) chunk_ * n_head_ * opt_.max_context;
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
    if (rows_ >= kStageRows && opt_.kv_format == KvFmt::kQ8_0) {
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

// A sequence's rows are every layer's.  Under --kv q8_0 a row is its codes and, in a second array, a scale every 32
// of them, saved one after the other.
size_t Glm4Moe::card_row_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (q8 ? B.k_codes != nullptr : B.keys != nullptr) ++layers;
    return layers * 2 * (code_row + scale_row);
}

void Glm4Moe::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    for (const Block& B : blocks_) {
        if (n == 0 || !(q8 ? B.k_codes != nullptr : B.keys != nullptr)) continue;
        if (q8) {
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            continue;
        }
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
        }
    }
}

void Glm4Moe::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    const size_t sk_c = (size_t) (first - from) * code_row, sk_s = (size_t) (first - from) * scale_row;
    const size_t all_c = (size_t) count * code_row, all_s = (size_t) count * scale_row;
    for (const Block& B : blocks_) {
        if (n == 0 || !(q8 ? B.k_codes != nullptr : B.keys != nullptr)) continue;
        if (q8) {
            ck(cudaMemcpy((uint8_t*) B.k_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy((uint8_t*) B.v_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            continue;
        }
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
        }
    }
}

// The recordings of a layer's head are numbered by the rows they are sized for.
int Glm4Moe::head_key(int, int n) const { return room_key(n); }

// A side of a layer's cache, at the positions.
void Glm4Moe::store(const float* x, const Block& B, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ8_0) {
        int8_t* codes = value ? B.v_codes : B.k_codes;
        uint16_t* scales = value ? B.v_scales : B.k_scales;
        if (recording_) return store_q8_at(x, codes, scales, width, at_, n, 0, st_);
        return store_q8(x, codes, scales, width, pos, n, st_);
    }
    uint16_t* cache = value ? B.vals : B.keys;
    if (recording_) return store_f16_at(x, cache, width, at_, n, 0, st_);
    store_f16(x, cache, width, pos, n, st_);
}

// A head's bias, its norm where the file has one, and its turn.
void Glm4Moe::turn(const Block&, const float* x, const Dev& bias, const Dev& norm, float* y, int heads, int64_t pos,
                   int rows, const long long* at) {
    head_bias_rope(x, fp(bias), norm.p ? fp(norm) : nullptr, y, head_dim_, heads, n_rot_, theta_, pos, rows, rms_eps_,
                   st_, at);
}

// The attention: biased queries and keys turned by the position; the biased values; the keys and values into the
// caches; each head's softmax over the rows; the output matrix.
void Glm4Moe::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const long long* rot = at ? at + 1 : nullptr;       // and the turn each of them is given, read there too
    lin(B.wq, xn_, qraw_, n, xq_);
    turn(B, qraw_, B.bq, B.q_norm, q_, n_head_, pos, n, rot);
    lin(B.wk, xn_, kraw_, n, xq_);
    turn(B, kraw_, B.bk, B.k_norm, k_, n_kv_, pos, n, rot);
    lin(B.wv, xn_, v_, n, xq_);
    add_bias(fp(B.bv), v_, n_kv_ * head_dim_, n, st_);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const Kvq8 kq = q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{};
    const Kvq8 vq = q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{};
    if (n >= kStageRows)
        attend_batch(blas_, q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, score_n_, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false, kq, vq,
                     q8 ? (uint16_t*) q8w_ : nullptr, q8 ? (uint16_t*) q8w_ + work_halves_ : nullptr,
                     q8 ? work_halves_ : 0);
    else
        attend(q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_,
               n_kv_, scale, st_, 0, 0, false, at, !at ? 0 : room(n), score_n_, kq, vq);
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence: the products for all of them at once, and a row at a time the turn (it
// takes one position and counts on from it), the stored keys and values and the attention, in the row's sequence.
void Glm4Moe::attn_rows(Block& B, int n) {
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    add_bias(fp(B.bv), v_, n_kv_ * head_dim_, n, st_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        turn(B, qraw_ + i * hq, B.bq, B.q_norm, q_ + i * hq, n_head_, 0, 1, ai + 1);
        turn(B, kraw_ + i * kv, B.bk, B.k_norm, k_ + i * kv, n_kv_, 0, 1, ai + 1);
        uint16_t* kch = q8 ? nullptr : (uint16_t*) seq_buf(r.seq, B.keys);
        uint16_t* vch = q8 ? nullptr : (uint16_t*) seq_buf(r.seq, B.vals);
        const Kvq8 kq = q8 ? Kvq8{(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales)}
                           : Kvq8{};
        const Kvq8 vq = q8 ? Kvq8{(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales)}
                           : Kvq8{};
        if (q8) {
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, 1, 0, st_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, 1, 0, st_);
        } else {
            store_f16_at(k_ + i * kv, kch, (int) kv, ai, 1, 0, st_);
            store_f16_at(v_ + i * kv, vch, (int) kv, ai, 1, 0, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = room(1);
        pos_ = keep;
        attend(q_ + i * hq, kch, vch, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_,
               0, 0, false, ai, sized, score_n_, kq, vq);
    }
    lin(B.wo, att_, ao_, n);
}

void Glm4Moe::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  Pre-norms only, as llama.cpp's graph:
// attn_norm before the attention, post_attention_norm before the FFN.
void Glm4Moe::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void Glm4Moe::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Glm4Moe::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from the
// bare gates, made to sum to one (expert_weights_norm) and scaled (expert_weights_scale).
void Glm4Moe::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Glm4Moe::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert, ungated: the routed experts' sum starts from its output.
void Glm4Moe::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

}  // namespace strata::poly
