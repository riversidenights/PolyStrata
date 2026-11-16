// src/poly/hy_v3.cpp - the layers of HY V3 (Hy-MT2-30B-A3B).  The reference for the math is llama.cpp's hy_v3
// graph (src/models/hy-v3.cpp), and beside it transformers' modeling_hy_v3.py.
#include "hy_v3.hpp"

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
// llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID as the GGUF counts it
constexpr int64_t kGatingSigmoid = 2;

// llama.cpp writes this architecture's selection bias without a suffix (blk.N.exp_probs_b); the other
// architectures' files call theirs exp_probs_b.bias
const Names kAlias[] = {
    {"exp_probs_b", {"exp_probs_b.bias", nullptr}},
};

}  // namespace

const Names* HyV3::tensor_alias(const char* name) const {
    for (const Names& n : kAlias)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

void HyV3::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_run_ = 0;
}

void HyV3::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    // the heads are wider than the model's width over their number: the file says how wide
    const MetaValue* klen = find("attention.key_length");
    const MetaValue* vlen = find("attention.value_length");
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    need(n_kv_ > 0 && n_head_ % n_kv_ == 0, "bad attention.head_count_kv");
    head_dim_ = klen ? (int) klen->u : n_embd_ / n_head_;
    need(!vlen || (int) vlen->u == head_dim_, "the attention's keys and values differ in length");
    need(head_dim_ > 0 && head_dim_ % 2 == 0 && head_dim_ <= 256, "bad attention.key_length");
    need(!q8() || (n_kv_ * head_dim_) % 32 == 0, "--kv q8_0 needs n_kv * head_dim to be a multiple of 32");
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) head_dim_);

    // llama.cpp takes a file without the key as a sigmoid too
    const MetaValue* gating = find("expert_gating_func");
    need(!gating || gating->u == 0 || (int64_t) gating->u == kGatingSigmoid, "the expert gate is not a sigmoid");
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : false;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;

    // A layer without a router has a dense FFN, as llama.cpp tells them apart (the model's first layer).
    gelu_ = false;
    dense_.assign((size_t) n_layer_, 0);
    int n_dense = 0;
    for (int il = 0; il < n_layer_; ++il) {
        dense_[(size_t) il] = tensor(il, "ffn_gate_inp.weight") == nullptr;
        n_dense += dense_[(size_t) il];
    }
    need(n_dense < n_layer_, "no layer with routed experts");
    need(n_expert_ > 0 && n_ff_exp_ > 0 && n_ff_shexp_ > 0, "no routed experts");
    n_ff_ = n_dense > 0 ? shape_i("feed_forward_length") : 0;

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    // Not measured for this model: whether a block of its experts' products sums past a half's range.  The guard
    // is on until a run without it is compared (it cost LFM2.5 4% of a 30K prompt's read on a 5090).
    half_sum_guard_ = true;
}

void HyV3::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    ++n_run_;
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.q_norm = f32("attn_q_norm.weight", head_dim_);
    B.k_norm = f32("attn_k_norm.weight", head_dim_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
    if (il < n_layer_ && dense_[(size_t) il]) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, true);   // the shared expert beside the routed ones
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b")) {   // llama.cpp reads it when the file has it
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, b->name, n_expert_, 1, kF32);
    }
}

int HyV3::widest() const { return std::max({n_head_ * head_dim_, n_ff_exp_, n_ff_shexp_, 256}); }

// Every layer's caches: a row a position.  Under --kv q8_0 a row is its codes and, in a second array, an fp16
// scale a 32-value block (store_q8).
void HyV3::alloc_caches() {
    const size_t row = (size_t) n_kv_ * head_dim_;
    const size_t scales = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);   // a q8_0 row's scales
    if (q8()) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row + scales);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (q8()) {
            B.k_codes = (int8_t*) kv_alloc(row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scales, 1);
            B.v_codes = (int8_t*) kv_alloc(row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scales, 1);
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void HyV3::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_n_ = std::max((size_t) chunk_ * n_head_ * opt_.max_context,
                        attend_split_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
    if (rows_ >= kStageRows && q8()) {
        // the K and V blocks a q8 prompt batch dequantizes, side by side; work_halves_ halves each
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

// The layers' rows of the whole context.  Under --kv q8_0 a row is its codes and, in a second array, a scale every
// 32 of them; the two are saved one after the other on each side.
size_t HyV3::card_row_bytes() const {
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8() ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8() ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    return (size_t) n_run_ * 2 * (code_row + scale_row);
}

void HyV3::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8() ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8() ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    for (const Block& B : blocks_) {
        if (n == 0 || !(q8() ? B.k_codes != nullptr : B.keys != nullptr)) continue;   // another card's layer
        if (q8()) {
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

void HyV3::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8() ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8() ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    const size_t sk_c = (size_t) (first - from) * code_row, sk_s = (size_t) (first - from) * scale_row;
    const size_t all_c = (size_t) count * code_row, all_s = (size_t) count * scale_row;
    for (const Block& B : blocks_) {
        if (n == 0 || !(q8() ? B.k_codes != nullptr : B.keys != nullptr)) continue;   // as save_card_rows() skips them
        if (q8()) {
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

// The attention: queries and keys normed a head and turned by the position (the NeoX pairing, transformers'
// rotate_half), the keys and values into the caches.  The values go as they come.
void HyV3::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, head_dim_, theta_scale_, pos, n, rms_eps_,
                   st_, nullptr, nullptr, 1, at ? at + 1 : nullptr);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, head_dim_, theta_scale_, pos, n, rms_eps_,
                   st_, nullptr, nullptr, 1, at ? at + 1 : nullptr);
    lin(B.wv, xn_, v_, n, xq_);
    const bool is_q8 = q8();
    const int width = n_kv_ * head_dim_;
    if (is_q8) {
        if (at) {
            store_q8_at(k_, B.k_codes, B.k_scales, width, at, n, 0, st_);
            store_q8_at(v_, B.v_codes, B.v_scales, width, at, n, 0, st_);
        } else {
            store_q8(k_, B.k_codes, B.k_scales, width, pos, n, st_);
            store_q8(v_, B.v_codes, B.v_scales, width, pos, n, st_);
        }
    } else if (at) {
        store_f16_at(k_, B.keys, width, at, n, 0, st_);
        store_f16_at(v_, B.vals, width, at, n, 0, st_);
    } else {
        store_f16(k_, B.keys, width, pos, n, st_);
        store_f16(v_, B.vals, width, pos, n, st_);
    }
    const float scale = 1.f / std::sqrt((float) head_dim_);   // llama.cpp's kq_scale
    const Kvq8 kq = is_q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{},
                vq = is_q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{};
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company.
    // Under q8 each block of K and V the prompt batch's products read is dequantized into q8w_ (kwork then vwork)
    if (n >= kStageRows) {
        uint16_t* kw = is_q8 ? (uint16_t*) q8w_ : nullptr;
        uint16_t* vw = is_q8 ? kw + work_halves_ : nullptr;
        attend_batch(blas_, q_, is_q8 ? nullptr : B.keys, is_q8 ? nullptr : B.vals, pos, n, score_,
                     (size_t) chunk_ * n_head_ * opt_.max_context, awork_, att_, head_dim_, n_head_, n_kv_, scale, st_,
                     0, 0, false, kq, vq, kw, vw, is_q8 ? work_halves_ : 0);
    } else {
        attend(q_, is_q8 ? nullptr : B.keys, is_q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_,
               n_kv_, scale, st_, 0, 0, false, at, at ? room(n) : 0, score_n_, kq, vq);
    }
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence: the products for all of them at once, and a row at a time the turn (it
// takes one position and counts on from it), the stored keys and values and the attention, in the row's sequence.
void HyV3::attn_rows(Block& B, int n) {
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
    const bool is_q8 = q8();
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        head_norm_rope(qraw_ + i * hq, head_dim_, fp(B.q_norm), q_ + i * hq, head_dim_, n_head_, head_dim_,
                       theta_scale_, 0, 1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        head_norm_rope(kraw_ + i * kv, head_dim_, fp(B.k_norm), k_ + i * kv, head_dim_, n_kv_, head_dim_,
                       theta_scale_, 0, 1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        Kvq8 kq{}, vq{};
        uint16_t* kc = nullptr;
        uint16_t* vc = nullptr;
        if (is_q8) {
            kq = {(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales)};
            vq = {(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales)};
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, 1, 0, st_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, 1, 0, st_);
        } else {
            kc = (uint16_t*) seq_buf(r.seq, B.keys);
            vc = (uint16_t*) seq_buf(r.seq, B.vals);
            store_f16_at(k_ + i * kv, kc, (int) kv, ai, 1, 0, st_);
            store_f16_at(v_ + i * kv, vc, (int) kv, ai, 1, 0, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;   // room(1) sizes the launch for the row's own place in its sequence
        const int sized = room(1);
        pos_ = keep;
        attend(q_ + i * hq, kc, vc, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_, 0, 0,
               false, ai, sized, score_n_, kq, vq);
    }
    lin(B.wo, att_, ao_, n);
}

// The recordings of a layer's head are numbered by the rows they are sized for.
int HyV3::head_key(int, int n) const { return room_key(n); }

void HyV3::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.  A norm and its quantization are one launch, and so are the residual's sum and the next norm.
void HyV3::layer_head(Layer& L, int il, int n) {
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

void HyV3::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void HyV3::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model (and transformers' HYV3TopKRouter): sigmoid gates, the n_used largest
// of gate + bias, and weights from the bare gates, made to sum to one and then scaled.
void HyV3::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
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

bool HyV3::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert, ungated: the routed experts' sum starts from its output.
void HyV3::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

}  // namespace strata::poly
