// src/poly/ds_layout.hpp - Inkling 2 Mini's DSpark drafter in llama.cpp's layout (ISTA-DASLab's file, for llama.cpp
// b12720), made into the buffers Inkling2::ds_load() otherwise makes from Unsloth's layout.  Both are the same drafter:
// compared tensor by tensor, the layers' weights are the same bytes, the norms the same values, and the rest are steps
// this engine takes at load already done by the converter:
//   ctx_mix (each layer's) and markov_ctx_mix  = softmax of dspark_fuse's rows (the four mixes' weights)
//   ctx_norm, markov_ctx_norm                  = dflare_norm, dspark_norm
//   mask_embd [E, block - 1]                   = Unsloth's mask_embd [E] in each column
//   markov_w2 [R, vocabulary] and its bias     = the fold: output x dspark_hidden_out, output . its bias (ds_fold())
//   markov_w1, joint_proj                      = the same in Q8_0 instead of BF16, markov_w1 without the padding rows
//   token_embd_norm                            = the target's token_embd_norm, on the anchor's embedding
// Header-only and on the host, so that ds_layout_test.cpp runs it on the published files without a card.
#pragma once

#include "gguf_types.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::poly {

/// What ds_load() uploads from a drafter in llama.cpp's layout.
struct DsFolded {
    std::vector<float> alpha;      ///< 4 mixes x 4 features: each layer's mix, then the Markov head's (ds_mix())
    std::vector<float> mask;       ///< n_embd: the mask row ds_block_in() writes to every row after the anchor
    std::vector<float> tnorm;      ///< n_embd: the anchor's norm (token_embd_norm), or empty where the file has none
    std::vector<uint16_t> w1;      ///< vocabulary x rank, BF16: the Markov head's token rows, the padding's zeros
    std::vector<uint16_t> joint;   ///< 3 rank rows of (2 rank + 2 n_embd), BF16: joint_proj's rows as published
    std::vector<uint8_t> fold;     ///< vocabulary rows of rank values, Q8_0: markov_w2, the padding's zeros
    std::vector<float> fold_b;     ///< vocabulary: markov_w2's bias, the padding's zeros
};

/// Whether a drafter file is in llama.cpp's layout (the fold made already).
template <class File>
bool ds_is_folded(const File& f) {
    return f.find("markov_w2.weight") != nullptr;
}

/// Tensor `name` of rows of n_in values (n_out of them, or any number with n_out 0), in any type the engine converts
/// (Q8_0 as published), as floats.
template <class File>
std::vector<float> ds_floats(const File& f, const std::string& name, uint64_t n_in, uint64_t n_out,
                             const std::string& file) {
    const TensorInfo* t = f.find(name);
    const int blk = t ? type_block_values(t->type) : 0;
    if (!t || t->shape.empty() || t->shape[0] != n_in || (n_out && t->elements() != n_in * n_out) ||
        t->elements() % n_in || (t->type != 0 && !type_converts(t->type)) || blk <= 0 || n_in % (uint64_t) blk)
        throw std::runtime_error(file + ": " + name + (t ? " is not rows of " + std::to_string(n_in) +
                                                                " values in a type this engine converts"
                                                          : " is missing"));
    f.payload(*t);
    std::vector<float> v((size_t) t->elements());
    type_to_floats(t->type, f.data(*t), v.data(), (int64_t) v.size());
    return v;
}

/// The drafter's buffers from a file in llama.cpp's layout: `layers` layers, n_embd E, Markov rank R, the target's
/// vocabulary V (its padded rows past the file's `rows` of tokens).  `file` names it in errors.
template <class File>
DsFolded ds_folded(const File& f, int layers, uint64_t E, uint64_t R, uint64_t V, uint64_t n_unpadded,
                   const std::string& file) {
    DsFolded d;
    const auto fail = [&](const std::string& what) { throw std::runtime_error(file + ": " + what); };
    if (R == 0 || R % 32) fail("a Markov rank of a multiple of 32");
    // the mixes' weights, after their softmax already: each layer's, then the Markov head's
    for (int l = 0; l < layers; ++l) {
        const std::vector<float> m = ds_floats(f, "blk." + std::to_string(l) + ".ctx_mix.weight", 4, 1, file);
        d.alpha.insert(d.alpha.end(), m.begin(), m.end());
    }
    const std::vector<float> mm = ds_floats(f, "markov_ctx_mix.weight", 4, 1, file);
    d.alpha.insert(d.alpha.end(), mm.begin(), mm.end());
    // a mask row for each position of a block after the anchor: the published ones are one row repeated
    const std::vector<float> mask = ds_floats(f, "mask_embd.weight", E, 0, file);
    for (size_t k = E; k < mask.size(); ++k)
        if (mask[k] != mask[k % E])
            fail("mask_embd.weight has a row of its own for each position of a block, which this engine does not "
                 "run yet");
    d.mask.assign(mask.begin(), mask.begin() + (ptrdiff_t) E);
    if (f.find("token_embd_norm.weight")) d.tnorm = ds_floats(f, "token_embd_norm.weight", E, 1, file);
    if (f.find("markov_w2.scale")) fail("markov_w2.scale is not supported");
    // the Markov head's token rows as BF16, a row a token of the target's vocabulary
    const std::vector<float> w1 = ds_floats(f, "markov_w1.weight", R, 0, file);
    if (w1.size() / R < n_unpadded || w1.size() / R > V) fail("markov_w1.weight is not a row for each token");
    d.w1.assign((size_t) (V * R), 0);
    for (size_t i = 0; i < w1.size(); ++i) d.w1[i] = bf16_of(w1[i]);
    // the recurrence's projection, [state ; previous token's row ; hidden ; context] to 3 rank values
    const std::vector<float> j = ds_floats(f, "joint_proj.weight", 2 * R + 2 * E, 3 * R, file);
    d.joint.resize(j.size());
    for (size_t i = 0; i < j.size(); ++i) d.joint[i] = bf16_of(j[i]);
    // the fold: a Q8_0 row of rank values a token, as ds_fold() makes it, and its bias
    const TensorInfo* t = f.find("markov_w2.weight");
    const uint64_t rows = t && t->shape.size() == 2 && t->shape[0] == R ? t->shape[1] : 0;
    if (rows < n_unpadded || rows > V) fail("markov_w2.weight is not a row of " + std::to_string(R) + " values for "
                                            "each token");
    const size_t row = (size_t) (R / 32) * 34;
    d.fold.assign((size_t) V * row, 0);   // Q8_0 zeros: a scale of 0
    if (t->type == 8u) {
        f.payload(*t);
        std::memcpy(d.fold.data(), f.data(*t), (size_t) rows * row);
    } else {
        const std::vector<float> w2 = ds_floats(f, "markov_w2.weight", R, rows, file);
        for (uint64_t v = 0; v < rows; ++v)
            strata::quant::quantize_q8_0_ref(w2.data() + v * R, d.fold.data() + v * row, (int64_t) R);
    }
    d.fold_b.assign((size_t) V, 0.f);
    if (f.find("markov_w2.bias")) {
        const std::vector<float> b = ds_floats(f, "markov_w2.bias", rows, 1, file);
        std::copy(b.begin(), b.end(), d.fold_b.begin());
    }
    return d;
}

}  // namespace strata::poly
