// src/poly/ds_layout_test.cpp - ds_layout.hpp on the published DSpark drafters of Inkling 2 Mini: ISTA-DASLab's file
// (llama.cpp's layout) made into the buffers Inkling2::ds_load() fills, against what the engine makes of Unsloth's
// file of the same drafter.  No card: the buffers are the host's, before their upload.
//
//   poly_ds_layout_test <ISTA-DASLab's dspark-Inkling-2-Mini-Q8_0.gguf> <Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf>
//
// or the paths in STRATA_DS_ISTA and STRATA_DS_UNSLOTH; without them it says SKIPPED.  The files: the pinned
// revisions of ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF and unsloth/Inkling-2-Mini-GGUF (terminal/setup.py).
#include "ds_layout.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::poly;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

/// GgufFile as ds_folded() reads a file (Shards' find, data and payload).
struct File {
    GgufFile g;
    explicit File(const std::string& path) : g(path) {}
    const TensorInfo* find(const std::string& n) const { return g.find(n); }
    const uint8_t* data(const TensorInfo& t) const { return g.tensor_data(t); }
    size_t payload(const TensorInfo& t) const { return (size_t) tensor_payload_bytes(t); }
};

std::vector<float> floats(const File& f, const std::string& name) {
    const TensorInfo* t = f.find(name);
    std::vector<float> v((size_t) t->elements());
    type_to_floats(t->type, f.data(*t), v.data(), (int64_t) v.size());
    return v;
}

float of_bf16(uint16_t b) { return quant::f32_from_bits((uint32_t) b << 16); }

/// |a - b| / |b| over the first n values.
double rel(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    double d = 0, z = 0;
    for (size_t i = 0; i < n; ++i) {
        d += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        z += (double) b[i] * b[i];
    }
    return std::sqrt(d / z);
}

}  // namespace

int main(int argc, char** argv) {
    const char* ista = argc > 2 ? argv[1] : std::getenv("STRATA_DS_ISTA");
    const char* unsloth = argc > 2 ? argv[2] : std::getenv("STRATA_DS_UNSLOTH");
    if (!ista || !unsloth) {
        std::printf("SKIPPED: poly_ds_layout_test needs the two DSpark drafter files (STRATA_DS_ISTA, "
                    "STRATA_DS_UNSLOTH), which are not in this tree\n");
        return 0;
    }
    const File I(ista), U(unsloth);
    constexpr uint64_t E = 2304, R = 256, V = 201024, kUnpadded = 200058;
    check(ds_is_folded(I) && !ds_is_folded(U), "ISTA-DASLab's file is llama.cpp's layout, Unsloth's is not");
    const DsFolded d = ds_folded(I, 3, E, R, V, kUnpadded, ista);

    // the mixes: the softmax of each of dspark_fuse's rows (Inkling2::ds_load()'s, in double)
    const std::vector<float> fuse = floats(U, "dspark_fuse.weight");
    double worst = 0;
    for (int j = 0; j < 4; ++j) {
        double m = fuse[j * 4], z = 0;
        for (int k = 1; k < 4; ++k) m = std::max(m, (double) fuse[j * 4 + k]);
        for (int k = 0; k < 4; ++k) z += std::exp((double) fuse[j * 4 + k] - m);
        for (int k = 0; k < 4; ++k)
            worst = std::max(worst, std::fabs(d.alpha[j * 4 + k] - std::exp((double) fuse[j * 4 + k] - m) / z));
    }
    check(d.alpha.size() == 16 && worst < 1e-6, "the mixes' weights are the softmax of dspark_fuse's rows (" +
                                                    std::to_string(worst) + ")");

    // the mask row: Unsloth's, in Q8_0
    const std::vector<float> mask = floats(U, "mask_embd.weight");
    check(d.mask.size() == E && rel(d.mask, mask, E) < 0.02, "the mask row is Unsloth's (" +
                                                                 std::to_string(rel(d.mask, mask, E)) + ")");
    check(d.tnorm.size() == E, "the anchor's norm is there");

    // the Markov head's token rows: Unsloth's BF16 rows in Q8_0, the padding's zeros
    const std::vector<float> w1 = floats(U, "markov_w1.weight");
    std::vector<float> ours(d.w1.size());
    for (size_t i = 0; i < ours.size(); ++i) ours[i] = of_bf16(d.w1[i]);
    bool pad = true;
    for (size_t i = kUnpadded * R; i < V * R; ++i) pad = pad && d.w1[i] == 0;
    check(d.w1.size() == V * R && rel(ours, w1, kUnpadded * R) < 0.01 && pad,
          "markov_w1's rows are Unsloth's (" + std::to_string(rel(ours, w1, kUnpadded * R)) + "), the padding's zeros");

    // joint_proj: Unsloth's BF16 rows in Q8_0, in the same order of inputs
    const std::vector<float> joint = floats(U, "joint_proj.weight");
    std::vector<float> jo(d.joint.size());
    for (size_t i = 0; i < jo.size(); ++i) jo[i] = of_bf16(d.joint[i]);
    check(jo.size() == joint.size() && rel(jo, joint, joint.size()) < 0.01,
          "joint_proj is Unsloth's (" + std::to_string(rel(jo, joint, joint.size())) + ")");

    // the fold: markov_w2's Q8_0 rows as published, then zeros; its bias, then zeros
    const TensorInfo* w2 = I.find("markov_w2.weight");
    const size_t row = (R / 32) * 34;
    bool same = std::memcmp(d.fold.data(), I.data(*w2), kUnpadded * row) == 0, zero = true;
    for (size_t i = kUnpadded * row; i < d.fold.size(); ++i) zero = zero && d.fold[i] == 0;
    check(d.fold.size() == V * row && same && zero, "the fold is markov_w2's rows, the padding's zeros");
    const std::vector<float> b2 = floats(I, "markov_w2.bias");
    bool bias = d.fold_b.size() == V;
    for (size_t v = 0; bias && v < V; ++v) bias = d.fold_b[v] == (v < kUnpadded ? b2[v] : 0.f);
    check(bias, "the fold's bias is markov_w2's, the padding's zeros");

    std::printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
