//! C ABI shim over llama.cpp's native Chatterbox TTS path (T010, layer B).
//!
//! The lane (`src/lanes/ggml_ffi.rs`) binds this file; the file binds
//! llama.cpp's production pieces DIRECTLY:
//!   - `common_init_from_params` (common/common.cpp, libllama-common.a) for
//!     model + context + sampler init,
//!   - `mtmd_context` + the `mtmd_helper_gen_audio_*` C API
//!     (tools/mtmd/mtmd-helper.h, libmtmd.a) for the gen-audio pipeline,
//!   - `common_sampler_*` (libllama-common.a) for the semantic-code sampler.
//!
//! The synthesis recipe mirrors `tools/tts/tts.cpp` LINE FOR LINE (the
//! faithful-port law): embedding=true backbone init, chat-TTS prompt
//! (empty for chatterbox -> raw prompt), speaker bitmap (shipped voice 0
//! or a caller WAV), `set_input` -> `step_prompt` loop ->
//! `sample_semantic_code` + `step_gen` frame loop -> `get_output`.
//! Deviations from the tool are labeled DLV-* in the module docs of
//! `src/lanes/ggml_ffi.rs`; the two structural ones:
//!   DLV-1  per-synth-call sampler rebuild (common_sampler_init/free) so
//!          each seam request carries its own seed (llama-tts is one-shot
//!          per process; our lane is multi-chunk per process), and
//!   DLV-2  the caller copies the returned WAV buffer before the next
//!          synth call (buffer is owned by the pipeline until set_input
//!          resets it).
//! Everything else — defaults (top_k 40, top_p 0.95, min_p 0.05, temp
//! 0.80, repeat 1.0), n_batch 2048, the 512-frame cap when `n_predict`
//! <= 0, `embedding = true`, mmproj CPU-only — matches the tool exactly.
//!
//! Link contract (from `tools/tts/CMakeLists.txt` + transitive ggml):
//!   tts_cbx_shim mtmd llama-common llama ggml ggml-cpu ggml-base
//!   + system: gomp numa m dl pthread stdc++
//! Emitted by the workspace `build.rs` behind the `ggml-ffi` feature.

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <cstring>
#include <memory>
#include <string>

// --------------------------------------------------------------------------
// Process-wide init (llama-tts runs it in main; we gate on a static flag so
// many lane instances in one process see the same behavior).
// --------------------------------------------------------------------------

static void cbx_backend_once() {
    static bool done = false;
    if (done) {
        return;
    }
    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    done = true;
}

// --------------------------------------------------------------------------
// C ABI surface (see the lane module for the Rust side)
// --------------------------------------------------------------------------

struct tts_cbx_session {
    bool ok = false;
    common_init_result_ptr llama;
    mtmd::context_ptr mctx;
    mtmd_helper::gen_audio gen;
    mtmd::bitmap_ptr speaker;      // shipped voice 0 (default) or caller WAV
    std::string voice_name;        // populated when a shipped voice was taken
    int32_t n_batch = 2048;
    int32_t n_predict_cap = 512;   // llama-tts law: n_predict > 0 ? n_predict : 512
    std::string last_error;
    std::string prompt_scratch;    // chat-TTS prompt text (must outlive inp)
    std::string lang_scratch;      // empty for chatterbox
    mtmd_gen_audio_info info{};
};

extern "C" tts_cbx_session * tts_cbx_init(
        const char * model_path,
        const char * mmproj_path,
        int32_t      threads,
        int32_t      n_batch,
        int32_t      n_predict_cap) {
    cbx_backend_once();
    auto * s = new tts_cbx_session();
    s->n_batch = n_batch > 0 ? n_batch : 2048;
    // The llama-tts default (n_predict = -1 -> cap 512) when 0/negative.
    s->n_predict_cap = n_predict_cap > 0 ? n_predict_cap : 512;

    common_params params;
    params.embedding = true; // always: hidden states feed the audio helper
    params.model.path = model_path ? model_path : "";
    if (threads > 0) {
        params.cpuparams.n_threads       = threads;
        params.cpuparams_batch.n_threads = threads;
    }
    // sampler chain = common defaults; the seed is per CALL (DLV-1), the
    // init-time sampler built inside common_init_from_params is unused.
    params.sampling.top_k          = 40;
    params.sampling.top_p          = 0.95f;
    params.sampling.min_p          = 0.05f;
    params.sampling.temp           = 0.80f;
    params.sampling.penalty_repeat = 1.0f;

    s->llama = common_init_from_params(params);
    llama_model   * model = s->llama ? s->llama->model() : nullptr;
    llama_context * lctx  = s->llama ? s->llama->context() : nullptr;
    if (!model || !lctx) {
        s->last_error = "failed to init model/context from "
                        + params.model.path;
        return s;
    }

    // mmproj: CPU-only (the workspace builds ggml without GPU backends —
    // mirroring the measured llama-tts CPU run).
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;
    mtmd_params.device  = nullptr;
    s->mctx.reset(mtmd_init_from_file(mmproj_path ? mmproj_path : "", model, mtmd_params));
    if (!s->mctx) {
        s->last_error = std::string("failed to load mmproj ")
                        + (mmproj_path ? mmproj_path : "(null)");
        return s;
    }
    s->info = mtmd_gen_audio_get_info(s->mctx.get());
    if (s->info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        s->last_error = "mmproj does not support audio generation";
        return s;
    }

    s->gen.init(lctx, s->mctx.get());
    if (!s->gen.valid()) {
        s->last_error = "gen_audio pipeline init failed";
        return s;
    }

    // Chatterbox ships one voice ("default", the conds.pt-derived pack);
    // mirror llama-tts: no explicit voice -> voice 0, so the lane's
    // VoiceRef::Default == VoiceRef::Id(<voice_name>).
    if (s->info.n_voices > 0) {
        const char * name = mtmd_gen_audio_voice_name(s->mctx.get(), 0);
        s->voice_name = name ? name : "";
        if (!s->voice_name.empty() && !s->speaker) {
            s->speaker.reset(mtmd_gen_audio_get_voice(s->mctx.get(), s->voice_name.c_str()));
            if (!s->speaker) {
                s->last_error = "failed to load shipped voice " + s->voice_name;
                return s;
            }
        }
    }

    s->ok = true;
    return s;
}

extern "C" int32_t tts_cbx_ok(tts_cbx_session * s) {
    return s && s->ok ? 1 : 0;
}

extern "C" int32_t tts_cbx_set_speaker_wav(tts_cbx_session * s, const char * wav_path) {
    if (!s || !s->ok || !wav_path) {
        return 1;
    }
    auto wrapper = mtmd_helper_bitmap_init_from_file(
        s->mctx.get(), wav_path, false, mtmd_helper_init_opt_default());
    if (!wrapper.bitmap) {
        s->last_error = std::string("failed to load speaker wav ") + wav_path;
        return 1;
    }
    s->speaker.reset(wrapper.bitmap);
    return 0;
}

// Restore the shipped-voice speaker (lane law: voice resolution is per
// request — Default/Id clear a previously set cloning reference so the
// speaker bitmap cannot leak across requests in a resident process).
extern "C" int32_t tts_cbx_reset_speaker(tts_cbx_session * s) {
    if (!s || !s->ok) {
        return 1;
    }
    if (s->info.n_voices > 0 && !s->voice_name.empty()) {
        auto shipped = mtmd::bitmap_ptr(
            mtmd_gen_audio_get_voice(s->mctx.get(), s->voice_name.c_str()));
        if (!shipped) {
            s->last_error = "failed to restore shipped voice " + s->voice_name;
            return 1;
        }
        s->speaker = std::move(shipped);
    }
    return 0;
}

extern "C" int32_t tts_cbx_sample_rate(tts_cbx_session * s) {
    if (!s || !s->mctx) {
        return 0;
    }
    // mtmd_gen_audio_info.sample_rate — the SAME info struct the pipeline's
    // get_output() publishes; known at load, before the first synth.
    return s->info.sample_rate;
}

extern "C" int32_t tts_cbx_n_voices(tts_cbx_session * s) {
    return s ? s->info.n_voices : 0;
}

extern "C" const char * tts_cbx_voice_name(tts_cbx_session * s) {
    return s && !s->voice_name.empty() ? s->voice_name.c_str() : nullptr;
}

// One synthesis call == one llama-tts main() body (minus process setup).
// Returns 0 on success; out_wav points into pipeline-owned memory valid
// until the next call or free (DLV-2). Stage timings are seconds.
extern "C" int32_t tts_cbx_synthesize(
        tts_cbx_session * s,
        const char * text,
        uint64_t seed,
        int32_t  top_k,
        float    top_p,
        float    min_p,
        float    temp,
        float    penalty_repeat,
        int32_t * out_sample_rate,
        const char ** out_wav,
        size_t * out_wav_len,
        int64_t * out_samples,
        int32_t * out_frames,
        double * out_prompt_s,
        double * out_gen_s,
        double * out_wav_s) {
    if (!s || !s->ok) {
        return 1;
    }
    llama_model   * model = s->llama->model();
    llama_context * lctx  = s->llama->context();

    // Chat-TTS prompt (LFM3-style templates); chatterbox has none -> raw.
    auto tmpls = common_chat_templates_init(model, "");
    std::string prompt = common_chat_get_tts_prompt(
        tmpls.get(), text ? text : "", s->lang_scratch, s->voice_name);
    if (prompt.empty()) {
        prompt = text ? text : "";
    }
    if (prompt.empty()) {
        s->last_error = "empty prompt";
        return 1;
    }
    s->prompt_scratch = std::move(prompt);

    // DLV-1: per-call sampler with the call's seed (single-sequence sampler;
    // penalties/chain = common defaults, dials mirroring llama-tts).
    common_params_sampling sparams;
    sparams.top_k          = top_k;
    sparams.top_p          = top_p;
    sparams.min_p          = min_p;
    sparams.temp           = temp;
    sparams.penalty_repeat = penalty_repeat;
    sparams.seed           = (uint32_t) seed;
    common_sampler * smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        s->last_error = "sampler init failed";
        return 1;
    }

    mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id      = 0;
    inp.prompt      = s->prompt_scratch.c_str();
    inp.prompt_len  = s->prompt_scratch.size();
    inp.speaker_ref = s->speaker.get();
    inp.lang        = s->lang_scratch.empty() ? nullptr : s->lang_scratch.c_str();
    inp.top_k       = top_k;
    inp.top_p       = top_p;
    inp.seed        = (uint32_t) seed;
    inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    inp.stream      = false;

    // set_input resets pipeline buffers AND the KV sequence: each call is a
    // fresh, position-independent decode (chatterbox pipeline law).
    if (s->gen.set_input(&inp) != 0) {
        s->last_error = "set_input failed";
        common_sampler_free(smpl);
        return 1;
    }

    const int64_t t_prompt_start_us = ggml_time_us();
    for (;;) {
        int32_t ret = s->gen.step_prompt(s->n_batch);
        if (ret < 0) {
            s->last_error = "prompt processing failed";
            common_sampler_free(smpl);
            return 1;
        }
        if (ret == 0) {
            break;
        }
    }
    const int64_t t_gen_start_us = ggml_time_us();

    auto sample_semantic_code = [&]() -> llama_token {
        llama_token t = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, t, true);
        return t;
    };

    const int max_new = s->n_predict_cap;
    int n_frames = 0;
    llama_token sampled = sample_semantic_code();
    const float * h_state = llama_get_embeddings_ith(lctx, -1);
    // The embeddings buffer must exist — embedding=true forced the context
    // to allocate it (common_init_from_params law from llama-tts).

    bool stop = false;
    while (!stop && n_frames < max_new) {
        const float * h_next = nullptr;
        // stage 2+3 fused: semantic -> acoustic -> audio in one step
        if (s->gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
            s->last_error = "step_gen failed at frame " + std::to_string(n_frames);
            common_sampler_free(smpl);
            return 1;
        }
        if (!h_next) {
            break; // stopped without generating a frame
        }
        n_frames++;
        h_state = h_next;
        sampled = sample_semantic_code();
    }
    const int64_t t_gen_done_us = ggml_time_us();

    int32_t sample_rate = 0;
    const char * data = nullptr;
    size_t data_len = 0;
    int64_t n_samples = 0;
    if (s->gen.get_output(&sample_rate, &data, &data_len, &n_samples) != 0) {
        s->last_error = "get_output failed";
        common_sampler_free(smpl);
        return 1;
    }
    const int64_t t_wav_done_us = ggml_time_us();

    common_sampler_free(smpl);

    if (out_sample_rate) *out_sample_rate = sample_rate;
    if (out_wav)         *out_wav = data;
    if (out_wav_len)     *out_wav_len = data_len;
    if (out_samples)     *out_samples = n_samples;
    if (out_frames)      *out_frames = n_frames;
    if (out_prompt_s)    *out_prompt_s = (t_gen_start_us - t_prompt_start_us) / 1e6;
    if (out_gen_s)       *out_gen_s = (t_gen_done_us - t_gen_start_us) / 1e6;
    if (out_wav_s)       *out_wav_s = (t_wav_done_us - t_gen_done_us) / 1e6;
    return 0;
}

extern "C" const char * tts_cbx_last_error(tts_cbx_session * s) {
    if (!s || s->last_error.empty()) {
        return nullptr;
    }
    return s->last_error.c_str();
}

extern "C" void tts_cbx_free(tts_cbx_session * s) {
    if (s) {
        // gen (RAII), mctx (unique_ptr) and llama (common_init_result_ptr)
        // release in member-destructor order; llama_backend_free is
        // deliberately NOT called here (process-lifetime backend, the
        // single llama-ffi lane per process holds it once).
        delete s;
    }
}
