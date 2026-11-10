#include <android/log.h>
#include <jni.h>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>

#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include "llama.h"
#include "common.h"
#include "sampling.h"

#define TAG "LlamaMeshJNI"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGw(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)

static std::recursive_mutex g_mutex;
static llama_model   * g_model   = nullptr;
static llama_context * g_context = nullptr;
static common_sampler * g_sampler = nullptr;
static int64_t g_allocated_bytes = 0;
static int g_start_layer = 0;
static int g_end_layer = -1;

static jclass g_pipeline_result_class = nullptr;
static jmethodID g_pipeline_result_ctor = nullptr;

static std::vector<int> g_high_power_cores;

static void free_model_resources() {
    if (g_sampler) {
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_allocated_bytes = 0;
}

/**
 * Profiles heterogeneous ARM DynamIQ / big.LITTLE CPUs.
 * Identifies high-power (Prime & Gold) cores and segregates low-power (Silver/Cortex-A55) cores.
 */
static void profile_cpu_cores() {
    g_high_power_cores.clear();
    int available_cores = (int) sysconf(_SC_NPROCESSORS_ONLN);
    if (available_cores <= 0) available_cores = 4;

    int max_freq = 0;
    int freqs[64] = {0};
    int cores_scanned = std::min(available_cores, 64);

    for (int i = 0; i < cores_scanned; ++i) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE *f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%d", &freqs[i]) == 1) {
                if (freqs[i] > max_freq) max_freq = freqs[i];
            }
            fclose(f);
        }
    }

    if (max_freq > 0) {
        int threshold = (int) (max_freq * 0.80); // within 80% of peak frequency (Prime + Gold cores)
        for (int i = 0; i < cores_scanned; ++i) {
            if (freqs[i] >= threshold) {
                g_high_power_cores.push_back(i);
            }
        }
    }

    // Standard Android mobile SoC fallback:
    // Higher core indices (e.g. 4-7 on an 8-core SoC) are the performance big cores.
    if (g_high_power_cores.empty()) {
        if (available_cores >= 8) {
            for (int i = 4; i < available_cores; ++i) g_high_power_cores.push_back(i);
        } else if (available_cores >= 4) {
            for (int i = 2; i < available_cores; ++i) g_high_power_cores.push_back(i);
        } else {
            for (int i = 0; i < available_cores; ++i) g_high_power_cores.push_back(i);
        }
    }

    LOGi("Heterogeneous CPU Architecture: Identified %zu high-power cores out of %d total cores for LLM compute",
         g_high_power_cores.size(), available_cores);
}

/**
 * Pins the calling thread strictly to High-Power CPU cores.
 * Keeps compute off low-power cores to eliminate barrier synchronization stalls.
 */
static void pin_calling_thread_to_high_power_cpus() {
    if (g_high_power_cores.empty()) {
        profile_cpu_cores();
    }
    if (g_high_power_cores.empty()) return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core_id : g_high_power_cores) {
        CPU_SET(core_id, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

static int detect_optimal_cpu_threads(int requested_threads) {
    if (requested_threads > 0) {
        return requested_threads;
    }
    if (g_high_power_cores.empty()) {
        profile_cpu_cores();
    }
    return std::max(1, (int) g_high_power_cores.size());
}

static void custom_llama_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text || text[0] == '\0') return;
    int android_level = ANDROID_LOG_INFO;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: android_level = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  android_level = ANDROID_LOG_WARN; break;
        case GGML_LOG_LEVEL_INFO:  android_level = ANDROID_LOG_INFO; break;
        case GGML_LOG_LEVEL_DEBUG: android_level = ANDROID_LOG_DEBUG; break;
        default:                   android_level = ANDROID_LOG_INFO; break;
    }
    __android_log_print(android_level, TAG, "%s", text);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeInit(JNIEnv *env, jobject thiz) {
    try {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        llama_log_set(custom_llama_log_callback, nullptr);
        profile_cpu_cores();
        llama_backend_init();

        jclass cls = env->FindClass("org/meshllm/node/engine/NativePipelineResult");
        if (cls) {
            if (g_pipeline_result_class) env->DeleteGlobalRef(g_pipeline_result_class);
            g_pipeline_result_class = (jclass) env->NewGlobalRef(cls);
            g_pipeline_result_ctor = env->GetMethodID(g_pipeline_result_class, "<init>", "(ILjava/lang/String;Z[F)V");
        }

        LOGi("Native llama.cpp CPU engine initialized with %zu high-power compute cores",
             g_high_power_cores.size());
        return JNI_TRUE;
    } catch (const std::exception &e) {
        LOGe("Exception in nativeInit: %s", e.what());
        return JNI_FALSE;
    } catch (...) {
        LOGe("Unknown native exception in nativeInit");
        return JNI_FALSE;
    }
}

JNIEXPORT jint JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeLoadModel(
    JNIEnv *env,
    jobject thiz,
    jstring jmodel_path,
    jint start_layer,
    jint end_layer,
    jint n_threads
) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    try {
        // Enforce thread affinity to high-power performance cores
        pin_calling_thread_to_high_power_cpus();

        // Free any previously loaded model completely
        free_model_resources();

        const char *model_path = env->GetStringUTFChars(jmodel_path, nullptr);
        if (!model_path) {
            LOGe("Failed to get model path string");
            return -1; // ERR_INVALID_ARGS
        }

        int optimal_threads = detect_optimal_cpu_threads(n_threads);
        int available_cores = (int) sysconf(_SC_NPROCESSORS_ONLN);

        LOGi("Loading model with pure CPU engine: %s (layers: [%d..%d], threads: %d on high-power cores, cores free: %d)",
             model_path, start_layer, end_layer, optimal_threads, available_cores - optimal_threads);

        // Strict CPU configuration: No GPU/VRAM allocation to prevent mobile crashes
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0; // Pure CPU execution
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.start_layer = start_layer;
        model_params.end_layer = end_layer;
        if (sizeof(void*) == 4 || end_layer >= 0) {
            model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        }

        llama_model *loaded_model = nullptr;
        try {
            loaded_model = llama_model_load_from_file(model_path, model_params);
        } catch (const std::exception &e) {
            LOGe("CPU model loading failed: %s", e.what());
            loaded_model = nullptr;
        } catch (...) {
            LOGe("Unknown exception during CPU model loading");
            loaded_model = nullptr;
        }

        env->ReleaseStringUTFChars(jmodel_path, model_path);

        if (!loaded_model) {
            LOGe("Failed to load model file with llama.cpp CPU engine");
            free_model_resources();
            return -2; // ERR_LOAD_FAILED
        }
        g_model = loaded_model;

        g_start_layer = start_layer;
        g_end_layer = end_layer;

        // Context setup optimized for pure mobile CPU execution
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.n_batch = 512;
        ctx_params.n_threads = optimal_threads;
        ctx_params.n_threads_batch = optimal_threads;
        ctx_params.type_k = GGML_TYPE_F16;
        ctx_params.type_v = GGML_TYPE_F16;
        ctx_params.embeddings = true; // Extract activation embeddings for pipeline parallelism

        llama_context *loaded_ctx = nullptr;
        try {
            loaded_ctx = llama_init_from_model(g_model, ctx_params);
        } catch (const std::exception &e) {
            LOGe("Failed to create llama CPU context: %s", e.what());
            loaded_ctx = nullptr;
        } catch (...) {
            LOGe("Unknown error during CPU context creation");
            loaded_ctx = nullptr;
        }

        if (!loaded_ctx) {
            LOGe("Failed to create llama context from CPU model");
            free_model_resources();
            return -3; // ERR_CONTEXT_FAILED
        }
        g_context = loaded_ctx;

        // Sampler setup with anti-repetition penalty
        try {
            common_params_sampling sparams;
            sparams.temp = 0.75f;
            sparams.top_k = 40;
            sparams.top_p = 0.90f;
            sparams.penalty_repeat = 1.20f;
            sparams.penalty_last_n = 64;
            sparams.penalty_freq = 0.40f;
            sparams.penalty_present = 0.30f;
            g_sampler = common_sampler_init(g_model, sparams);
        } catch (const std::exception &e) {
            LOGw("Sampler init warning: %s", e.what());
        }

        // Track allocated memory proportionally for the node's assigned layers
        int32_t total_layers = llama_model_n_layer(g_model);
        int64_t full_size = (int64_t) llama_model_size(g_model);
        if (g_end_layer >= g_start_layer && total_layers > 0) {
            int node_layers = std::min(total_layers, g_end_layer - g_start_layer + 1);
            g_allocated_bytes = (int64_t) std::round(((double) node_layers / (double) total_layers) * (double) full_size);
        } else {
            g_allocated_bytes = full_size;
        }
        LOGi("Model loaded successfully into RAM! Layers [%d..%d] of %d, Threads: %d (pinned to Prime/Gold CPU cores), RAM: %.1f MB (Full: %.1f MB)",
             g_start_layer, g_end_layer, total_layers, optimal_threads,
             (double) g_allocated_bytes / (1024.0 * 1024.0), (double) full_size / (1024.0 * 1024.0));

        return 0; // SUCCESS
    } catch (const std::exception &e) {
        LOGe("Fatal native exception in nativeLoadModel: %s", e.what());
        free_model_resources();
        return -2;
    } catch (...) {
        LOGe("Fatal unknown native exception in nativeLoadModel");
        free_model_resources();
        return -2;
    }
}

JNIEXPORT jintArray JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeTokenize(
    JNIEnv *env,
    jobject thiz,
    jstring jtext
) {
    try {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_model) {
            return env->NewIntArray(0);
        }

        const char *text = env->GetStringUTFChars(jtext, nullptr);
        if (!text) return env->NewIntArray(0);
        int text_len = (int) strlen(text);

        std::vector<llama_token> tokens(text_len + 16);
        int n_tokens = llama_tokenize(
            llama_model_get_vocab(g_model),
            text,
            text_len,
            tokens.data(),
            (int) tokens.size(),
            true,  // add_special
            false  // parse_special
        );

        env->ReleaseStringUTFChars(jtext, text);

        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(
                llama_model_get_vocab(g_model),
                text,
                text_len,
                tokens.data(),
                (int) tokens.size(),
                true,
                false
            );
        }

        jintArray result = env->NewIntArray(std::max(0, n_tokens));
        if (n_tokens > 0) {
            env->SetIntArrayRegion(result, 0, n_tokens, (const jint *) tokens.data());
        }
        return result;
    } catch (const std::exception &e) {
        LOGe("Exception in nativeTokenize: %s", e.what());
        return env->NewIntArray(0);
    } catch (...) {
        return env->NewIntArray(0);
    }
}

static std::string decode_token_to_piece(llama_token token) {
    if (!g_model) return "";
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) return "";

    if (llama_vocab_is_eog(vocab, token)) {
        return "";
    }

    std::string piece = common_token_to_piece(vocab, token, false);
    if (piece.empty()) {
        piece = common_token_to_piece(vocab, token, true);
    }
    // LEXICON-CORE: Sanitize special control tokens (e.g. Tekken [control_XX]) from leaking into text stream
    if (piece.rfind("[control_", 0) == 0 && !piece.empty() && piece.back() == ']') {
        return "";
    }
    return piece;
}

static jstring create_java_string_from_utf8(JNIEnv *env, const std::string &str) {
    if (str.empty()) return env->NewStringUTF("");

    jstring res = env->NewStringUTF(str.c_str());
    if (res && !env->ExceptionCheck()) {
        return res;
    }
    if (env->ExceptionCheck()) env->ExceptionClear();

    // Fallback: java.lang.String(byte[], "UTF-8") to handle byte-fallback / raw byte tokens on Android ART
    jbyteArray bytes = env->NewByteArray((jsize) str.size());
    if (!bytes) return env->NewStringUTF("");
    env->SetByteArrayRegion(bytes, 0, (jsize) str.size(), (const jbyte*) str.data());

    jclass strCls = env->FindClass("java/lang/String");
    if (!strCls) {
        env->DeleteLocalRef(bytes);
        return env->NewStringUTF("");
    }
    jstring charset = env->NewStringUTF("UTF-8");
    jmethodID strCtor = env->GetMethodID(strCls, "<init>", "([BLjava/lang/String;)V");
    jstring utfString = (jstring) env->NewObject(strCls, strCtor, bytes, charset);
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(charset);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        return env->NewStringUTF("");
    }
    return utfString;
}

static jobject create_pipeline_result(JNIEnv *env, int sampled_token, const std::string &piece, bool is_finished, const float *out_hidden, int out_hidden_size) {
    if (!g_pipeline_result_class || !g_pipeline_result_ctor) {
        jclass cls = env->FindClass("org/meshllm/node/engine/NativePipelineResult");
        if (cls) {
            if (g_pipeline_result_class) env->DeleteGlobalRef(g_pipeline_result_class);
            g_pipeline_result_class = (jclass) env->NewGlobalRef(cls);
            g_pipeline_result_ctor = env->GetMethodID(g_pipeline_result_class, "<init>", "(ILjava/lang/String;Z[F)V");
        }
    }
    if (!g_pipeline_result_class || !g_pipeline_result_ctor) {
        LOGe("Failed to locate NativePipelineResult class or constructor");
        return nullptr;
    }
    jstring jpiece = create_java_string_from_utf8(env, piece);
    jfloatArray jout = nullptr;
    if (out_hidden != nullptr && out_hidden_size > 0) {
        jout = env->NewFloatArray(out_hidden_size);
        env->SetFloatArrayRegion(jout, 0, out_hidden_size, out_hidden);
    }
    jobject res = env->NewObject(
        g_pipeline_result_class,
        g_pipeline_result_ctor,
        (jint) sampled_token,
        jpiece,
        (jboolean) is_finished,
        jout
    );
    if (jpiece) env->DeleteLocalRef(jpiece);
    if (jout) env->DeleteLocalRef(jout);
    return res;
}

static void batch_add_embd(
    struct llama_batch & batch,
    const float * embd_vector,
    int n_embd,
    llama_pos pos,
    const std::vector<llama_seq_id> & seq_ids,
    bool logits
) {
    int idx = batch.n_tokens;
    if (batch.embd && embd_vector) {
        memcpy(batch.embd + (size_t) idx * n_embd, embd_vector, n_embd * sizeof(float));
    }
    batch.pos[idx] = pos;
    batch.n_seq_id[idx] = (int32_t) seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[idx][i] = seq_ids[i];
    }
    batch.logits[idx] = logits ? 1 : 0;
    batch.n_tokens++;
}

JNIEXPORT jstring JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeDecodeToken(
    JNIEnv *env,
    jobject thiz,
    jint token_id
) {
    try {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_model) {
            return env->NewStringUTF("");
        }
        std::string piece = decode_token_to_piece((llama_token) token_id);
        return create_java_string_from_utf8(env, piece);
    } catch (const std::exception &e) {
        LOGe("Exception in nativeDecodeToken: %s", e.what());
        return env->NewStringUTF("");
    } catch (...) {
        return env->NewStringUTF("");
    }
}

JNIEXPORT jint JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeEvalTokensBatch(
    JNIEnv *env,
    jobject thiz,
    jintArray jtokens,
    jint start_pos,
    jboolean sample_last
) {
    try {
        pin_calling_thread_to_high_power_cpus();
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_context || !g_model) {
            return -1;
        }

        jsize n_tokens = env->GetArrayLength(jtokens);
        if (n_tokens <= 0) {
            return 0;
        }

        jint *tokens_ptr = env->GetIntArrayElements(jtokens, nullptr);
        if (!tokens_ptr) {
            return -1;
        }

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; ++i) {
            bool need_logits = (sample_last && i == n_tokens - 1);
            common_batch_add(batch, (llama_token) tokens_ptr[i], (llama_pos) (start_pos + i), { 0 }, need_logits);
        }

        int decode_res = llama_decode(g_context, batch);
        llama_batch_free(batch);
        env->ReleaseIntArrayElements(jtokens, tokens_ptr, JNI_ABORT);

        if (decode_res != 0) {
            LOGe("llama_decode batch failed with code %d", decode_res);
            return -2;
        }

        if (sample_last && g_sampler) {
            float * logits = llama_get_logits(g_context);
            if (!logits) {
                LOGe("No logits available from context for batch sampling");
                return -2;
            }
            llama_token sampled = common_sampler_sample(g_sampler, g_context, -1);
            common_sampler_accept(g_sampler, sampled, true);
            return (jint) sampled;
        }

        return 0;
    } catch (const std::exception &e) {
        LOGe("Exception in nativeEvalTokensBatch: %s", e.what());
        return -2;
    } catch (...) {
        LOGe("Unknown exception in nativeEvalTokensBatch");
        return -2;
    }
}

JNIEXPORT jint JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeEvalToken(
    JNIEnv *env,
    jobject thiz,
    jint token_id,
    jint pos,
    jboolean is_output
) {
    try {
        pin_calling_thread_to_high_power_cpus();
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_context || !g_model) {
            return -1;
        }

        llama_batch batch = llama_batch_get_one((llama_token *) &token_id, 1);
        if (llama_decode(g_context, batch) != 0) {
            LOGe("llama_decode failed at pos %d", pos);
            return -2;
        }

        if (is_output && g_sampler) {
            float * logits = llama_get_logits(g_context);
            if (!logits) {
                LOGe("No logits available from context for token sampling");
                return -2;
            }
            llama_token sampled = common_sampler_sample(g_sampler, g_context, -1);
            common_sampler_accept(g_sampler, sampled, true);
            return (jint) sampled;
        }

        return 0; // Intermediate layer activation completed
    } catch (const std::exception &e) {
        LOGe("Exception in nativeEvalToken: %s", e.what());
        return -2;
    } catch (...) {
        LOGe("Unknown exception in nativeEvalToken");
        return -2;
    }
}

JNIEXPORT jobject JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativePrefillPipeline(
    JNIEnv *env,
    jobject thiz,
    jintArray jtokens,
    jfloatArray jincoming_hidden,
    jboolean is_input,
    jboolean is_output
) {
    try {
        pin_calling_thread_to_high_power_cpus();
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_context || !g_model) {
            LOGe("nativePrefillPipeline: context or model is null");
            return nullptr;
        }

        int32_t n_embd = llama_model_n_embd(g_model);

        if (is_input) {
            if (!jtokens) {
                LOGe("nativePrefillPipeline: input block requires jtokens");
                return nullptr;
            }
            jsize n_tokens = env->GetArrayLength(jtokens);
            if (n_tokens <= 0) return nullptr;

            jint *tokens_ptr = env->GetIntArrayElements(jtokens, nullptr);
            if (!tokens_ptr) return nullptr;

            llama_batch batch = llama_batch_init(n_tokens, 0, 1);
            for (int i = 0; i < n_tokens; ++i) {
                bool need_output = is_output ? (i == n_tokens - 1) : true;
                common_batch_add(batch, (llama_token) tokens_ptr[i], (llama_pos) i, { 0 }, need_output);
            }

            int decode_res = llama_decode(g_context, batch);
            llama_batch_free(batch);
            env->ReleaseIntArrayElements(jtokens, tokens_ptr, JNI_ABORT);

            if (decode_res != 0) {
                LOGe("nativePrefillPipeline: llama_decode failed with code %d", decode_res);
                return nullptr;
            }

            if (is_output) {
                // Single-device mode: sample token from last prompt position
                float * logits = llama_get_logits(g_context);
                if (!logits) {
                    LOGe("nativePrefillPipeline: no logits available for sampling");
                    return nullptr;
                }
                llama_token sampled = common_sampler_sample(g_sampler, g_context, -1);
                common_sampler_accept(g_sampler, sampled, true);

                const llama_vocab * vocab = llama_model_get_vocab(g_model);
                bool is_finished = llama_vocab_is_eog(vocab, sampled);
                std::string piece = decode_token_to_piece(sampled);
                LOGi("nativePrefillPipeline (single): sampled token %d, piece: '%s' (len %zu), is_finished: %d",
                     (int) sampled, piece.c_str(), piece.size(), (int) is_finished);

                return create_pipeline_result(env, (int) sampled, piece, is_finished, nullptr, 0);
            } else {
                // Intermediate / Master node: extract embeddings for all n_tokens
                float * embd = llama_get_embeddings(g_context);
                if (!embd) {
                    LOGe("nativePrefillPipeline: no embeddings extracted from context");
                    return nullptr;
                }
                int total_floats = n_tokens * n_embd;
                return create_pipeline_result(env, -1, "", false, embd, total_floats);
            }
        } else {
            // Downstream node: receive incoming prompt embeddings
            if (!jincoming_hidden) {
                LOGe("nativePrefillPipeline: downstream node requires jincoming_hidden");
                return nullptr;
            }
            jsize hidden_len = env->GetArrayLength(jincoming_hidden);
            int n_tokens = hidden_len / n_embd;
            if (n_tokens <= 0) {
                LOGe("nativePrefillPipeline: hidden_len %d < n_embd %d", hidden_len, n_embd);
                return nullptr;
            }

            jfloat *hidden_ptr = env->GetFloatArrayElements(jincoming_hidden, nullptr);
            if (!hidden_ptr) return nullptr;

            llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
            for (int i = 0; i < n_tokens; ++i) {
                bool need_output = is_output ? (i == n_tokens - 1) : true;
                batch_add_embd(batch, hidden_ptr + (size_t) i * n_embd, n_embd, (llama_pos) i, { 0 }, need_output);
            }

            int decode_res = llama_decode(g_context, batch);
            llama_batch_free(batch);
            env->ReleaseFloatArrayElements(jincoming_hidden, hidden_ptr, JNI_ABORT);

            if (decode_res != 0) {
                LOGe("nativePrefillPipeline: downstream llama_decode failed with code %d", decode_res);
                return nullptr;
            }

            if (is_output) {
                // Final node in pipeline: sample first token from last prompt position
                float * logits = llama_get_logits(g_context);
                if (!logits) {
                    LOGe("nativePrefillPipeline: no logits available on final node");
                    return nullptr;
                }
                llama_token sampled = common_sampler_sample(g_sampler, g_context, -1);
                common_sampler_accept(g_sampler, sampled, true);

                const llama_vocab * vocab = llama_model_get_vocab(g_model);
                bool is_finished = llama_vocab_is_eog(vocab, sampled);
                std::string piece = decode_token_to_piece(sampled);
                LOGi("nativePrefillPipeline (downstream): sampled token %d, piece: '%s' (len %zu), is_finished: %d",
                     (int) sampled, piece.c_str(), piece.size(), (int) is_finished);

                return create_pipeline_result(env, (int) sampled, piece, is_finished, nullptr, 0);
            } else {
                // Intermediate slave node: extract embeddings for all n_tokens to send to next node
                float * embd = llama_get_embeddings(g_context);
                if (!embd) {
                    LOGe("nativePrefillPipeline: no embeddings extracted on intermediate node");
                    return nullptr;
                }
                int total_floats = n_tokens * n_embd;
                return create_pipeline_result(env, -1, "", false, embd, total_floats);
            }
        }
    } catch (const std::exception &e) {
        LOGe("Exception in nativePrefillPipeline: %s", e.what());
        return nullptr;
    } catch (...) {
        LOGe("Unknown exception in nativePrefillPipeline");
        return nullptr;
    }
}

JNIEXPORT jobject JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeForwardPipeline(
    JNIEnv *env,
    jobject thiz,
    jint token_id,
    jint pos,
    jfloatArray jincoming_hidden,
    jboolean is_input,
    jboolean is_output
) {
    try {
        pin_calling_thread_to_high_power_cpus();
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_context || !g_model) {
            LOGe("nativeForwardPipeline: context or model is null");
            return nullptr;
        }

        int32_t n_embd = llama_model_n_embd(g_model);

        if (is_input) {
            // Input node: evaluate single token
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, (llama_token) token_id, (llama_pos) pos, { 0 }, true);

            int decode_res = llama_decode(g_context, batch);
            llama_batch_free(batch);

            if (decode_res != 0) {
                LOGe("nativeForwardPipeline: llama_decode failed at pos %d with code %d", pos, decode_res);
                return nullptr;
            }

            if (is_output) {
                // Single-device mode
                float * logits = llama_get_logits(g_context);
                if (!logits) return nullptr;
                llama_token sampled = common_sampler_sample(g_sampler, g_context, -1);
                common_sampler_accept(g_sampler, sampled, true);

                const llama_vocab * vocab = llama_model_get_vocab(g_model);
                bool is_finished = llama_vocab_is_eog(vocab, sampled);
                std::string piece = decode_token_to_piece(sampled);
                LOGi("nativeForwardPipeline (single): sampled token %d, piece: '%s' (len %zu), is_finished: %d",
                     (int) sampled, piece.c_str(), piece.size(), (int) is_finished);

                return create_pipeline_result(env, (int) sampled, piece, is_finished, nullptr, 0);
            } else {
                // Master / intermediate node: extract 1 embedding vector (n_embd floats)
                float * embd = llama_get_embeddings(g_context);
                if (!embd) {
                    LOGe("nativeForwardPipeline: no embeddings extracted");
                    return nullptr;
                }
                return create_pipeline_result(env, -1, "", false, embd, n_embd);
            }
        } else {
            // Downstream node: evaluate single embedding vector
            if (!jincoming_hidden) {
                LOGe("nativeForwardPipeline: downstream node requires jincoming_hidden");
                return nullptr;
            }
            jsize hidden_len = env->GetArrayLength(jincoming_hidden);
            if (hidden_len < n_embd) {
                LOGe("nativeForwardPipeline: hidden_len %d < n_embd %d", hidden_len, n_embd);
                return nullptr;
            }

            jfloat *hidden_ptr = env->GetFloatArrayElements(jincoming_hidden, nullptr);
            if (!hidden_ptr) return nullptr;

            llama_batch batch = llama_batch_init(1, n_embd, 1);
            batch_add_embd(batch, hidden_ptr, n_embd, (llama_pos) pos, { 0 }, true);

            int decode_res = llama_decode(g_context, batch);
            llama_batch_free(batch);
            env->ReleaseFloatArrayElements(jincoming_hidden, hidden_ptr, JNI_ABORT);

            if (decode_res != 0) {
                LOGe("nativeForwardPipeline: downstream llama_decode failed at pos %d with code %d", pos, decode_res);
                return nullptr;
            }

            if (is_output) {
                // Final node in pipeline: sample from logits
                float * logits = llama_get_logits(g_context);
                if (!logits) {
                    LOGe("nativeForwardPipeline: no logits available on final node");
                    return nullptr;
                }
                llama_token sampled = common_sampler_sample(g_sampler, g_context, -1);
                common_sampler_accept(g_sampler, sampled, true);

                const llama_vocab * vocab = llama_model_get_vocab(g_model);
                bool is_finished = llama_vocab_is_eog(vocab, sampled);
                std::string piece = decode_token_to_piece(sampled);
                LOGi("nativeForwardPipeline (downstream): sampled token %d, piece: '%s' (len %zu), is_finished: %d",
                     (int) sampled, piece.c_str(), piece.size(), (int) is_finished);

                return create_pipeline_result(env, (int) sampled, piece, is_finished, nullptr, 0);
            } else {
                // Intermediate node: extract 1 embedding vector
                float * embd = llama_get_embeddings(g_context);
                if (!embd) {
                    LOGe("nativeForwardPipeline: no embeddings on intermediate node");
                    return nullptr;
                }
                return create_pipeline_result(env, -1, "", false, embd, n_embd);
            }
        }
    } catch (const std::exception &e) {
        LOGe("Exception in nativeForwardPipeline: %s", e.what());
        return nullptr;
    } catch (...) {
        LOGe("Unknown exception in nativeForwardPipeline");
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeUnloadModel(
    JNIEnv *env,
    jobject thiz
) {
    try {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        free_model_resources();
        LOGi("Model successfully unloaded from RAM (10m inactivity timeout or manual trigger)");
        return JNI_TRUE;
    } catch (const std::exception &e) {
        LOGe("Exception in nativeUnloadModel: %s", e.what());
        return JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jlong JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeGetAllocatedRamBytes(
    JNIEnv *env,
    jobject thiz
) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (jlong) g_allocated_bytes;
}

JNIEXPORT jboolean JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeIsModelLoaded(
    JNIEnv *env,
    jobject thiz
) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return (g_model != nullptr && g_context != nullptr) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeResetContext(
    JNIEnv *env,
    jobject thiz
) {
    try {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (g_context) {
            llama_memory_t mem = llama_get_memory(g_context);
            if (mem) {
                llama_memory_clear(mem, true);
            }
        }
        if (g_sampler) {
            common_sampler_reset(g_sampler);
        }
    } catch (const std::exception &e) {
        LOGe("Exception in nativeResetContext: %s", e.what());
    } catch (...) {}
}

JNIEXPORT jboolean JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeIsGpuAccelerated(
    JNIEnv *env,
    jobject thiz
) {
    return JNI_FALSE; // Pure CPU compute
}

JNIEXPORT jboolean JNICALL
Java_org_meshllm_node_engine_LlamaNativeBridge_nativeIsEog(
    JNIEnv *env,
    jobject thiz,
    jint token_id
) {
    try {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (!g_model) return JNI_FALSE;
        const struct llama_vocab * vocab = llama_model_get_vocab(g_model);
        if (!vocab) return JNI_FALSE;
        return llama_vocab_is_eog(vocab, (llama_token) token_id) ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

} // extern "C"
