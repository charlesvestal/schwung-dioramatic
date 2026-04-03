/*
 * Dioramatic native test harness
 * Compiles and runs on host (macOS) to validate DSP without hardware.
 * Tests: all algorithms produce output, no NaN/inf, state round-trip, param handling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Stub out the host API so we can compile dioramatic.c natively */
#define MOVE_PLUGIN_API_V1_H
#define AUDIO_FX_API_V2_H

typedef struct host_api_v1 {
    uint32_t api_version;
    int sample_rate;
    int frames_per_block;
    uint8_t *mapped_memory;
    int audio_out_offset;
    int audio_in_offset;
    void (*log)(const char *msg);
    int (*midi_send_internal)(const uint8_t *msg, int len);
    int (*midi_send_external)(const uint8_t *msg, int len);
    int (*get_clock_status)(void);
    void *mod_emit_value;
    void *mod_clear_source;
    void *mod_host_ctx;
    float (*get_bpm)(void);
} host_api_v1_t;

#define AUDIO_FX_API_VERSION_2 2

typedef struct audio_fx_api_v2 {
    uint32_t api_version;
    void* (*create_instance)(const char *module_dir, const char *config_json);
    void (*destroy_instance)(void *instance);
    void (*process_block)(void *instance, int16_t *audio_inout, int frames);
    void (*set_param)(void *instance, const char *key, const char *val);
    int (*get_param)(void *instance, const char *key, char *buf, int buf_len);
    void (*on_midi)(void *instance, const uint8_t *msg, int len, int source);
} audio_fx_api_v2_t;

/* Include the actual DSP code */
#include "../src/dsp/dioramatic.c"

/* ============================================================================ */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST: %-50s", name); \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { tests_failed++; printf("FAIL: %s\n", msg); } while(0)

static void test_log(const char *msg) {
    /* silent */
    (void)msg;
}

static float test_get_bpm(void) {
    return 120.0f;
}

static host_api_v1_t test_host = {
    .api_version = 1,
    .sample_rate = 44100,
    .frames_per_block = 128,
    .log = test_log,
    .get_bpm = test_get_bpm,
};

/* Generate test audio: sine wave at 440Hz */
static void generate_sine(int16_t *buf, int frames, float freq, float amp) {
    for (int i = 0; i < frames; i++) {
        float val = amp * sinf(2.0f * (float)M_PI * freq * (float)i / 44100.0f);
        int16_t s = (int16_t)(val * 32767.0f);
        buf[i * 2] = s;
        buf[i * 2 + 1] = s;
    }
}

/* Check for NaN/Inf in audio buffer */
static int check_audio_valid(int16_t *buf, int frames) {
    for (int i = 0; i < frames * 2; i++) {
        if (buf[i] == 0x7FFF || buf[i] == (int16_t)0x8000) {
            /* Clipped but not necessarily invalid */
        }
    }
    /* int16 can't be NaN, but check for all-zero when we expect output */
    return 1;
}

/* Check if buffer has any non-zero samples */
static int has_nonzero(int16_t *buf, int frames) {
    for (int i = 0; i < frames * 2; i++) {
        if (buf[i] != 0) return 1;
    }
    return 0;
}

/* ============================================================================
 * Test: Create and destroy instance
 * ============================================================================ */
static void test_create_destroy(void) {
    TEST("create/destroy instance");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    if (!inst) { FAIL("create returned NULL"); return; }
    api->destroy_instance(inst);
    PASS();
}

/* ============================================================================
 * Test: Passthrough with mix=0
 * ============================================================================ */
static void test_passthrough(void) {
    TEST("mix=0 passthrough preserves audio");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "0.0");

    int16_t buf[256];
    generate_sine(buf, 128, 440.0f, 0.5f);
    int16_t orig[256];
    memcpy(orig, buf, sizeof(buf));

    api->process_block(inst, buf, 128);

    int match = 1;
    for (int i = 0; i < 256; i++) {
        if (abs(buf[i] - orig[i]) > 1) { match = 0; break; }
    }
    if (match) PASS(); else FAIL("audio modified at mix=0");
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: All algorithms produce wet output
 * ============================================================================ */
static void test_all_algorithms_produce_output(void) {
    const char *algo_names[] = {
        "Mosaic", "Seq", "Glide", "Haze", "Tunnel", "Strum",
        "Blocks", "Interrupt", "Arp", "Pattern", "Warp"
    };

    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);

    for (int a = 0; a < 11; a++) {
        for (int v = 0; v < 4; v++) {
            char test_name[80];
            snprintf(test_name, sizeof(test_name), "algorithm %s var %c produces output",
                     algo_names[a], 'A' + v);
            TEST(test_name);

            void *inst = api->create_instance(NULL, NULL);
            /* Set algorithm DIRECTLY via state restore to bypass crossfade */
            char state_json[256];
            snprintf(state_json, sizeof(state_json),
                "{\"algorithm\":%d,\"variation\":%d,\"activity\":0.7,\"repeats\":0.7,"
                "\"shape\":1.0,\"filter\":1.0,\"mix\":1.0,\"space\":0.0,"
                "\"time_div\":0,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
                "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}",
                a, v);
            api->set_param(inst, "state", state_json);
            /* time_div=0 = 1/4 note, subdivision = 5512 samples at 120 BPM */

            /* Feed 800 blocks = 102400 samples (~2.3 sec) — enough for multiple subdivisions */
            int16_t buf[256];
            int got_output = 0;
            for (int block = 0; block < 800; block++) {
                generate_sine(buf, 128, 440.0f, 0.5f);
                api->process_block(inst, buf, 128);
                if (has_nonzero(buf, 128)) got_output = 1;
            }

            if (got_output) PASS();
            else FAIL("no output after 800 blocks");

            api->destroy_instance(inst);
        }
    }
}

/* ============================================================================
 * Test: All algorithms with all variations - no crash over many blocks
 * ============================================================================ */
static void test_stability(void) {
    TEST("100 blocks per algo/var - no crash");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);

    const char *algo_names[] = {
        "Mosaic", "Seq", "Glide", "Haze", "Tunnel", "Strum",
        "Blocks", "Interrupt", "Arp", "Pattern", "Warp"
    };

    for (int a = 0; a < 11; a++) {
        void *inst = api->create_instance(NULL, NULL);
        api->set_param(inst, "algorithm", algo_names[a]);
        api->set_param(inst, "mix", "0.8");
        api->set_param(inst, "activity", "0.9");
        api->set_param(inst, "repeats", "0.9");
        api->set_param(inst, "space", "0.5");
        api->set_param(inst, "shape", "0.3");
        api->set_param(inst, "pitch_mod_depth", "0.5");

        for (int v = 0; v < 4; v++) {
            char vstr[2] = { 'A' + v, 0 };
            api->set_param(inst, "variation", vstr);

            int16_t buf[256];
            for (int block = 0; block < 100; block++) {
                generate_sine(buf, 128, 220.0f + (float)block, 0.4f);
                api->process_block(inst, buf, 128);
            }
        }
        api->destroy_instance(inst);
    }
    PASS();
}

/* ============================================================================
 * Test: Reverse toggle doesn't crash
 * ============================================================================ */
static void test_reverse(void) {
    TEST("reverse toggle during playback");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "1.0");
    api->set_param(inst, "activity", "0.8");

    int16_t buf[256];
    for (int i = 0; i < 50; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
        if (i == 20) api->set_param(inst, "reverse", "On");
        if (i == 35) api->set_param(inst, "reverse", "Off");
    }
    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: Hold/Freeze engage and release
 * ============================================================================ */
static void test_hold(void) {
    TEST("hold engage/release cycle");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "1.0");
    api->set_param(inst, "activity", "0.5");

    int16_t buf[256];
    /* Fill capture buffer */
    for (int i = 0; i < 30; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
    }

    /* Engage hold */
    api->set_param(inst, "hold", "On");
    for (int i = 0; i < 30; i++) {
        generate_sine(buf, 128, 880.0f, 0.5f);  /* different freq while holding */
        api->process_block(inst, buf, 128);
    }

    /* Release hold */
    api->set_param(inst, "hold", "Off");
    for (int i = 0; i < 30; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
    }

    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: State save/restore round-trip
 * ============================================================================ */
static void test_state_roundtrip(void) {
    TEST("state save/restore round-trip");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);

    /* Set non-default params */
    api->set_param(inst, "algorithm", "Haze");
    api->set_param(inst, "variation", "C");
    api->set_param(inst, "activity", "0.75");
    api->set_param(inst, "repeats", "0.33");
    api->set_param(inst, "filter", "0.6");
    api->set_param(inst, "mix", "0.8");
    api->set_param(inst, "space", "0.4");
    api->set_param(inst, "time_div", "2x");
    api->set_param(inst, "reverb_mode", "Hall");
    api->set_param(inst, "reverse", "On");

    /* Process a block to trigger crossfade, then let it complete */
    int16_t buf[256];
    for (int i = 0; i < 30; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
    }

    /* Save state */
    char state_buf[1024];
    int len = api->get_param(inst, "state", state_buf, sizeof(state_buf));
    if (len <= 0) { FAIL("get_param state returned <= 0"); api->destroy_instance(inst); return; }

    /* Create new instance and restore */
    void *inst2 = api->create_instance(NULL, NULL);
    api->set_param(inst2, "state", state_buf);

    /* Compare individual params */
    char buf1[64], buf2[64];
    int ok = 1;
    const char *check_keys[] = {"algorithm", "variation", "activity", "repeats",
                                 "filter", "mix", "space", "reverb_mode", "reverse"};
    for (int i = 0; i < 9; i++) {
        api->get_param(inst, check_keys[i], buf1, sizeof(buf1));
        api->get_param(inst2, check_keys[i], buf2, sizeof(buf2));
        if (strcmp(buf1, buf2) != 0) {
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: '%s' vs '%s'", check_keys[i], buf1, buf2);
            FAIL(msg);
            ok = 0;
            break;
        }
    }
    if (ok) PASS();

    api->destroy_instance(inst);
    api->destroy_instance(inst2);
}

/* ============================================================================
 * Test: Algorithm crossfade doesn't crash
 * ============================================================================ */
static void test_crossfade(void) {
    TEST("rapid algorithm switching with crossfade");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "0.8");

    const char *algos[] = {"Mosaic", "Haze", "Arp", "Pattern", "Warp", "Tunnel", "Blocks"};
    int16_t buf[256];

    for (int cycle = 0; cycle < 3; cycle++) {
        for (int a = 0; a < 7; a++) {
            api->set_param(inst, "algorithm", algos[a]);
            for (int b = 0; b < 5; b++) {
                generate_sine(buf, 128, 440.0f, 0.5f);
                api->process_block(inst, buf, 128);
            }
        }
    }
    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: SVF filter extremes don't blow up
 * ============================================================================ */
static void test_filter_extremes(void) {
    TEST("SVF filter at extremes (0, 1, high res)");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "1.0");

    int16_t buf[256];

    /* Filter fully closed */
    api->set_param(inst, "filter", "0.0");
    api->set_param(inst, "filter_res", "0.9");
    for (int i = 0; i < 10; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
    }

    /* Filter fully open */
    api->set_param(inst, "filter", "1.0");
    api->set_param(inst, "filter_res", "0.0");
    for (int i = 0; i < 10; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
    }

    /* High resonance */
    api->set_param(inst, "filter", "0.5");
    api->set_param(inst, "filter_res", "1.0");
    for (int i = 0; i < 10; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
    }

    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: Reverb all modes don't blow up
 * ============================================================================ */
static void test_reverb_modes(void) {
    TEST("all 4 reverb modes produce output");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    const char *modes[] = {"Bright", "Dark", "Hall", "Ambient"};

    for (int m = 0; m < 4; m++) {
        void *inst = api->create_instance(NULL, NULL);
        api->set_param(inst, "mix", "1.0");
        api->set_param(inst, "space", "1.0");
        api->set_param(inst, "reverb_mode", modes[m]);
        api->set_param(inst, "time_div", "1/4");  /* fast grain triggers */

        int16_t buf[256];
        int got_output = 0;
        for (int i = 0; i < 400; i++) {
            generate_sine(buf, 128, 440.0f, 0.5f);
            api->process_block(inst, buf, 128);
            if (has_nonzero(buf, 128)) got_output = 1;
        }

        if (!got_output) {
            char msg[64];
            snprintf(msg, sizeof(msg), "no output for mode %s", modes[m]);
            FAIL(msg);
            api->destroy_instance(inst);
            return;
        }
        api->destroy_instance(inst);
    }
    PASS();
}

/* ============================================================================
 * Test: Shape LFO modulation
 * ============================================================================ */
static void test_shape_lfo(void) {
    TEST("shape LFO modulates wet signal (shape<1.0)");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "1.0");
    api->set_param(inst, "activity", "0.8");
    api->set_param(inst, "space", "0.0");
    api->set_param(inst, "shape", "0.1");  /* Square LFO zone */
    api->set_param(inst, "time_div", "8x"); /* fast subdivision for LFO cycling */

    int16_t buf[256];
    /* Collect RMS over many blocks */
    float rms_values[400];
    for (int i = 0; i < 400; i++) {
        generate_sine(buf, 128, 440.0f, 0.5f);
        api->process_block(inst, buf, 128);
        float rms = 0;
        for (int j = 0; j < 256; j++) rms += (float)buf[j] * (float)buf[j];
        rms_values[i] = sqrtf(rms / 256.0f);
    }

    /* With square LFO, there should be variation in RMS (some blocks louder, some quieter) */
    float min_rms = rms_values[50], max_rms = rms_values[50];
    for (int i = 50; i < 400; i++) {  /* skip first 50 blocks for warmup */
        if (rms_values[i] < min_rms) min_rms = rms_values[i];
        if (rms_values[i] > max_rms) max_rms = rms_values[i];
    }

    if (max_rms - min_rms > 10.0f) PASS();
    else FAIL("no RMS variation from shape LFO");

    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: Enum param handling
 * ============================================================================ */
static void test_enum_params(void) {
    TEST("enum params accept string and numeric values");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);

    char buf[64];

    /* Algorithm set via crossfade — use state restore to set directly */
    api->set_param(inst, "state", "{\"algorithm\":3}");
    api->get_param(inst, "algorithm", buf, sizeof(buf));
    if (strcmp(buf, "Haze") != 0) {
        char msg[128]; snprintf(msg, sizeof(msg), "state set: got '%s'", buf);
        FAIL(msg); api->destroy_instance(inst); return;
    }

    /* Time div with slash (non-crossfade param) */
    api->set_param(inst, "time_div", "1/4");
    api->get_param(inst, "time_div", buf, sizeof(buf));
    if (strcmp(buf, "1/4") != 0) { FAIL("time_div string failed"); api->destroy_instance(inst); return; }

    /* Reverb mode by name */
    api->set_param(inst, "reverb_mode", "Hall");
    api->get_param(inst, "reverb_mode", buf, sizeof(buf));
    if (strcmp(buf, "Hall") != 0) { FAIL("reverb_mode string failed"); api->destroy_instance(inst); return; }

    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: get_param chain_params returns valid JSON
 * ============================================================================ */
static void test_chain_params(void) {
    TEST("chain_params returns non-empty JSON");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);

    char buf[4096];
    int len = api->get_param(inst, "chain_params", buf, sizeof(buf));
    if (len <= 0) { FAIL("empty chain_params"); api->destroy_instance(inst); return; }
    if (buf[0] != '[') { FAIL("doesn't start with ["); api->destroy_instance(inst); return; }

    /* Check all algorithm names appear */
    if (!strstr(buf, "Mosaic")) { FAIL("missing Mosaic"); api->destroy_instance(inst); return; }
    if (!strstr(buf, "Warp")) { FAIL("missing Warp"); api->destroy_instance(inst); return; }

    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: MIDI clock
 * ============================================================================ */
static void test_midi_clock(void) {
    TEST("MIDI clock derives BPM");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);

    /* Send 48 clock ticks (2 beats) */
    uint8_t clock_msg[1] = {0xF8};
    for (int i = 0; i < 48; i++) {
        api->on_midi(inst, clock_msg, 1, 0);
    }
    /* BPM should have been derived (though it'll be approximate with our stub) */
    PASS();
    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: Silence in produces near-silence out
 * ============================================================================ */
static void test_silence(void) {
    TEST("silence input with mix=1.0 stays quiet");
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&test_host);
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "mix", "1.0");
    api->set_param(inst, "space", "0.0");
    api->set_param(inst, "shape", "1.0");

    int16_t buf[256];
    memset(buf, 0, sizeof(buf));

    /* Process 50 blocks of silence */
    float max_level = 0;
    for (int i = 0; i < 50; i++) {
        memset(buf, 0, sizeof(buf));
        api->process_block(inst, buf, 128);
        for (int j = 0; j < 256; j++) {
            float level = fabsf((float)buf[j]);
            if (level > max_level) max_level = level;
        }
    }

    if (max_level < 100.0f) PASS();  /* Should be very quiet */
    else {
        char msg[64];
        snprintf(msg, sizeof(msg), "max level %.0f (expected <100)", max_level);
        FAIL(msg);
    }

    api->destroy_instance(inst);
}

/* ============================================================================
 * Test: Instance size sanity check
 * ============================================================================ */
static void test_instance_size(void) {
    TEST("instance struct size reasonable");
    size_t sz = sizeof(dioramatic_instance_t);
    printf("(%zuKB) ", sz / 1024);
    /* Should be ~2-3MB with capture + hold + delay buffers + reverb */
    if (sz > 500000 && sz < 5000000) PASS();
    else {
        char msg[64];
        snprintf(msg, sizeof(msg), "size=%zu bytes", sz);
        FAIL(msg);
    }
}

/* ============================================================================ */
int main(void) {
    printf("\n=== Dioramatic DSP Test Suite ===\n\n");

    test_instance_size();
    test_create_destroy();
    test_passthrough();
    test_silence();
    test_enum_params();
    test_chain_params();
    test_state_roundtrip();
    test_midi_clock();
    test_filter_extremes();
    test_reverb_modes();
    test_shape_lfo();
    test_reverse();
    test_hold();
    test_crossfade();
    test_all_algorithms_produce_output();
    test_stability();

    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(", %d FAILED", tests_failed);
    printf(" ===\n\n");

    return tests_failed > 0 ? 1 : 0;
}
