/*
 * Dioramatic Audio Quality Analysis
 *
 * Feeds audio through each algorithm and measures characteristics:
 * - RMS level (is it producing meaningful output?)
 * - Spectral content (pitch shifting working?)
 * - Stereo width (panning working?)
 * - Temporal variation (is the effect evolving over time?)
 * - Algorithm-specific behavioral checks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>

/* Stub host API */
#define MOVE_PLUGIN_API_V1_H
#define AUDIO_FX_API_V2_H
typedef struct host_api_v1 { uint32_t api_version; int sample_rate; int frames_per_block;
    uint8_t *mapped_memory; int audio_out_offset; int audio_in_offset;
    void (*log)(const char *msg); int (*midi_send_internal)(const uint8_t *msg, int len);
    int (*midi_send_external)(const uint8_t *msg, int len); int (*get_clock_status)(void);
    void *mod_emit_value; void *mod_clear_source; void *mod_host_ctx;
    float (*get_bpm)(void); } host_api_v1_t;
#define AUDIO_FX_API_VERSION_2 2
typedef struct audio_fx_api_v2 { uint32_t api_version;
    void* (*create_instance)(const char *d, const char *c);
    void (*destroy_instance)(void *i);
    void (*process_block)(void *i, int16_t *a, int f);
    void (*set_param)(void *i, const char *k, const char *v);
    int (*get_param)(void *i, const char *k, char *b, int l);
    void (*on_midi)(void *i, const uint8_t *m, int l, int s);
} audio_fx_api_v2_t;

#include "../src/dsp/dioramatic.c"

static void tlog(const char *m) { (void)m; }
static float tbpm(void) { return 120.0f; }
static host_api_v1_t th = { .log = tlog, .get_bpm = tbpm };

/* ============================================================================
 * Audio analysis utilities
 * ============================================================================ */

typedef struct {
    float rms_l, rms_r;
    float peak_l, peak_r;
    float stereo_diff;     /* RMS of L-R difference (stereo width indicator) */
    float zero_crossings;  /* normalized zero crossing rate (brightness indicator) */
} block_stats_t;

static block_stats_t analyze_block(int16_t *buf, int frames) {
    block_stats_t s = {0};
    float sum_l2 = 0, sum_r2 = 0, sum_diff2 = 0;
    int zc = 0;
    int16_t prev_l = 0;

    for (int i = 0; i < frames; i++) {
        float l = (float)buf[i * 2] / 32768.0f;
        float r = (float)buf[i * 2 + 1] / 32768.0f;
        sum_l2 += l * l;
        sum_r2 += r * r;
        sum_diff2 += (l - r) * (l - r);

        float al = fabsf(l), ar = fabsf(r);
        if (al > s.peak_l) s.peak_l = al;
        if (ar > s.peak_r) s.peak_r = ar;

        if (i > 0 && ((buf[i * 2] > 0) != (prev_l > 0))) zc++;
        prev_l = buf[i * 2];
    }

    s.rms_l = sqrtf(sum_l2 / (float)frames);
    s.rms_r = sqrtf(sum_r2 / (float)frames);
    s.stereo_diff = sqrtf(sum_diff2 / (float)frames);
    s.zero_crossings = (float)zc / (float)frames;
    return s;
}

/* Run algorithm and collect stats over many blocks */
typedef struct {
    float avg_rms;          /* average RMS across all blocks */
    float max_rms;          /* peak block RMS */
    float avg_stereo_width; /* average L-R difference */
    float temporal_variation; /* std dev of per-block RMS (how much it changes over time) */
    float avg_zero_crossings; /* spectral brightness indicator */
    int blocks_with_output; /* number of blocks that had non-negligible output */
    int total_blocks;
} algo_analysis_t;

static algo_analysis_t run_analysis(audio_fx_api_v2_t *api, int algorithm, int variation,
                                     float activity, float repeats, int num_blocks) {
    void *inst = api->create_instance(NULL, NULL);
    char state[256];
    snprintf(state, sizeof(state),
        "{\"algorithm\":%d,\"variation\":%d,\"activity\":%.2f,\"repeats\":%.2f,"
        "\"shape\":1.0,\"filter\":1.0,\"mix\":1.0,\"space\":0.0,"
        "\"time_div\":0,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
        "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}",
        algorithm, variation, activity, repeats);
    api->set_param(inst, "state", state);

    float rms_sum = 0, rms_max = 0, stereo_sum = 0, zc_sum = 0;
    float *block_rms = (float *)calloc(num_blocks, sizeof(float));
    int output_blocks = 0;

    int16_t buf[256];
    for (int b = 0; b < num_blocks; b++) {
        /* Generate input: 440Hz sine, amplitude 0.5 */
        for (int i = 0; i < 128; i++) {
            float v = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)(b * 128 + i) / 44100.0f);
            int16_t s = (int16_t)(v * 32767.0f);
            buf[i * 2] = s;
            buf[i * 2 + 1] = s;
        }

        api->process_block(inst, buf, 128);
        block_stats_t stats = analyze_block(buf, 128);

        float block_avg_rms = (stats.rms_l + stats.rms_r) * 0.5f;
        block_rms[b] = block_avg_rms;
        rms_sum += block_avg_rms;
        if (block_avg_rms > rms_max) rms_max = block_avg_rms;
        stereo_sum += stats.stereo_diff;
        zc_sum += stats.zero_crossings;
        if (block_avg_rms > 0.001f) output_blocks++;
    }

    algo_analysis_t a = {0};
    a.total_blocks = num_blocks;
    a.blocks_with_output = output_blocks;
    a.avg_rms = rms_sum / (float)num_blocks;
    a.max_rms = rms_max;
    a.avg_stereo_width = stereo_sum / (float)num_blocks;
    a.avg_zero_crossings = zc_sum / (float)num_blocks;

    /* Temporal variation: std dev of per-block RMS */
    float var_sum = 0;
    for (int b = 0; b < num_blocks; b++) {
        float diff = block_rms[b] - a.avg_rms;
        var_sum += diff * diff;
    }
    a.temporal_variation = sqrtf(var_sum / (float)num_blocks);

    free(block_rms);
    api->destroy_instance(inst);
    return a;
}

/* ============================================================================ */

static const char *algo_names[] = {
    "Mosaic", "Seq", "Glide", "Haze", "Tunnel", "Strum",
    "Blocks", "Interrupt", "Arp", "Pattern", "Warp"
};

static int issues_found = 0;

static void check(int ok, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (!ok) {
        printf("  ISSUE: ");
        vprintf(fmt, args);
        printf("\n");
        issues_found++;
    }
    va_end(args);
}

int main(void) {
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&th);
    int num_blocks = 800;

    printf("\n=== Dioramatic Audio Quality Analysis ===\n");
    printf("(800 blocks per test = ~2.3 seconds, 440Hz sine input, mix=1.0)\n\n");

    /* ----------------------------------------------------------------
     * 1. Per-algorithm basic analysis
     * ---------------------------------------------------------------- */
    printf("--- Per-Algorithm Analysis (variation A, activity=0.5, repeats=0.5) ---\n\n");
    printf("%-12s %8s %8s %8s %8s %8s %8s\n",
           "Algorithm", "AvgRMS", "MaxRMS", "Stereo", "Temporal", "ZeroCr", "OutBlks");
    printf("%-12s %8s %8s %8s %8s %8s %8s\n",
           "=========", "======", "======", "======", "========", "======", "=======");

    algo_analysis_t base[11];
    for (int a = 0; a < 11; a++) {
        base[a] = run_analysis(api, a, 0, 0.5f, 0.5f, num_blocks);
        printf("%-12s %8.4f %8.4f %8.4f %8.4f %8.3f %5d/%d\n",
               algo_names[a], base[a].avg_rms, base[a].max_rms,
               base[a].avg_stereo_width, base[a].temporal_variation,
               base[a].avg_zero_crossings, base[a].blocks_with_output, num_blocks);
    }

    printf("\n--- Basic Checks ---\n");
    for (int a = 0; a < 11; a++) {
        check(base[a].avg_rms > 0.001f,
              "%s: avg RMS too low (%.6f) — not producing meaningful audio", algo_names[a], base[a].avg_rms);
        check(base[a].blocks_with_output > num_blocks / 4,
              "%s: only %d/%d blocks have output — effect too sparse", algo_names[a], base[a].blocks_with_output, num_blocks);
        check(base[a].max_rms < 1.0f,
              "%s: max RMS = %.4f — clipping/distortion", algo_names[a], base[a].max_rms);
    }

    /* Grain algorithms should have stereo width (random panning) */
    for (int a = 0; a < 9; a++) {
        check(base[a].avg_stereo_width > 0.0001f,
              "%s: no stereo width (%.6f) — panning may not be working", algo_names[a], base[a].avg_stereo_width);
    }

    /* ----------------------------------------------------------------
     * 2. Variation differences — each variation should sound different
     * ---------------------------------------------------------------- */
    printf("\n--- Variation Comparison (activity=0.7, repeats=0.7) ---\n\n");

    for (int a = 0; a < 11; a++) {
        algo_analysis_t vars[4];
        for (int v = 0; v < 4; v++) {
            vars[v] = run_analysis(api, a, v, 0.7f, 0.7f, num_blocks);
        }

        /* Check that at least SOME metric differs between variations */
        float rms_range = 0, zc_range = 0, stereo_range = 0;
        float rms_min = vars[0].avg_rms, rms_max_v = vars[0].avg_rms;
        float zc_min = vars[0].avg_zero_crossings, zc_max = vars[0].avg_zero_crossings;
        float st_min = vars[0].avg_stereo_width, st_max = vars[0].avg_stereo_width;

        for (int v = 1; v < 4; v++) {
            if (vars[v].avg_rms < rms_min) rms_min = vars[v].avg_rms;
            if (vars[v].avg_rms > rms_max_v) rms_max_v = vars[v].avg_rms;
            if (vars[v].avg_zero_crossings < zc_min) zc_min = vars[v].avg_zero_crossings;
            if (vars[v].avg_zero_crossings > zc_max) zc_max = vars[v].avg_zero_crossings;
            if (vars[v].avg_stereo_width < st_min) st_min = vars[v].avg_stereo_width;
            if (vars[v].avg_stereo_width > st_max) st_max = vars[v].avg_stereo_width;
        }
        rms_range = rms_max_v - rms_min;
        zc_range = zc_max - zc_min;
        stereo_range = st_max - st_min;

        int has_difference = (rms_range > 0.005f) || (zc_range > 0.01f) || (stereo_range > 0.001f);

        printf("%-12s  A:rms=%.4f zc=%.3f  B:rms=%.4f zc=%.3f  C:rms=%.4f zc=%.3f  D:rms=%.4f zc=%.3f  %s\n",
               algo_names[a],
               vars[0].avg_rms, vars[0].avg_zero_crossings,
               vars[1].avg_rms, vars[1].avg_zero_crossings,
               vars[2].avg_rms, vars[2].avg_zero_crossings,
               vars[3].avg_rms, vars[3].avg_zero_crossings,
               has_difference ? "OK" : "SAME?");

        check(has_difference,
              "%s: variations A-D produce nearly identical output", algo_names[a]);
    }

    /* ----------------------------------------------------------------
     * 3. Algorithm-specific behavioral checks
     * ---------------------------------------------------------------- */
    printf("\n--- Algorithm-Specific Behavior ---\n\n");

    /* Mosaic C (all octave up) should have higher zero-crossing rate than B (octave down) */
    {
        algo_analysis_t mosaic_b = run_analysis(api, 0, 1, 0.7f, 0.7f, num_blocks);
        algo_analysis_t mosaic_c = run_analysis(api, 0, 2, 0.7f, 0.7f, num_blocks);
        int octave_ok = mosaic_c.avg_zero_crossings > mosaic_b.avg_zero_crossings;
        printf("Mosaic: var C (octave up) zc=%.3f > var B (octave down) zc=%.3f? %s\n",
               mosaic_c.avg_zero_crossings, mosaic_b.avg_zero_crossings,
               octave_ok ? "YES" : "NO");
        check(octave_ok, "Mosaic C should be brighter (higher pitch) than B");
    }

    /* Haze should have high temporal variation (dense cloud of grains) */
    {
        algo_analysis_t haze = run_analysis(api, 3, 0, 0.9f, 0.7f, num_blocks);
        printf("Haze: high activity temporal_var=%.4f (expect >0.01) %s\n",
               haze.temporal_variation,
               haze.temporal_variation > 0.01f ? "OK" : "LOW");
    }

    /* Activity should increase output density/level */
    {
        algo_analysis_t low = run_analysis(api, 0, 0, 0.1f, 0.5f, num_blocks);
        algo_analysis_t high = run_analysis(api, 0, 0, 0.9f, 0.5f, num_blocks);
        int activity_ok = high.avg_rms > low.avg_rms;
        printf("Mosaic: high activity rms=%.4f > low activity rms=%.4f? %s\n",
               high.avg_rms, low.avg_rms, activity_ok ? "YES" : "NO");
        check(activity_ok, "Higher activity should produce more output");
    }

    /* Repeats should increase output level */
    {
        algo_analysis_t low = run_analysis(api, 0, 0, 0.5f, 0.1f, num_blocks);
        algo_analysis_t high = run_analysis(api, 0, 0, 0.5f, 0.9f, num_blocks);
        int repeats_ok = high.avg_rms > low.avg_rms;
        printf("Mosaic: high repeats rms=%.4f > low repeats rms=%.4f? %s\n",
               high.avg_rms, low.avg_rms, repeats_ok ? "YES" : "NO");
        check(repeats_ok, "Higher repeats should produce louder output");
    }

    /* Pattern (delay) should have rhythmic temporal variation */
    {
        algo_analysis_t pat = run_analysis(api, 9, 0, 0.7f, 0.5f, num_blocks);
        printf("Pattern: temporal_var=%.4f stereo=%.4f (delay should have rhythmic echoes)\n",
               pat.temporal_variation, pat.avg_stereo_width);
    }

    /* Interrupt: should have blocks of silence (dry passthrough) between glitch bursts */
    {
        algo_analysis_t intr = run_analysis(api, 7, 0, 0.5f, 0.8f, num_blocks);
        int sparse = intr.blocks_with_output < num_blocks;
        printf("Interrupt: %d/%d blocks with output (expect sparse, not constant) %s\n",
               intr.blocks_with_output, num_blocks,
               sparse ? "OK" : "TOO DENSE");
    }

    /* ----------------------------------------------------------------
     * 4. Post-processing chain checks
     * ---------------------------------------------------------------- */
    printf("\n--- Post-Processing Chain ---\n\n");

    /* Filter: closed filter should reduce zero crossings */
    {
        void *inst = api->create_instance(NULL, NULL);
        api->set_param(inst, "state", "{\"algorithm\":0,\"variation\":0,\"activity\":0.7,\"repeats\":0.7,"
            "\"shape\":1.0,\"filter\":1.0,\"mix\":1.0,\"space\":0.0,\"time_div\":0}");
        int16_t buf[256];
        float zc_open = 0, zc_closed = 0;

        /* Open filter */
        api->set_param(inst, "filter", "1.0");
        for (int b = 0; b < 400; b++) {
            for (int i = 0; i < 128; i++) {
                float v = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)(b*128+i) / 44100.0f);
                buf[i*2] = buf[i*2+1] = (int16_t)(v * 32767.0f);
            }
            api->process_block(inst, buf, 128);
            if (b >= 200) zc_open += analyze_block(buf, 128).zero_crossings;
        }
        /* Close filter */
        api->set_param(inst, "filter", "0.1");
        for (int b = 0; b < 400; b++) {
            for (int i = 0; i < 128; i++) {
                float v = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)((b+400)*128+i) / 44100.0f);
                buf[i*2] = buf[i*2+1] = (int16_t)(v * 32767.0f);
            }
            api->process_block(inst, buf, 128);
            if (b >= 200) zc_closed += analyze_block(buf, 128).zero_crossings;
        }
        int filter_ok = zc_closed < zc_open;
        printf("Filter: open zc=%.1f, closed zc=%.1f %s\n",
               zc_open, zc_closed, filter_ok ? "OK (closed is darker)" : "ISSUE");
        check(filter_ok, "Closed filter should reduce brightness");
        api->destroy_instance(inst);
    }

    /* Reverb: space=1.0 should increase RMS and temporal smearing */
    {
        algo_analysis_t dry = run_analysis(api, 0, 0, 0.5f, 0.5f, num_blocks);

        void *inst = api->create_instance(NULL, NULL);
        api->set_param(inst, "state", "{\"algorithm\":0,\"variation\":0,\"activity\":0.5,\"repeats\":0.5,"
            "\"shape\":1.0,\"filter\":1.0,\"mix\":1.0,\"space\":1.0,\"time_div\":0}");
        int16_t buf[256];
        float rms_sum = 0;
        for (int b = 0; b < num_blocks; b++) {
            for (int i = 0; i < 128; i++) {
                float v = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)(b*128+i) / 44100.0f);
                buf[i*2] = buf[i*2+1] = (int16_t)(v * 32767.0f);
            }
            api->process_block(inst, buf, 128);
            block_stats_t s = analyze_block(buf, 128);
            rms_sum += (s.rms_l + s.rms_r) * 0.5f;
        }
        float wet_rms = rms_sum / (float)num_blocks;
        int reverb_ok = wet_rms > dry.avg_rms;
        printf("Reverb: space=1.0 rms=%.4f > space=0.0 rms=%.4f? %s\n",
               wet_rms, dry.avg_rms, reverb_ok ? "YES (reverb adds energy)" : "NO");
        check(reverb_ok, "Reverb should add energy to the signal");
        api->destroy_instance(inst);
    }

    /* ---------------------------------------------------------------- */
    printf("\n=== Analysis Complete: %d issues found ===\n\n", issues_found);
    return issues_found > 0 ? 1 : 0;
}
