/*
 * Dioramatic — Granular Shimmer Effect
 *
 * A single cohesive instrument with 8 musical controls:
 * Space, Shimmer, Smear, Warmth, Drift, Sustain, Scatter, Mix
 *
 * DSP: grain cloud + shimmer grains → SVF filter → FDN reverb with
 * shimmer pitch-shift feedback, modulated allpass, stereo decorrelation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_fx_api_v2.h"

#define SAMPLE_RATE 44100
#define CAPTURE_SAMPLES (SAMPLE_RATE * 2)
#define MAX_GRAINS 32
#define ENV_TABLE_SIZE 256
#define FDN_LINES 4
#define FDN_MAX_DELAY 8192
#define FDN_NUM_AP 4
#define FDN_AP_MAX 512
#define SHIMMER_BUF_SIZE 4096
#define SHIMMER_GRAIN_SIZE 1024

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Types
 * ============================================================================ */

typedef struct { float l, r; } stereo_sample_t;

typedef struct {
    stereo_sample_t buffer[CAPTURE_SAMPLES];
    int write_pos;
} capture_buffer_t;

typedef struct {
    int active;
    int start;
    int length;
    float position;
    float speed;
    float detune;
    float amplitude;
    float env_phase;
    float env_inc;
    float pan_l, pan_r;
    float lp_state;    /* per-grain one-pole LP for darkening over lifetime */
    float lp_target;   /* cutoff sweeps from 1.0 (bright) toward this as grain ages */
} grain_t;

typedef struct { float ic1, ic2; } svf_state_t;

typedef struct {
    float lines[FDN_LINES][FDN_MAX_DELAY];
    int write_pos[FDN_LINES];
    float lp_state[FDN_LINES];
    float ap_buf[FDN_NUM_AP][FDN_AP_MAX];
    int ap_pos[FDN_NUM_AP];
    float mod_phase[FDN_LINES];
    /* Shimmer pitch shift */
    float shimmer_buf[SHIMMER_BUF_SIZE];
    int shimmer_write_pos;
    float shimmer_read_phase;
    /* Feedback allpass for lush tail */
    float fb_ap_buf[2][512];
    int fb_ap_pos[2];
    float fb_ap_mod_phase[2];
    /* Stereo decorrelation */
    float stereo_buf[1024];
    int stereo_pos;
    /* Sparkle shelf */
    float sparkle_state_l, sparkle_state_r;
} fdn_reverb_t;

typedef struct {
    /* 8 musical parameters (0-1) */
    float space;
    float shimmer;
    float smear;
    float warmth;
    float drift;
    float sustain;
    float scatter;
    float mix;

    /* Capture buffer */
    capture_buffer_t capture;

    /* Grain engine */
    grain_t grains[MAX_GRAINS];
    float env_table[ENV_TABLE_SIZE];
    int cloud_timer;
    int shimmer_grain_timer;
    uint32_t rng_state;

    /* LFO */
    float lfo_phase;

    /* SVF filter */
    svf_state_t svf_l, svf_r;

    /* FDN reverb */
    fdn_reverb_t reverb;

    /* DC blocker */
    float dc_prev_in_l, dc_prev_in_r, dc_prev_out_l, dc_prev_out_r;

} dioramatic_instance_t;

/* ============================================================================
 * Globals
 * ============================================================================ */

static const host_api_v1_t *g_host = NULL;
static audio_fx_api_v2_t g_fx_api_v2;

/* ============================================================================
 * Utilities
 * ============================================================================ */

static inline uint32_t rng_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static inline float rng_float(uint32_t *state) {
    return (float)(rng_next(state) >> 8) / 16777216.0f;
}

static int json_get_number(const char *json, const char *key, float *out) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '"') p++;
    *out = (float)atof(p);
    return 0;
}

/* ============================================================================
 * Grain trigger
 * ============================================================================ */

static void trigger_grain(dioramatic_instance_t *inst, float speed, float amp,
                          float len_ms, float pan_width) {
    grain_t *g = NULL;
    for (int i = 0; i < MAX_GRAINS; i++) {
        if (!inst->grains[i].active) { g = &inst->grains[i]; break; }
    }
    if (!g) return;

    int len = (int)(SAMPLE_RATE * len_ms / 1000.0f);
    if (len < 128) len = 128;
    if (len > CAPTURE_SAMPLES - 1) len = CAPTURE_SAMPLES - 1;

    int recent = (int)(SAMPLE_RATE * 0.05f + rng_float(&inst->rng_state) * SAMPLE_RATE * 0.4f);
    g->active = 1;
    g->start = (inst->capture.write_pos - recent + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
    g->length = len;
    g->position = 0.0f;
    g->speed = speed;
    g->amplitude = amp;
    g->env_phase = 0.0f;
    g->env_inc = 1.0f / (float)len;
    g->detune = 1.0f + (rng_float(&inst->rng_state) - 0.5f) * 0.009f;

    float pan_pos = (rng_float(&inst->rng_state) - 0.5f) * pan_width * 0.7f;
    g->pan_l = sqrtf(0.5f - pan_pos * 0.5f);
    g->pan_r = sqrtf(0.5f + pan_pos * 0.5f);

    /* Per-grain darkening filter: starts bright, sweeps darker as grain ages.
       At high sustain, grains darken significantly — like a piano string
       losing high harmonics as it rings out. */
    g->lp_state = 0.0f;
    g->lp_target = 0.3f - inst->sustain * 0.25f;  /* 0.3 (mild) to 0.05 (very dark) at end */
}

/* ============================================================================
 * SVF lowpass
 * ============================================================================ */

static inline float svf_tick(svf_state_t *s, float input, float a1, float a2, float a3) {
    float v3 = input - s->ic2;
    float v1 = a1 * s->ic1 + a2 * v3;
    float v2 = s->ic2 + a2 * s->ic1 + a3 * v3;
    s->ic1 = 2.0f * v1 - s->ic1;
    s->ic2 = 2.0f * v2 - s->ic2;
    return v2;
}

/* ============================================================================
 * FDN reverb with shimmer, feedback allpass, stereo decorrelation
 * ============================================================================ */

static void fdn_process(fdn_reverb_t *rev, float in_l, float in_r,
                         float *out_l, float *out_r,
                         const int *lengths, float feedback, float damping,
                         float shim_amount, float mod_depth) {

    /* Input diffusion */
    float diff = (in_l + in_r) * 0.5f;
    static const int ap_lens[FDN_NUM_AP] = {241, 173, 419, 313};
    for (int i = 0; i < FDN_NUM_AP; i++) {
        float delayed = rev->ap_buf[i][rev->ap_pos[i]];
        float ap_out = delayed - 0.5f * diff;
        rev->ap_buf[i][rev->ap_pos[i]] = diff + 0.5f * ap_out;
        diff = ap_out;
        rev->ap_pos[i] = (rev->ap_pos[i] + 1) % ap_lens[i];
    }

    /* Read delay lines with modulation */
    float taps[FDN_LINES];
    static const float mod_rates[FDN_LINES] = {0.37f, 0.47f, 0.31f, 0.53f};
    for (int i = 0; i < FDN_LINES; i++) {
        float mod_off = sinf(2.0f * (float)M_PI * rev->mod_phase[i]) * mod_depth;
        float frac_delay = (float)lengths[i] + mod_off;
        if (frac_delay < 1.0f) frac_delay = 1.0f;
        if (frac_delay >= (float)(FDN_MAX_DELAY - 1)) frac_delay = (float)(FDN_MAX_DELAY - 2);
        int rd_int = (int)frac_delay;
        float rd_frac = frac_delay - (float)rd_int;
        int rp0 = (rev->write_pos[i] - rd_int + FDN_MAX_DELAY) & (FDN_MAX_DELAY - 1);
        int rp1 = (rp0 - 1 + FDN_MAX_DELAY) & (FDN_MAX_DELAY - 1);
        taps[i] = rev->lines[i][rp0] * (1.0f - rd_frac) + rev->lines[i][rp1] * rd_frac;
        rev->mod_phase[i] += mod_rates[i] / (float)SAMPLE_RATE;
        if (rev->mod_phase[i] >= 1.0f) rev->mod_phase[i] -= 1.0f;
    }

    /* Hadamard mixing */
    float mx[FDN_LINES];
    mx[0] = 0.5f * (taps[0] + taps[1] + taps[2] + taps[3]);
    mx[1] = 0.5f * (taps[0] - taps[1] + taps[2] - taps[3]);
    mx[2] = 0.5f * (taps[0] + taps[1] - taps[2] - taps[3]);
    mx[3] = 0.5f * (taps[0] - taps[1] - taps[2] + taps[3]);

    /* Shimmer pitch shift (octave up via two crossfaded grains) */
    float fb_mono = (mx[0] + mx[1] + mx[2] + mx[3]) * 0.25f;
    rev->shimmer_buf[rev->shimmer_write_pos] = fb_mono;
    rev->shimmer_write_pos = (rev->shimmer_write_pos + 1) & (SHIMMER_BUF_SIZE - 1);

    int rd_a = (int)rev->shimmer_read_phase;
    float fr_a = rev->shimmer_read_phase - (float)rd_a;
    float s_a = rev->shimmer_buf[rd_a & (SHIMMER_BUF_SIZE - 1)] * (1.0f - fr_a)
              + rev->shimmer_buf[(rd_a + 1) & (SHIMMER_BUF_SIZE - 1)] * fr_a;
    float e_a = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * fmodf(rev->shimmer_read_phase / (float)SHIMMER_GRAIN_SIZE, 1.0f)));

    float ph_b = rev->shimmer_read_phase + (float)(SHIMMER_GRAIN_SIZE / 2);
    int rd_b = (int)ph_b;
    float fr_b = ph_b - (float)rd_b;
    float s_b = rev->shimmer_buf[rd_b & (SHIMMER_BUF_SIZE - 1)] * (1.0f - fr_b)
              + rev->shimmer_buf[(rd_b + 1) & (SHIMMER_BUF_SIZE - 1)] * fr_b;
    float e_b = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * fmodf(ph_b / (float)SHIMMER_GRAIN_SIZE, 1.0f)));

    float shim_out = s_a * e_a + s_b * e_b;
    rev->shimmer_read_phase += 2.0f;
    /* Wrap to prevent float precision loss */
    while (rev->shimmer_read_phase >= (float)SHIMMER_BUF_SIZE)
        rev->shimmer_read_phase -= (float)SHIMMER_BUF_SIZE;
    /* Keep read phase near write */
    float dist = (float)rev->shimmer_write_pos - rev->shimmer_read_phase;
    if (dist < 0) dist += (float)SHIMMER_BUF_SIZE;
    if (dist > (float)(SHIMMER_BUF_SIZE - SHIMMER_GRAIN_SIZE)) {
        rev->shimmer_read_phase = (float)rev->shimmer_write_pos - (float)(SHIMMER_BUF_SIZE / 2);
        if (rev->shimmer_read_phase < 0) rev->shimmer_read_phase += (float)SHIMMER_BUF_SIZE;
    }

    /* Feedback allpass for lush tail density */
    float fb_diff = fb_mono;
    static const int fb_ap_lens[2] = {347, 461};
    static const float fb_ap_rates[2] = {0.23f, 0.29f};
    for (int ap = 0; ap < 2; ap++) {
        float mod = sinf(2.0f * (float)M_PI * rev->fb_ap_mod_phase[ap]) * 3.0f;
        int rd = (rev->fb_ap_pos[ap] - fb_ap_lens[ap] - (int)mod + 512) & 511;
        float delayed = rev->fb_ap_buf[ap][rd];
        float ap_out = delayed - 0.45f * fb_diff;
        rev->fb_ap_buf[ap][rev->fb_ap_pos[ap]] = fb_diff + 0.45f * ap_out;
        fb_diff = ap_out;
        rev->fb_ap_pos[ap] = (rev->fb_ap_pos[ap] + 1) & 511;
        rev->fb_ap_mod_phase[ap] += fb_ap_rates[ap] / (float)SAMPLE_RATE;
        if (rev->fb_ap_mod_phase[ap] >= 1.0f) rev->fb_ap_mod_phase[ap] -= 1.0f;
    }

    /* Write feedback + shimmer + diffusion back to delay lines */
    for (int i = 0; i < FDN_LINES; i++) {
        float fb = mx[i] * feedback;
        rev->lp_state[i] += damping * (fb - rev->lp_state[i]);
        fb = rev->lp_state[i] + shim_out * shim_amount + fb_diff * 0.3f;
        fb = tanhf(fb * 1.1f) * 0.9f;
        rev->lines[i][rev->write_pos[i]] = diff + fb;
        rev->write_pos[i] = (rev->write_pos[i] + 1) & (FDN_MAX_DELAY - 1);
    }

    /* Stereo output with decorrelation */
    float raw_l = (taps[0] + taps[2]) * 0.45f;
    float raw_r = (taps[1] + taps[3]) * 0.45f;

    /* Right channel decorrelation (~8ms) */
    rev->stereo_buf[rev->stereo_pos] = raw_r;
    int stereo_rd = (rev->stereo_pos - 353 + 1024) & 1023;
    raw_r = rev->stereo_buf[stereo_rd];
    rev->stereo_pos = (rev->stereo_pos + 1) & 1023;

    /* Sparkle shelf */
    float hf_l = raw_l - rev->sparkle_state_l;
    rev->sparkle_state_l += 0.15f * hf_l;
    float hf_r = raw_r - rev->sparkle_state_r;
    rev->sparkle_state_r += 0.15f * hf_r;

    *out_l = raw_l + hf_l * 0.2f;
    *out_r = raw_r + hf_r * 0.2f;
}

/* ============================================================================
 * API Implementation
 * ============================================================================ */

static void *v2_create_instance(const char *module_dir, const char *config_json) {
    (void)module_dir; (void)config_json;
    dioramatic_instance_t *inst = calloc(1, sizeof(dioramatic_instance_t));
    if (!inst) return NULL;

    inst->space = 0.55f;
    inst->shimmer = 0.35f;
    inst->smear = 0.40f;
    inst->warmth = 0.45f;
    inst->drift = 0.35f;
    inst->sustain = 0.50f;
    inst->scatter = 0.40f;
    inst->mix = 0.60f;

    inst->rng_state = 12345;

    /* Hann envelope table */
    for (int i = 0; i < ENV_TABLE_SIZE; i++) {
        float t = (float)i / (float)(ENV_TABLE_SIZE - 1);
        inst->env_table[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * t));
    }

    if (g_host && g_host->log) g_host->log("dioramatic: instance created");
    return inst;
}

static void v2_destroy_instance(void *instance) {
    if (instance) free(instance);
}

static void v2_process_block(void *instance, int16_t *audio_inout, int frames) {
    if (!instance || !audio_inout) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    /* Derive DSP params from musical knobs */
    float rev_send = 0.2f + inst->space * 0.75f;
    int fdn_lengths[4] = {
        1087 + (int)(inst->space * 2500.0f),
        1283 + (int)(inst->space * 2900.0f),
        1447 + (int)(inst->space * 3400.0f),
        1663 + (int)(inst->space * 3900.0f)
    };
    /* Scatter controls overall grain density + stereo spread.
       Low = few focused grains. High = many grains scattered wide. */
    float density = 0.2f + inst->scatter * 0.8f;  /* 0.2 to 1.0 */
    float pan_width_val = inst->scatter;

    float shim_feedback = inst->shimmer * 0.12f;
    int shim_grain_rate = inst->shimmer > 0.1f
        ? (int)(8000.0f / ((0.3f + inst->shimmer * 4.0f) * density)) : 999999;
    if (shim_grain_rate < 128) shim_grain_rate = 128;
    int cloud_rate = inst->smear > 0.05f
        ? (int)(6000.0f / ((0.2f + inst->smear * 6.0f) * density)) : 999999;
    if (cloud_rate < 128) cloud_rate = 128;
    float cloud_len_ms = 15.0f + inst->smear * 80.0f;
    float cutoff_hz = 200.0f * powf(100.0f, 1.0f - inst->warmth);
    if (cutoff_hz > 20000.0f) cutoff_hz = 20000.0f;
    float rev_damping = 0.15f + inst->warmth * 0.5f;
    float pitch_mod_depth = inst->drift * 0.15f;
    float pitch_mod_rate = 0.05f + inst->drift * 0.4f;
    float rev_mod_depth = 4.0f + inst->drift * 20.0f;
    float rev_feedback = fminf(0.97f, 0.65f + inst->sustain * 0.32f);
    float sustain_grain_len = 50.0f + inst->sustain * 350.0f;
    float pan_width = pan_width_val;

    /* SVF coefficients */
    float svf_g = tanf((float)M_PI * cutoff_hz / (float)SAMPLE_RATE);
    float svf_k = 2.0f - 0.19f;
    float svf_a1 = 1.0f / (1.0f + svf_g * (svf_g + svf_k));
    float svf_a2 = svf_g * svf_a1;
    float svf_a3 = svf_g * svf_a2;

    float lfo_inc = pitch_mod_rate / (float)SAMPLE_RATE;

    /* No envelope gating — grains fire continuously at the rate set by
       Smear and Shimmer knobs. They read from the capture buffer which
       contains whatever was last played. When input stops, the grains
       keep reading the last captured audio and the reverb tail carries
       them. The natural decay comes from:
       1. Per-grain Hann envelope (each grain fades itself)
       2. Per-grain darkening filter (each grain gets warmer over its life)
       3. The reverb tail decaying based on Sustain
       4. The capture buffer eventually containing silence
       This creates the wind-chime effect: grains ring out from the last
       sound you played, getting darker, carried by reverb, until the
       capture buffer is all silence and new grains produce nothing. */

    for (int i = 0; i < frames; i++) {
        float dry_l = (float)audio_inout[i * 2] / 32768.0f;
        float dry_r = (float)audio_inout[i * 2 + 1] / 32768.0f;

        /* Capture buffer */
        inst->capture.buffer[inst->capture.write_pos].l = dry_l;
        inst->capture.buffer[inst->capture.write_pos].r = dry_r;
        inst->capture.write_pos = (inst->capture.write_pos + 1) % CAPTURE_SAMPLES;

        /* Pitch mod LFO */
        float lfo_val = sinf(2.0f * (float)M_PI * inst->lfo_phase);
        float pitch_mod = powf(2.0f, pitch_mod_depth * lfo_val * (100.0f / 1200.0f));
        inst->lfo_phase += lfo_inc;
        if (inst->lfo_phase >= 1.0f) inst->lfo_phase -= 1.0f;

        /* === GRAIN TRIGGERS === */

        /* Input level — loud hits spawn extra grains for denser texture */
        float in_level = fabsf(dry_l) + fabsf(dry_r);

        /* Cloud grains (Smear): fire continuously at the set rate */
        inst->cloud_timer++;
        if (inst->cloud_timer >= cloud_rate) {
            inst->cloud_timer = 0;
            float sp = rng_float(&inst->rng_state) < 0.85f ? 1.0f : 2.0f;
            trigger_grain(inst, sp, 0.2f + inst->sustain * 0.3f,
                          cloud_len_ms + rng_float(&inst->rng_state) * 20.0f, pan_width);
            /* Extra grain burst on loud input */
            if (in_level > 0.15f && rng_float(&inst->rng_state) < in_level * 2.0f) {
                float sp2 = rng_float(&inst->rng_state) < 0.5f ? 1.0f : 2.0f;
                trigger_grain(inst, sp2, 0.15f + inst->sustain * 0.2f,
                              cloud_len_ms * 0.7f, pan_width);
            }
        }

        /* Shimmer grains: fire continuously */
        inst->shimmer_grain_timer++;
        if (inst->shimmer_grain_timer >= shim_grain_rate) {
            inst->shimmer_grain_timer = 0;
            trigger_grain(inst, 2.0f, 0.12f + inst->shimmer * 0.25f,
                          sustain_grain_len + rng_float(&inst->rng_state) * 100.0f,
                          pan_width * 0.8f);
            if (rng_float(&inst->rng_state) < 0.15f + inst->shimmer * 0.3f) {
                trigger_grain(inst, 4.0f, 0.2f + inst->shimmer * 0.2f,
                              3.0f + rng_float(&inst->rng_state) * 5.0f, pan_width);
            }
            /* Extra shimmer burst on loud input */
            if (in_level > 0.2f && rng_float(&inst->rng_state) < in_level * 1.5f) {
                trigger_grain(inst, 2.0f, 0.2f + inst->shimmer * 0.15f,
                              sustain_grain_len * 0.5f, pan_width);
            }
        }

        /* === GRAIN PLAYBACK === */
        float wet_l = 0.0f, wet_r = 0.0f;
        for (int g = 0; g < MAX_GRAINS; g++) {
            grain_t *gr = &inst->grains[g];
            if (!gr->active) continue;

            int base_idx = gr->start + (int)gr->position;
            int idx0 = ((base_idx % CAPTURE_SAMPLES) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
            int idx1 = (idx0 + 1) % CAPTURE_SAMPLES;
            float frac = gr->position - floorf(gr->position);

            float samp_l = inst->capture.buffer[idx0].l * (1.0f - frac)
                         + inst->capture.buffer[idx1].l * frac;
            float samp_r = inst->capture.buffer[idx0].r * (1.0f - frac)
                         + inst->capture.buffer[idx1].r * frac;

            /* Per-grain darkening filter: one-pole LP that closes as grain ages.
               Coefficient sweeps from ~1.0 (open) toward lp_target (dark) over lifetime.
               Like a piano string losing brightness as it decays. */
            float lp_coeff = 1.0f - gr->env_phase * (1.0f - gr->lp_target);
            gr->lp_state += lp_coeff * (((samp_l + samp_r) * 0.5f) - gr->lp_state);
            samp_l = samp_l * lp_coeff + gr->lp_state * (1.0f - lp_coeff);
            samp_r = samp_r * lp_coeff + gr->lp_state * (1.0f - lp_coeff);

            int env_idx = (int)(gr->env_phase * (float)(ENV_TABLE_SIZE - 1));
            if (env_idx > ENV_TABLE_SIZE - 1) env_idx = ENV_TABLE_SIZE - 1;
            float env = inst->env_table[env_idx];

            float amp = gr->amplitude * env;
            wet_l += samp_l * amp * gr->pan_l;
            wet_r += samp_r * amp * gr->pan_r;

            gr->position += gr->speed * gr->detune * pitch_mod;
            gr->env_phase += gr->env_inc;
            if (gr->env_phase >= 1.0f) gr->active = 0;
        }

        /* SVF lowpass filter */
        wet_l = svf_tick(&inst->svf_l, wet_l, svf_a1, svf_a2, svf_a3);
        wet_r = svf_tick(&inst->svf_r, wet_r, svf_a1, svf_a2, svf_a3);

        /* FDN reverb */
        float rev_l = 0.0f, rev_r = 0.0f;
        if (rev_send > 0.01f) {
            fdn_process(&inst->reverb, wet_l * rev_send, wet_r * rev_send,
                        &rev_l, &rev_r, fdn_lengths, rev_feedback,
                        rev_damping, shim_feedback, rev_mod_depth);
            wet_l += rev_l;
            wet_r += rev_r;
        }

        /* Mix */
        float out_l = dry_l * (1.0f - inst->mix) + wet_l * inst->mix;
        float out_r = dry_r * (1.0f - inst->mix) + wet_r * inst->mix;

        /* DC blocker */
        float dc_coeff = 0.9975f;
        float dc_l = out_l - inst->dc_prev_in_l + dc_coeff * inst->dc_prev_out_l;
        float dc_r = out_r - inst->dc_prev_in_r + dc_coeff * inst->dc_prev_out_r;
        inst->dc_prev_in_l = out_l; inst->dc_prev_in_r = out_r;
        inst->dc_prev_out_l = dc_l; inst->dc_prev_out_r = dc_r;
        out_l = dc_l; out_r = dc_r;

        /* Soft clip */
        out_l = tanhf(out_l * 1.5f) * 0.667f;
        out_r = tanhf(out_r * 1.5f) * 0.667f;

        audio_inout[i * 2] = (int16_t)(out_l * 32767.0f);
        audio_inout[i * 2 + 1] = (int16_t)(out_r * 32767.0f);
    }
}

static void v2_set_param(void *instance, const char *key, const char *val) {
    if (!instance || !key || !val) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    float v = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));

    if (strcmp(key, "space") == 0) inst->space = v;
    else if (strcmp(key, "shimmer") == 0) inst->shimmer = v;
    else if (strcmp(key, "smear") == 0) inst->smear = v;
    else if (strcmp(key, "warmth") == 0) inst->warmth = v;
    else if (strcmp(key, "drift") == 0) inst->drift = v;
    else if (strcmp(key, "sustain") == 0) inst->sustain = v;
    else if (strcmp(key, "scatter") == 0) inst->scatter = v;
    else if (strcmp(key, "mix") == 0) inst->mix = v;
    else if (strcmp(key, "state") == 0) {
        float f;
        if (json_get_number(val, "space", &f) == 0) inst->space = f;
        if (json_get_number(val, "shimmer", &f) == 0) inst->shimmer = f;
        if (json_get_number(val, "smear", &f) == 0) inst->smear = f;
        if (json_get_number(val, "warmth", &f) == 0) inst->warmth = f;
        if (json_get_number(val, "drift", &f) == 0) inst->drift = f;
        if (json_get_number(val, "sustain", &f) == 0) inst->sustain = f;
        if (json_get_number(val, "scatter", &f) == 0) inst->scatter = f;
        if (json_get_number(val, "mix", &f) == 0) inst->mix = f;
    }
}

static int v2_get_param(void *instance, const char *key, char *buf, int buf_len) {
    if (!instance || !key || !buf || buf_len <= 0) return -1;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    if (strcmp(key, "name") == 0) return snprintf(buf, buf_len, "Dioramatic");
    else if (strcmp(key, "space") == 0) return snprintf(buf, buf_len, "%.3f", inst->space);
    else if (strcmp(key, "shimmer") == 0) return snprintf(buf, buf_len, "%.3f", inst->shimmer);
    else if (strcmp(key, "smear") == 0) return snprintf(buf, buf_len, "%.3f", inst->smear);
    else if (strcmp(key, "warmth") == 0) return snprintf(buf, buf_len, "%.3f", inst->warmth);
    else if (strcmp(key, "drift") == 0) return snprintf(buf, buf_len, "%.3f", inst->drift);
    else if (strcmp(key, "sustain") == 0) return snprintf(buf, buf_len, "%.3f", inst->sustain);
    else if (strcmp(key, "scatter") == 0) return snprintf(buf, buf_len, "%.3f", inst->scatter);
    else if (strcmp(key, "mix") == 0) return snprintf(buf, buf_len, "%.3f", inst->mix);
    else if (strcmp(key, "state") == 0) {
        return snprintf(buf, buf_len,
            "{\"space\":%.3f,\"shimmer\":%.3f,\"smear\":%.3f,\"warmth\":%.3f,"
            "\"drift\":%.3f,\"sustain\":%.3f,\"scatter\":%.3f,\"mix\":%.3f}",
            inst->space, inst->shimmer, inst->smear, inst->warmth,
            inst->drift, inst->sustain, inst->scatter, inst->mix);
    }
    else if (strcmp(key, "chain_params") == 0) {
        return snprintf(buf, buf_len,
            "["
            "{\"key\":\"space\",\"name\":\"Space\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"shimmer\",\"name\":\"Shimmer\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"smear\",\"name\":\"Smear\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"warmth\",\"name\":\"Warmth\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"drift\",\"name\":\"Drift\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"sustain\",\"name\":\"Sustain\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"scatter\",\"name\":\"Scatter\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"mix\",\"name\":\"Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01}"
            "]");
    }
    else if (strcmp(key, "ui_hierarchy") == 0) {
        return snprintf(buf, buf_len,
            "{\"modes\":null,\"levels\":{\"root\":{\"children\":null,"
            "\"knobs\":[\"space\",\"shimmer\",\"smear\",\"warmth\",\"drift\",\"sustain\",\"scatter\",\"mix\"],"
            "\"params\":[\"space\",\"shimmer\",\"smear\",\"warmth\",\"drift\",\"sustain\",\"scatter\",\"mix\"]}}}");
    }
    return -1;
}

static void v2_on_midi(void *instance, const uint8_t *msg, int len, int source) {
    (void)instance; (void)msg; (void)len; (void)source;
}

/* ============================================================================
 * Entry point
 * ============================================================================ */

audio_fx_api_v2_t *move_audio_fx_init_v2(const host_api_v1_t *host) {
    g_host = host;
    g_fx_api_v2.api_version = AUDIO_FX_API_VERSION_2;
    g_fx_api_v2.create_instance = v2_create_instance;
    g_fx_api_v2.destroy_instance = v2_destroy_instance;
    g_fx_api_v2.process_block = v2_process_block;
    g_fx_api_v2.set_param = v2_set_param;
    g_fx_api_v2.get_param = v2_get_param;
    g_fx_api_v2.on_midi = v2_on_midi;
    if (host && host->log) host->log("dioramatic: initialized");
    return &g_fx_api_v2;
}
