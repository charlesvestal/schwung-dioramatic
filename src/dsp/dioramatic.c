/*
 * Dioramatic Audio FX Plugin
 *
 * Granular effects processor with 11 algorithms.
 * Currently a passthrough stub — DSP processing to be added.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_fx_api_v2.h"

/* ============================================================================
 * Audio and grain engine constants
 * ============================================================================ */

#define SAMPLE_RATE 44100
#define CAPTURE_SECONDS 2
#define CAPTURE_SAMPLES (SAMPLE_RATE * CAPTURE_SECONDS)  /* 88200 */
#define MAX_GRAINS 32
#define ENV_TABLE_SIZE 256

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Time division multipliers
 * ============================================================================ */

static const float time_div_multipliers[6] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};

/* ============================================================================
 * RNG (linear congruential)
 * ============================================================================ */

static inline uint32_t rng_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static inline float rng_float(uint32_t *state) {
    return (float)(rng_next(state) >> 8) / 16777216.0f;
}

/* ============================================================================
 * Capture buffer
 * ============================================================================ */

typedef struct {
    float l, r;
} stereo_sample_t;

typedef struct {
    stereo_sample_t buffer[CAPTURE_SAMPLES];
    int write_pos;
} capture_buffer_t;

/* ============================================================================
 * Grain
 * ============================================================================ */

typedef struct {
    int active;
    int start;              /* position in capture buffer */
    int length;             /* grain duration in samples */
    float position;         /* fractional read position */
    float speed;            /* playback rate */
    int direction;          /* +1 forward, -1 reverse */
    float pan;              /* -1.0 to +1.0 */
    float amplitude;        /* base amplitude */
    float env_phase;        /* 0.0-1.0 envelope progress */
    float env_inc;          /* per-sample envelope increment */
    int env_shape;          /* 0=hann, 1=trapezoid, 2=triangle, 3=rectangle */
} grain_t;

/* ============================================================================
 * Algorithm and enum definitions
 * ============================================================================ */

#define NUM_ALGORITHMS 11

static const char *algorithm_names[NUM_ALGORITHMS] = {
    "Mosaic", "Seq", "Glide", "Haze", "Tunnel", "Strum",
    "Blocks", "Interrupt", "Arp", "Pattern", "Warp"
};

#define NUM_VARIATIONS 4
static const char *variation_names[NUM_VARIATIONS] = { "A", "B", "C", "D" };

#define NUM_TIME_DIVS 6
static const char *time_div_names[NUM_TIME_DIVS] = {
    "1/4", "1/2", "1x", "2x", "4x", "8x"
};

#define NUM_REVERB_MODES 4
static const char *reverb_mode_names[NUM_REVERB_MODES] = {
    "Room", "Plate", "Hall", "Ambient"
};

static const char *onoff_names[2] = { "Off", "On" };

/* ============================================================================
 * SVF lowpass filter state
 * ============================================================================ */

typedef struct {
    float ic1eq, ic2eq;
} svf_state_t;

/* ============================================================================
 * FDN stereo reverb
 * ============================================================================ */

#define FDN_LINES 4
#define FDN_MAX_DELAY 8192
#define FDN_NUM_AP 4   /* input allpass diffusers */
#define FDN_AP_MAX 512

typedef struct {
    /* Delay lines */
    float lines[FDN_LINES][FDN_MAX_DELAY];
    int write_pos[FDN_LINES];

    /* Per-line one-pole lowpass for damping */
    float lp_state[FDN_LINES];

    /* Input allpass diffusers */
    float ap_buf[FDN_NUM_AP][FDN_AP_MAX];
    int ap_pos[FDN_NUM_AP];

    /* Pre-delay buffer */
    float predelay_buf_l[4096];
    float predelay_buf_r[4096];
    int predelay_pos;
} fdn_reverb_t;

typedef struct {
    int delay_lengths[FDN_LINES];  /* prime numbers */
    float feedback;
    float damping;    /* one-pole LP coefficient, 0=no damping, 1=max */
    int predelay;     /* samples */
    int ap_lengths[FDN_NUM_AP];
} reverb_preset_t;

static const reverb_preset_t reverb_presets[4] = {
    /* Room */    {{1087, 1283, 1447, 1663}, 0.75f, 0.3f, 0,    {142, 107, 379, 277}},
    /* Plate */   {{2017, 2389, 2777, 3191}, 0.85f, 0.5f, 441,  {142, 107, 379, 277}},
    /* Hall */    {{3547, 4177, 4831, 5557}, 0.92f, 0.6f, 882,  {142, 107, 379, 277}},
    /* Ambient */ {{5501, 6469, 7481, 8179}, 0.97f, 0.7f, 1323, {142, 107, 379, 277}},
};

/* ============================================================================
 * Instance structure
 * ============================================================================ */

typedef struct {
    /* Enum parameters (stored as int indices) */
    int algorithm;       /* 0-10 */
    int variation;       /* 0-3 */
    int time_div;        /* 0-5 */
    int reverb_mode;     /* 0-3 */
    int reverse;         /* 0-1 */
    int hold;            /* 0-1 */

    /* Float parameters (0.0 - 1.0) */
    float activity;
    float repeats;
    float shape;
    float filter;
    float mix;
    float space;
    float pitch_mod_depth;
    float pitch_mod_rate;
    float filter_res;

    /* Grain engine state */
    capture_buffer_t capture;
    grain_t grains[MAX_GRAINS];
    float env_tables[4][ENV_TABLE_SIZE];
    float bpm;
    int subdivision_samples;
    int trigger_counter;
    uint32_t rng_state;

    /* LFO for pitch modulation */
    float lfo_phase;

    /* SVF lowpass filter */
    svf_state_t svf_l, svf_r;
    float svf_g, svf_k, svf_a1, svf_a2, svf_a3;
    float svf_cached_filter;
    float svf_cached_filter_res;

    /* FDN stereo reverb */
    fdn_reverb_t reverb;
} dioramatic_instance_t;

/* ============================================================================
 * Globals
 * ============================================================================ */

static const host_api_v1_t *g_host = NULL;
static audio_fx_api_v2_t g_fx_api_v2;

/* ============================================================================
 * JSON helper — extract a number from a JSON string by key
 * ============================================================================ */

static int json_get_number(const char *json, const char *key, float *out) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;
    /* Handle quoted numeric strings */
    if (*p == '"') {
        p++;
        *out = (float)atof(p);
    } else {
        *out = (float)atof(p);
    }
    return 0;
}

static int json_get_string(const char *json, const char *key, char *out, int out_len) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    const char *end = strchr(p, '"');
    if (!end) return -1;
    int len = (int)(end - p);
    if (len >= out_len) len = out_len - 1;
    memcpy(out, p, len);
    out[len] = '\0';
    return 0;
}

/* ============================================================================
 * Enum lookup helpers
 * ============================================================================ */

static int find_enum_index(const char *val, const char **names, int count) {
    for (int i = 0; i < count; i++) {
        if (strcasecmp(val, names[i]) == 0) return i;
    }
    /* Try numeric */
    char *end;
    long v = strtol(val, &end, 10);
    if (end != val && v >= 0 && v < count) return (int)v;
    return -1;
}

/* ============================================================================
 * Envelope table initialization
 * ============================================================================ */

static void init_envelope_tables(float tables[4][ENV_TABLE_SIZE]) {
    for (int i = 0; i < ENV_TABLE_SIZE; i++) {
        float t = (float)i / (float)(ENV_TABLE_SIZE - 1);

        /* 0: Hann window */
        tables[0][i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * t));

        /* 1: Trapezoid - 10% attack, 80% sustain, 10% release */
        if (t < 0.1f)
            tables[1][i] = t / 0.1f;
        else if (t < 0.9f)
            tables[1][i] = 1.0f;
        else
            tables[1][i] = (1.0f - t) / 0.1f;

        /* 2: Triangle */
        if (t < 0.5f)
            tables[2][i] = t * 2.0f;
        else
            tables[2][i] = (1.0f - t) * 2.0f;

        /* 3: Rectangle with 4-sample fade at edges */
        if (i < 4)
            tables[3][i] = (float)(i + 1) / 4.0f;
        else if (i >= ENV_TABLE_SIZE - 4)
            tables[3][i] = (float)(ENV_TABLE_SIZE - i) / 4.0f;
        else
            tables[3][i] = 1.0f;
    }
}

/* ============================================================================
 * Subdivision calculation
 * ============================================================================ */

static void recalculate_subdivision(dioramatic_instance_t *inst) {
    float mult = time_div_multipliers[inst->time_div];
    int sub = (int)(60.0f / inst->bpm * (float)SAMPLE_RATE * mult);
    /* Clamp: 10ms minimum, 2s maximum */
    if (sub < 441) sub = 441;
    if (sub > CAPTURE_SAMPLES - 1) sub = CAPTURE_SAMPLES - 1;
    inst->subdivision_samples = sub;
}

/* ============================================================================
 * SVF lowpass filter (per-sample, using cached coefficients)
 * ============================================================================ */

static void svf_update_coefficients(dioramatic_instance_t *inst) {
    float cutoff_hz = 80.0f * powf(225.0f, inst->filter);
    if (cutoff_hz > 20000.0f) cutoff_hz = 20000.0f;
    if (cutoff_hz < 20.0f) cutoff_hz = 20.0f;

    float g = tanf((float)M_PI * cutoff_hz / (float)SAMPLE_RATE);
    float k = 2.0f - 2.0f * inst->filter_res * 0.95f;

    inst->svf_g = g;
    inst->svf_k = k;
    inst->svf_a1 = 1.0f / (1.0f + g * (g + k));
    inst->svf_a2 = g * inst->svf_a1;
    inst->svf_a3 = g * inst->svf_a2;
    inst->svf_cached_filter = inst->filter;
    inst->svf_cached_filter_res = inst->filter_res;
}

static inline float svf_lowpass(svf_state_t *s, const dioramatic_instance_t *inst, float input) {
    float v3 = input - s->ic2eq;
    float v1 = inst->svf_a1 * s->ic1eq + inst->svf_a2 * v3;
    float v2 = s->ic2eq + inst->svf_a2 * s->ic1eq + inst->svf_a3 * v3;
    s->ic1eq = 2.0f * v1 - s->ic1eq;
    s->ic2eq = 2.0f * v2 - s->ic2eq;
    return v2;  /* lowpass output */
}

/* ============================================================================
 * FDN stereo reverb (per-sample)
 * ============================================================================ */

static void fdn_process(fdn_reverb_t *rev, int mode, float in_l, float in_r, float *out_l, float *out_r) {
    const reverb_preset_t *p = &reverb_presets[mode];

    /* Pre-delay */
    float pd_l, pd_r;
    if (p->predelay > 0) {
        int pd_read = (rev->predelay_pos - p->predelay + 4096) & 4095;
        pd_l = rev->predelay_buf_l[pd_read];
        pd_r = rev->predelay_buf_r[pd_read];
        rev->predelay_buf_l[rev->predelay_pos] = in_l;
        rev->predelay_buf_r[rev->predelay_pos] = in_r;
        rev->predelay_pos = (rev->predelay_pos + 1) & 4095;
    } else {
        pd_l = in_l;
        pd_r = in_r;
    }

    /* Input diffusion: 4 series allpass filters */
    float diff_in = (pd_l + pd_r) * 0.5f;
    for (int i = 0; i < FDN_NUM_AP; i++) {
        float *buf = rev->ap_buf[i];
        int len = p->ap_lengths[i];
        int pos = rev->ap_pos[i];
        float delayed = buf[pos];
        float ap_out = delayed - 0.6f * diff_in;
        buf[pos] = diff_in + 0.6f * ap_out;
        diff_in = ap_out;
        rev->ap_pos[i] = (pos + 1) % len;
    }

    /* Read from delay lines */
    float taps[FDN_LINES];
    for (int i = 0; i < FDN_LINES; i++) {
        int read_pos = (rev->write_pos[i] - p->delay_lengths[i] + FDN_MAX_DELAY) & (FDN_MAX_DELAY - 1);
        taps[i] = rev->lines[i][read_pos];
    }

    /* Hadamard mixing (4x4 unnormalized, then scale by 0.5) */
    float mixed[FDN_LINES];
    mixed[0] = 0.5f * ( taps[0] + taps[1] + taps[2] + taps[3]);
    mixed[1] = 0.5f * ( taps[0] - taps[1] + taps[2] - taps[3]);
    mixed[2] = 0.5f * ( taps[0] + taps[1] - taps[2] - taps[3]);
    mixed[3] = 0.5f * ( taps[0] - taps[1] - taps[2] + taps[3]);

    /* Apply feedback, damping, and write back with input */
    for (int i = 0; i < FDN_LINES; i++) {
        float fb = mixed[i] * p->feedback;
        /* One-pole lowpass damping */
        rev->lp_state[i] += p->damping * (fb - rev->lp_state[i]);
        fb = rev->lp_state[i];
        /* Write with input */
        rev->lines[i][rev->write_pos[i]] = diff_in + fb;
        rev->write_pos[i] = (rev->write_pos[i] + 1) & (FDN_MAX_DELAY - 1);
    }

    /* Stereo output from alternating lines */
    *out_l = taps[0] + taps[2];
    *out_r = taps[1] + taps[3];
}

/* ============================================================================
 * Mosaic algorithm - grain trigger
 * ============================================================================ */

static void mosaic_trigger_grain(dioramatic_instance_t *inst) {
    /* Find a free grain slot */
    grain_t *g = NULL;
    for (int i = 0; i < MAX_GRAINS; i++) {
        if (!inst->grains[i].active) {
            g = &inst->grains[i];
            break;
        }
    }
    if (!g) return;  /* No free slots */

    int sub = inst->subdivision_samples;
    int wp = inst->capture.write_pos;

    g->active = 1;
    g->start = (wp - (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
    g->length = sub;
    g->env_inc = 1.0f / (float)g->length;
    g->env_phase = 0.0f;
    g->position = 0.0f;
    g->direction = inst->reverse ? -1 : 1;
    g->pan = rng_float(&inst->rng_state) * 1.0f - 0.5f;
    g->amplitude = 0.3f + inst->repeats * 0.7f;

    /* Envelope shape from Shape knob */
    if (inst->shape < 0.25f)
        g->env_shape = 0;
    else if (inst->shape < 0.50f)
        g->env_shape = 1;
    else if (inst->shape < 0.75f)
        g->env_shape = 2;
    else
        g->env_shape = 3;

    /* Speed per variation */
    float r = rng_float(&inst->rng_state);
    switch (inst->variation) {
        case 0:  /* A: normal + octave up */
            g->speed = (r < 0.5f) ? 1.0f : 2.0f;
            break;
        case 1:  /* B: normal + octave down */
            g->speed = (r < 0.5f) ? 0.5f : 1.0f;
            break;
        case 2:  /* C: all octave up */
            g->speed = 2.0f;
            break;
        case 3:  /* D: full range */
        default: {
            int choice = (int)(r * 4.0f);
            if (choice > 3) choice = 3;
            static const float speeds[4] = {0.5f, 1.0f, 2.0f, 4.0f};
            g->speed = speeds[choice];
            break;
        }
    }
}

/* ============================================================================
 * Algorithm tick (called per sample)
 * ============================================================================ */

static void algorithm_tick(dioramatic_instance_t *inst) {
    if (inst->algorithm != 0) return;  /* Only Mosaic for now */

    /* Trigger interval: more frequent with higher activity */
    int trigger_interval = inst->subdivision_samples / (1 + (int)(inst->activity * 3.0f));
    if (trigger_interval < 1) trigger_interval = 1;

    inst->trigger_counter++;
    if (inst->trigger_counter >= trigger_interval) {
        inst->trigger_counter = 0;
        mosaic_trigger_grain(inst);
    }
}

/* ============================================================================
 * v2 API implementation
 * ============================================================================ */

static void *v2_create_instance(const char *module_dir, const char *config_json) {
    dioramatic_instance_t *inst = calloc(1, sizeof(dioramatic_instance_t));
    if (!inst) return NULL;

    /* Set defaults matching module.json */
    inst->algorithm = 0;        /* Mosaic */
    inst->variation = 0;        /* A */
    inst->activity = 0.5f;
    inst->repeats = 0.5f;
    inst->shape = 0.5f;
    inst->filter = 1.0f;
    inst->mix = 0.5f;
    inst->space = 0.3f;
    inst->time_div = 2;         /* 1x */
    inst->pitch_mod_depth = 0.0f;
    inst->pitch_mod_rate = 0.3f;
    inst->filter_res = 0.0f;
    inst->reverb_mode = 0;      /* Room */
    inst->reverse = 0;          /* Off */
    inst->hold = 0;             /* Off */

    /* Grain engine defaults */
    inst->bpm = 120.0f;
    inst->rng_state = 12345;
    inst->lfo_phase = 0.0f;
    init_envelope_tables(inst->env_tables);
    recalculate_subdivision(inst);

    /* Initialize SVF coefficients */
    inst->svf_cached_filter = -1.0f;  /* force first update */
    svf_update_coefficients(inst);

    if (g_host && g_host->log) {
        g_host->log("dioramatic: instance created");
    }
    return inst;
}

static void v2_destroy_instance(void *instance) {
    if (instance) {
        free(instance);
        if (g_host && g_host->log) {
            g_host->log("dioramatic: instance destroyed");
        }
    }
}

static void v2_process_block(void *instance, int16_t *audio_inout, int frames) {
    if (!instance || !audio_inout) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    /* Update tempo from host */
    if (g_host && g_host->get_bpm) {
        float bpm = g_host->get_bpm();
        if (bpm > 0.0f) inst->bpm = bpm;
    }
    recalculate_subdivision(inst);

    /* Recompute SVF coefficients if filter params changed */
    if (inst->filter != inst->svf_cached_filter ||
        inst->filter_res != inst->svf_cached_filter_res) {
        svf_update_coefficients(inst);
    }

    /* Precompute LFO rate for this block */
    float lfo_rate_hz = 0.1f + inst->pitch_mod_rate * 9.9f;
    float lfo_phase_inc = lfo_rate_hz / (float)SAMPLE_RATE;
    int do_pitch_mod = (inst->pitch_mod_depth > 0.001f);

    for (int i = 0; i < frames; i++) {
        /* 1. Read input, convert to float */
        float dry_l = (float)audio_inout[i * 2] / 32768.0f;
        float dry_r = (float)audio_inout[i * 2 + 1] / 32768.0f;

        /* 2. Write to capture buffer */
        int wp = inst->capture.write_pos;
        inst->capture.buffer[wp].l = dry_l;
        inst->capture.buffer[wp].r = dry_r;
        inst->capture.write_pos = (wp + 1) % CAPTURE_SAMPLES;

        /* 3. Advance LFO phase (once per sample, before grain loop) */
        float pitch_mod = 1.0f;
        if (do_pitch_mod) {
            float lfo_val = sinf(2.0f * (float)M_PI * inst->lfo_phase);
            pitch_mod = powf(2.0f, inst->pitch_mod_depth * lfo_val * (100.0f / 1200.0f));
        }
        inst->lfo_phase += lfo_phase_inc;
        if (inst->lfo_phase >= 1.0f) inst->lfo_phase -= 1.0f;

        /* 4. Algorithm tick (may trigger new grains) */
        algorithm_tick(inst);

        /* 5. Accumulate all active grains (with pitch modulation) */
        float wet_l = 0.0f, wet_r = 0.0f;
        for (int g = 0; g < MAX_GRAINS; g++) {
            grain_t *gr = &inst->grains[g];
            if (!gr->active) continue;

            /* Buffer index with linear interpolation */
            int base_idx = gr->start + (int)gr->position;
            /* Wrap for both forward and reverse */
            int idx0 = ((base_idx % CAPTURE_SAMPLES) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
            int idx1 = (idx0 + 1) % CAPTURE_SAMPLES;
            float frac = gr->position - floorf(gr->position);

            float samp_l = inst->capture.buffer[idx0].l * (1.0f - frac)
                         + inst->capture.buffer[idx1].l * frac;
            float samp_r = inst->capture.buffer[idx0].r * (1.0f - frac)
                         + inst->capture.buffer[idx1].r * frac;

            /* Envelope lookup */
            int env_idx = (int)(gr->env_phase * (float)(ENV_TABLE_SIZE - 1));
            if (env_idx > ENV_TABLE_SIZE - 1) env_idx = ENV_TABLE_SIZE - 1;
            float env = inst->env_tables[gr->env_shape][env_idx];

            /* Equal-power pan */
            float pan_angle = (gr->pan + 1.0f) * (float)M_PI * 0.25f;
            float pan_l = cosf(pan_angle);
            float pan_r = sinf(pan_angle);

            float amp = gr->amplitude * env;
            wet_l += samp_l * amp * pan_l;
            wet_r += samp_r * amp * pan_r;

            /* Advance grain with pitch modulation */
            gr->position += gr->speed * pitch_mod * (float)gr->direction;
            gr->env_phase += gr->env_inc;

            if (gr->env_phase >= 1.0f) {
                gr->active = 0;
            }
        }

        /* 6. SVF lowpass filter on wet signal */
        wet_l = svf_lowpass(&inst->svf_l, inst, wet_l);
        wet_r = svf_lowpass(&inst->svf_r, inst, wet_r);

        /* 7. FDN reverb send/return */
        if (inst->space > 0.001f) {
            float rev_out_l = 0.0f, rev_out_r = 0.0f;
            fdn_process(&inst->reverb, inst->reverb_mode, wet_l, wet_r, &rev_out_l, &rev_out_r);
            wet_l += rev_out_l * inst->space;
            wet_r += rev_out_r * inst->space;
        }

        /* 8. Mix dry/wet */
        float out_l = dry_l * (1.0f - inst->mix) + wet_l * inst->mix;
        float out_r = dry_r * (1.0f - inst->mix) + wet_r * inst->mix;

        /* 9. Soft clip and convert back to int16 */
        if (out_l > 1.0f) out_l = 1.0f;
        else if (out_l < -1.0f) out_l = -1.0f;
        if (out_r > 1.0f) out_r = 1.0f;
        else if (out_r < -1.0f) out_r = -1.0f;

        audio_inout[i * 2]     = (int16_t)(out_l * 32767.0f);
        audio_inout[i * 2 + 1] = (int16_t)(out_r * 32767.0f);
    }
}

static void v2_set_param(void *instance, const char *key, const char *val) {
    if (!instance || !key || !val) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    if (strcmp(key, "algorithm") == 0) {
        int idx = find_enum_index(val, algorithm_names, NUM_ALGORITHMS);
        if (idx >= 0) inst->algorithm = idx;
    } else if (strcmp(key, "variation") == 0) {
        int idx = find_enum_index(val, variation_names, NUM_VARIATIONS);
        if (idx >= 0) inst->variation = idx;
    } else if (strcmp(key, "activity") == 0) {
        inst->activity = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "repeats") == 0) {
        inst->repeats = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "shape") == 0) {
        inst->shape = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "filter") == 0) {
        inst->filter = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "mix") == 0) {
        inst->mix = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "space") == 0) {
        inst->space = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "time_div") == 0) {
        int idx = find_enum_index(val, time_div_names, NUM_TIME_DIVS);
        if (idx >= 0) inst->time_div = idx;
    } else if (strcmp(key, "pitch_mod_depth") == 0) {
        inst->pitch_mod_depth = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "pitch_mod_rate") == 0) {
        inst->pitch_mod_rate = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "filter_res") == 0) {
        inst->filter_res = fminf(1.0f, fmaxf(0.0f, (float)atof(val)));
    } else if (strcmp(key, "reverb_mode") == 0) {
        int idx = find_enum_index(val, reverb_mode_names, NUM_REVERB_MODES);
        if (idx >= 0) inst->reverb_mode = idx;
    } else if (strcmp(key, "reverse") == 0) {
        int idx = find_enum_index(val, onoff_names, 2);
        if (idx >= 0) inst->reverse = idx;
    } else if (strcmp(key, "hold") == 0) {
        int idx = find_enum_index(val, onoff_names, 2);
        if (idx >= 0) inst->hold = idx;
    } else if (strcmp(key, "state") == 0) {
        /* Restore full state from JSON */
        float v;
        char sv[64];
        if (json_get_number(val, "activity", &v) == 0) inst->activity = v;
        if (json_get_number(val, "repeats", &v) == 0) inst->repeats = v;
        if (json_get_number(val, "shape", &v) == 0) inst->shape = v;
        if (json_get_number(val, "filter", &v) == 0) inst->filter = v;
        if (json_get_number(val, "mix", &v) == 0) inst->mix = v;
        if (json_get_number(val, "space", &v) == 0) inst->space = v;
        if (json_get_number(val, "pitch_mod_depth", &v) == 0) inst->pitch_mod_depth = v;
        if (json_get_number(val, "pitch_mod_rate", &v) == 0) inst->pitch_mod_rate = v;
        if (json_get_number(val, "filter_res", &v) == 0) inst->filter_res = v;
        if (json_get_number(val, "algorithm", &v) == 0) inst->algorithm = (int)v;
        if (json_get_number(val, "variation", &v) == 0) inst->variation = (int)v;
        if (json_get_number(val, "time_div", &v) == 0) inst->time_div = (int)v;
        if (json_get_number(val, "reverb_mode", &v) == 0) inst->reverb_mode = (int)v;
        if (json_get_number(val, "reverse", &v) == 0) inst->reverse = (int)v;
        if (json_get_number(val, "hold", &v) == 0) inst->hold = (int)v;
        /* Also try string values for enums */
        if (json_get_string(val, "algorithm", sv, sizeof(sv)) == 0) {
            int idx = find_enum_index(sv, algorithm_names, NUM_ALGORITHMS);
            if (idx >= 0) inst->algorithm = idx;
        }
        if (json_get_string(val, "variation", sv, sizeof(sv)) == 0) {
            int idx = find_enum_index(sv, variation_names, NUM_VARIATIONS);
            if (idx >= 0) inst->variation = idx;
        }
        if (json_get_string(val, "time_div", sv, sizeof(sv)) == 0) {
            int idx = find_enum_index(sv, time_div_names, NUM_TIME_DIVS);
            if (idx >= 0) inst->time_div = idx;
        }
        if (json_get_string(val, "reverb_mode", sv, sizeof(sv)) == 0) {
            int idx = find_enum_index(sv, reverb_mode_names, NUM_REVERB_MODES);
            if (idx >= 0) inst->reverb_mode = idx;
        }
        if (json_get_string(val, "reverse", sv, sizeof(sv)) == 0) {
            int idx = find_enum_index(sv, onoff_names, 2);
            if (idx >= 0) inst->reverse = idx;
        }
        if (json_get_string(val, "hold", sv, sizeof(sv)) == 0) {
            int idx = find_enum_index(sv, onoff_names, 2);
            if (idx >= 0) inst->hold = idx;
        }
    }
}

static int v2_get_param(void *instance, const char *key, char *buf, int buf_len) {
    if (!instance || !key || !buf || buf_len <= 0) return -1;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    if (strcmp(key, "name") == 0) {
        return snprintf(buf, buf_len, "Dioramatic");
    } else if (strcmp(key, "algorithm") == 0) {
        return snprintf(buf, buf_len, "%s", algorithm_names[inst->algorithm]);
    } else if (strcmp(key, "variation") == 0) {
        return snprintf(buf, buf_len, "%s", variation_names[inst->variation]);
    } else if (strcmp(key, "activity") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->activity);
    } else if (strcmp(key, "repeats") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->repeats);
    } else if (strcmp(key, "shape") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->shape);
    } else if (strcmp(key, "filter") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->filter);
    } else if (strcmp(key, "mix") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->mix);
    } else if (strcmp(key, "space") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->space);
    } else if (strcmp(key, "time_div") == 0) {
        return snprintf(buf, buf_len, "%s", time_div_names[inst->time_div]);
    } else if (strcmp(key, "pitch_mod_depth") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->pitch_mod_depth);
    } else if (strcmp(key, "pitch_mod_rate") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->pitch_mod_rate);
    } else if (strcmp(key, "filter_res") == 0) {
        return snprintf(buf, buf_len, "%.3f", inst->filter_res);
    } else if (strcmp(key, "reverb_mode") == 0) {
        return snprintf(buf, buf_len, "%s", reverb_mode_names[inst->reverb_mode]);
    } else if (strcmp(key, "reverse") == 0) {
        return snprintf(buf, buf_len, "%s", onoff_names[inst->reverse]);
    } else if (strcmp(key, "hold") == 0) {
        return snprintf(buf, buf_len, "%s", onoff_names[inst->hold]);
    } else if (strcmp(key, "state") == 0) {
        return snprintf(buf, buf_len,
            "{\"algorithm\":%d,\"variation\":%d,\"activity\":%.3f,\"repeats\":%.3f,"
            "\"shape\":%.3f,\"filter\":%.3f,\"mix\":%.3f,\"space\":%.3f,"
            "\"time_div\":%d,\"pitch_mod_depth\":%.3f,\"pitch_mod_rate\":%.3f,"
            "\"filter_res\":%.3f,\"reverb_mode\":%d,\"reverse\":%d,\"hold\":%d}",
            inst->algorithm, inst->variation, inst->activity, inst->repeats,
            inst->shape, inst->filter, inst->mix, inst->space,
            inst->time_div, inst->pitch_mod_depth, inst->pitch_mod_rate,
            inst->filter_res, inst->reverb_mode, inst->reverse, inst->hold);
    } else if (strcmp(key, "chain_params") == 0) {
        return snprintf(buf, buf_len,
            "["
            "{\"key\":\"algorithm\",\"name\":\"Algorithm\",\"type\":\"enum\",\"options\":[\"Mosaic\",\"Seq\",\"Glide\",\"Haze\",\"Tunnel\",\"Strum\",\"Blocks\",\"Interrupt\",\"Arp\",\"Pattern\",\"Warp\"]},"
            "{\"key\":\"variation\",\"name\":\"Variation\",\"type\":\"enum\",\"options\":[\"A\",\"B\",\"C\",\"D\"]},"
            "{\"key\":\"activity\",\"name\":\"Activity\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"repeats\",\"name\":\"Repeats\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"shape\",\"name\":\"Shape\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"filter\",\"name\":\"Filter\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"mix\",\"name\":\"Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"space\",\"name\":\"Space\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"time_div\",\"name\":\"Time\",\"type\":\"enum\",\"options\":[\"1/4\",\"1/2\",\"1x\",\"2x\",\"4x\",\"8x\"]},"
            "{\"key\":\"pitch_mod_depth\",\"name\":\"Pitch Depth\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"pitch_mod_rate\",\"name\":\"Pitch Rate\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"filter_res\",\"name\":\"Resonance\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"reverb_mode\",\"name\":\"Reverb Mode\",\"type\":\"enum\",\"options\":[\"Room\",\"Plate\",\"Hall\",\"Ambient\"]},"
            "{\"key\":\"reverse\",\"name\":\"Reverse\",\"type\":\"enum\",\"options\":[\"Off\",\"On\"]},"
            "{\"key\":\"hold\",\"name\":\"Hold\",\"type\":\"enum\",\"options\":[\"Off\",\"On\"]}"
            "]");
    } else if (strcmp(key, "ui_hierarchy") == 0) {
        return snprintf(buf, buf_len,
            "{\"modes\":null,\"levels\":{\"root\":{\"children\":null,"
            "\"knobs\":[\"activity\",\"repeats\",\"shape\",\"filter\",\"mix\",\"time_div\",\"space\",\"algorithm\"],"
            "\"params\":[\"algorithm\",\"variation\",\"activity\",\"repeats\",\"shape\",\"filter\",\"mix\",\"space\","
            "\"time_div\",\"pitch_mod_depth\",\"pitch_mod_rate\",\"filter_res\",\"reverb_mode\",\"reverse\",\"hold\"]}}}");
    }

    return -1;
}

static void v2_on_midi(void *instance, const uint8_t *msg, int len, int source) {
    /* No MIDI handling yet */
    (void)instance;
    (void)msg;
    (void)len;
    (void)source;
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

    if (host && host->log) {
        host->log("dioramatic: audio FX v2 initialized");
    }

    return &g_fx_api_v2;
}
