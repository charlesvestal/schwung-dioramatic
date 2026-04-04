/*
 * Dioramatic Audio FX Plugin
 *
 * Granular effects processor with 12 algorithms.
 * Algorithms 0-8, 11: grain engine. Algorithms 9-10: multi-tap delay engine.
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
    float speed_target;     /* for glide algorithm */
    float speed_glide_rate; /* per-sample lerp rate */
    float detune;           /* micro-detuning multiplier (0.995-1.005) for chorus shimmer */
    int direction;          /* +1 forward, -1 reverse */
    float pan;              /* -1.0 to +1.0 */
    float amplitude;        /* base amplitude */
    float env_phase;        /* 0.0-1.0 envelope progress */
    float env_inc;          /* per-sample envelope increment */
    int env_shape;          /* 0=hann, 1=trapezoid, 2=triangle, 3=rectangle */
} grain_t;

/* ============================================================================
 * Delay engine (for algorithms 9-10: Pattern, Warp)
 * ============================================================================ */

#define DELAY_BUFFER_SAMPLES 88200  /* 2 seconds at 44100 */
#define MAX_DELAY_TAPS 8

typedef struct {
    int delay_samples;    /* tap delay time */
    float feedback;       /* per-tap feedback */
    float level;          /* per-tap output level */
    float lp_state_l;     /* per-tap lowpass filter state L */
    float lp_state_r;     /* per-tap lowpass filter state R */
    float lp_coeff;       /* lowpass coefficient (0=bypass, higher=more filtering) */
    float speed;          /* pitch shift (1.0 = normal, for Warp) */
} delay_tap_t;

typedef struct {
    stereo_sample_t buffer[DELAY_BUFFER_SAMPLES];
    int write_pos;
    delay_tap_t taps[MAX_DELAY_TAPS];
    int active_taps;
} delay_engine_t;

/* Pattern tap ratios: fractions of subdivision time */
static const float pattern_tap_ratios[4][MAX_DELAY_TAPS] = {
    /* A: Linear evenly spaced */
    {0.125f, 0.250f, 0.375f, 0.500f, 0.625f, 0.750f, 0.875f, 1.000f},
    /* B: Dotted/syncopated */
    {0.167f, 0.333f, 0.500f, 0.667f, 0.833f, 1.000f, 1.167f, 1.333f},
    /* C: Triplet-based */
    {0.333f, 0.667f, 1.000f, 1.333f, 1.667f, 2.000f, 2.333f, 2.667f},
    /* D: Irregular/complex */
    {0.125f, 0.375f, 0.500f, 0.625f, 1.000f, 1.125f, 1.500f, 2.000f},
};

/* ============================================================================
 * Hold/Freeze state
 * ============================================================================ */

typedef struct {
    stereo_sample_t buffer[CAPTURE_SAMPLES];
    int length;          /* captured length */
    int read_pos;        /* current playback position */
    int active;          /* currently holding */
    float fade;          /* crossfade state 0-1 */
    int fade_dir;        /* +1 fading in, -1 fading out */
} hold_state_t;

/* ============================================================================
 * Algorithm and enum definitions
 * ============================================================================ */

#define NUM_ALGORITHMS 12

static const char *algorithm_names[NUM_ALGORITHMS] = {
    "Mosaic", "Seq", "Glide", "Haze", "Tunnel", "Strum",
    "Blocks", "Interrupt", "Arp", "Pattern", "Warp", "Ethereal"
};

#define NUM_VARIATIONS 4
static const char *variation_names[NUM_VARIATIONS] = { "A", "B", "C", "D" };

#define NUM_TIME_DIVS 6
static const char *time_div_names[NUM_TIME_DIVS] = {
    "1/4", "1/2", "1x", "2x", "4x", "8x"
};

#define NUM_REVERB_MODES 4
static const char *reverb_mode_names[NUM_REVERB_MODES] = {
    "Bright", "Dark", "Hall", "Ambient"
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

/* Shimmer pitch shift buffer — simple granular octave-up in the reverb feedback */
#define SHIMMER_BUF_SIZE 4096
#define SHIMMER_GRAIN_SIZE 1024   /* ~23ms grain for smooth pitch shifting */

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

    /* Delay line modulation for chorus-like movement in the tail */
    float mod_phase[FDN_LINES];

    /* Shimmer: octave-up pitch shift in the feedback path.
       Two overlapping grains read at 2x speed, crossfaded with Hann window.
       This is the Eno/Lanois technique — creates cascading harmonics
       that build up in the reverb tail like a crystal cave. */
    float shimmer_buf[SHIMMER_BUF_SIZE];
    int shimmer_write_pos;
    float shimmer_read_phase_a;  /* grain A read position (fractional) */
    float shimmer_read_phase_b;  /* grain B read position (offset by half grain) */

    /* High shelf state for sparkle boost */
    float sparkle_state_l, sparkle_state_r;
} fdn_reverb_t;

typedef struct {
    int delay_lengths[FDN_LINES];  /* prime numbers */
    float feedback;
    float damping;    /* one-pole LP coefficient, 0=no damping, 1=max */
    int predelay;     /* samples */
    int ap_lengths[FDN_NUM_AP];
} reverb_preset_t;

static const reverb_preset_t reverb_presets[4] = {
    /* Bright: tight, clear */
    {{1087, 1283, 1447, 1663}, 0.72f, 0.25f, 0,    {241, 173, 419, 313}},
    /* Dark: warm, intimate */
    {{2017, 2389, 2777, 3191}, 0.80f, 0.65f, 331,  {241, 173, 419, 313}},
    /* Hall: spacious, clear */
    {{3547, 4177, 4831, 5557}, 0.88f, 0.45f, 661,  {241, 173, 419, 313}},
    /* Ambient: massive, wash */
    {{6101, 7103, 7919, 8191}, 0.95f, 0.55f, 1103, {241, 173, 419, 313}},
};

/* ============================================================================
 * Instance structure
 * ============================================================================ */

typedef struct {
    /* Enum parameters (stored as int indices) */
    int algorithm;       /* 0-11 */
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

    /* Shape LFO for wet signal volume modulation */
    float shape_lfo_phase;

    /* SVF lowpass filter */
    svf_state_t svf_l, svf_r;
    float svf_g, svf_k, svf_a1, svf_a2, svf_a3;
    float svf_cached_filter;
    float svf_cached_filter_res;

    /* FDN stereo reverb */
    fdn_reverb_t reverb;

    /* Seq algorithm state */
    int seq_step;
    int seq_positions[8];
    int seq_order[8];

    /* Glide algorithm state */
    int glide_toggle;       /* alternating up/down for variation C */

    /* Haze algorithm state */
    int haze_counter;

    /* Tunnel algorithm state */
    int tunnel_sub_counter;

    /* Strum algorithm state - onset detection */
    int onset_positions[8];
    int onset_count;
    int onset_head;
    float onset_prev_rms;
    int onset_debounce;      /* samples since last onset (prevents re-triggering) */
    int strum_cascade_step;
    int strum_cascade_timer;
    int strum_cascade_total;

    /* Blocks/Interrupt state */
    int interrupt_active;
    int interrupt_remaining;

    /* Arp state */
    int arp_step;
    int arp_step_timer;
    int arp_cycle;

    /* Ethereal algorithm state */
    int ethereal_cloud_counter;     /* micro-grain cloud trigger */
    int ethereal_drone_counter;     /* drone/overtone trigger */
    int ethereal_sparkle_counter;   /* occasional pitch-glitch sparkle */

    /* Delay engine (algorithms 9-10) */
    delay_engine_t delay;
    int delay_taps_dirty;        /* reconfigure taps when params change */
    int delay_prev_algorithm;
    int delay_prev_variation;
    float delay_prev_activity;
    float delay_prev_repeats;
    int delay_prev_subdivision;

    /* Hold/Freeze */
    hold_state_t hold_state;
    int hold_prev;               /* previous hold param value for edge detection */

    /* Algorithm crossfade */
    int crossfade_active;
    int crossfade_counter;
    float crossfade_level;       /* 1.0 -> 0.0 (fade out old) then 0.0 -> 1.0 (fade in new) */
    int pending_algorithm;
    int pending_variation;

    /* MIDI clock */
    uint32_t midi_clock_tick_count;
    uint32_t midi_clock_sample_counter;  /* samples since last clock reset */
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
        float ap_coeff = 0.5f;
        float ap_out = delayed - ap_coeff * diff_in;
        buf[pos] = diff_in + ap_coeff * ap_out;
        diff_in = ap_out;
        rev->ap_pos[i] = (pos + 1) % len;
    }

    /* Read from delay lines with subtle modulation for lush chorus-like tail */
    float taps[FDN_LINES];
    static const float mod_rates[FDN_LINES] = {0.37f, 0.47f, 0.31f, 0.53f};  /* Hz, different per line */
    static const float mod_depth = 12.0f;  /* samples of modulation excursion */
    for (int i = 0; i < FDN_LINES; i++) {
        /* Modulated read position — creates subtle pitch/time variation in the tail */
        float mod_offset = sinf(2.0f * (float)M_PI * rev->mod_phase[i]) * mod_depth;
        int base_delay = p->delay_lengths[i] + (int)mod_offset;
        if (base_delay < 1) base_delay = 1;
        if (base_delay >= FDN_MAX_DELAY) base_delay = FDN_MAX_DELAY - 1;
        int read_pos = (rev->write_pos[i] - base_delay + FDN_MAX_DELAY) & (FDN_MAX_DELAY - 1);
        taps[i] = rev->lines[i][read_pos];
        rev->mod_phase[i] += mod_rates[i] / 44100.0f;
        if (rev->mod_phase[i] >= 1.0f) rev->mod_phase[i] -= 1.0f;
    }

    /* Hadamard mixing (4x4 unnormalized, then scale by 0.5) */
    float mixed[FDN_LINES];
    mixed[0] = 0.5f * ( taps[0] + taps[1] + taps[2] + taps[3]);
    mixed[1] = 0.5f * ( taps[0] - taps[1] + taps[2] - taps[3]);
    mixed[2] = 0.5f * ( taps[0] + taps[1] - taps[2] - taps[3]);
    mixed[3] = 0.5f * ( taps[0] - taps[1] - taps[2] + taps[3]);

    /* Shimmer: pitch-shift the mixed feedback up one octave.
       Two overlapping grains read at 2x speed from a circular buffer,
       crossfaded with Hann windows for smooth, artifact-free shifting.
       This creates the cascading crystalline harmonics. */
    float fb_mono = (mixed[0] + mixed[1] + mixed[2] + mixed[3]) * 0.25f;
    rev->shimmer_buf[rev->shimmer_write_pos] = fb_mono;
    rev->shimmer_write_pos = (rev->shimmer_write_pos + 1) & (SHIMMER_BUF_SIZE - 1);

    /* Grain A: read at 2x speed */
    int rd_a = (int)rev->shimmer_read_phase_a;
    float frac_a = rev->shimmer_read_phase_a - (float)rd_a;
    int idx_a0 = rd_a & (SHIMMER_BUF_SIZE - 1);
    int idx_a1 = (rd_a + 1) & (SHIMMER_BUF_SIZE - 1);
    float samp_a = rev->shimmer_buf[idx_a0] * (1.0f - frac_a) + rev->shimmer_buf[idx_a1] * frac_a;
    /* Hann window for grain A based on position within grain cycle */
    float grain_phase_a = fmodf(rev->shimmer_read_phase_a / (float)SHIMMER_GRAIN_SIZE, 1.0f);
    float env_a = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * grain_phase_a));

    /* Grain B: offset by half a grain length */
    float phase_b_offset = (float)(SHIMMER_GRAIN_SIZE / 2);
    int rd_b = (int)(rev->shimmer_read_phase_a + phase_b_offset);
    float frac_b = (rev->shimmer_read_phase_a + phase_b_offset) - (float)rd_b;
    int idx_b0 = rd_b & (SHIMMER_BUF_SIZE - 1);
    int idx_b1 = (rd_b + 1) & (SHIMMER_BUF_SIZE - 1);
    float samp_b = rev->shimmer_buf[idx_b0] * (1.0f - frac_b) + rev->shimmer_buf[idx_b1] * frac_b;
    float grain_phase_b = fmodf((rev->shimmer_read_phase_a + phase_b_offset) / (float)SHIMMER_GRAIN_SIZE, 1.0f);
    float env_b = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * grain_phase_b));

    float shimmer_out = samp_a * env_a + samp_b * env_b;
    rev->shimmer_read_phase_a += 2.0f;  /* 2x speed = octave up */
    /* Keep read phase from drifting too far from write */
    float dist = (float)rev->shimmer_write_pos - rev->shimmer_read_phase_a;
    if (dist < 0) dist += (float)SHIMMER_BUF_SIZE;
    if (dist > (float)(SHIMMER_BUF_SIZE - SHIMMER_GRAIN_SIZE)) {
        rev->shimmer_read_phase_a = (float)rev->shimmer_write_pos - (float)(SHIMMER_BUF_SIZE / 2);
        if (rev->shimmer_read_phase_a < 0) rev->shimmer_read_phase_a += (float)SHIMMER_BUF_SIZE;
    }

    /* Shimmer amount increases with reverb mode: subtle on Bright, heavy on Ambient */
    static const float shimmer_amounts[4] = {0.08f, 0.12f, 0.18f, 0.30f};
    float shimmer_level = shimmer_amounts[mode];

    /* Apply feedback, damping, shimmer injection, and write back */
    for (int i = 0; i < FDN_LINES; i++) {
        float fb = mixed[i] * p->feedback;
        /* One-pole lowpass damping */
        rev->lp_state[i] += p->damping * (fb - rev->lp_state[i]);
        fb = rev->lp_state[i];
        /* Inject shimmer (octave-up) into the feedback path — this is what
           creates the cascading harmonics building up over the reverb tail */
        fb += shimmer_out * shimmer_level;
        /* Write with input */
        rev->lines[i][rev->write_pos[i]] = diff_in + fb;
        rev->write_pos[i] = (rev->write_pos[i] + 1) & (FDN_MAX_DELAY - 1);
    }

    /* Raw stereo output from alternating lines */
    float raw_l = (taps[0] + taps[2]) * 0.45f;
    float raw_r = (taps[1] + taps[3]) * 0.45f;

    /* High-frequency sparkle shelf: gentle boost to upper harmonics.
       One-pole highpass extracts HF content, blend it back in for brightness.
       This creates the "crystal cave" quality — bright reflections without harshness. */
    float hf_l = raw_l - rev->sparkle_state_l;
    rev->sparkle_state_l += 0.15f * hf_l;  /* ~1kHz crossover */
    float hf_r = raw_r - rev->sparkle_state_r;
    rev->sparkle_state_r += 0.15f * hf_r;

    /* Blend: original + boosted highs (sparkle) */
    static const float sparkle_amounts[4] = {0.15f, 0.05f, 0.20f, 0.25f};
    float sparkle = sparkle_amounts[mode];  /* Bright & Hall/Ambient get more sparkle, Dark gets less */
    *out_l = raw_l + hf_l * sparkle;
    *out_r = raw_r + hf_r * sparkle;
}

/* ============================================================================
 * Shared grain helpers
 * ============================================================================ */

static inline int shape_to_env(float shape) {
    if (shape < 0.25f) return 0;
    if (shape < 0.50f) return 1;
    if (shape < 0.75f) return 2;
    return 3;
}

static grain_t *find_free_grain(dioramatic_instance_t *inst) {
    for (int i = 0; i < MAX_GRAINS; i++) {
        if (!inst->grains[i].active) return &inst->grains[i];
    }
    return NULL;
}

static void init_grain_common(grain_t *g, dioramatic_instance_t *inst, int start, int length, float speed) {
    g->active = 1;
    g->start = start;
    g->length = length;
    g->env_inc = 1.0f / (float)length;
    g->env_phase = 0.0f;
    g->position = 0.0f;
    g->direction = inst->reverse ? -1 : 1;
    g->env_shape = 0;  /* Always Hann — Shape knob controls LFO modulation instead */
    g->pan = (rng_float(&inst->rng_state) - 0.5f) * 0.6f;  /* ±0.3 instead of ±0.5 */
    g->amplitude = 0.5f + inst->repeats * 0.5f;
    g->speed = speed;
    g->speed_target = 0.0f;
    g->speed_glide_rate = 0.0f;
    /* Micro-detuning: ±8 cents random per grain for natural chorus shimmer.
       This is what makes granular effects sound "crystalline" and "glimmering"
       rather than digital and sterile. Each grain is slightly different. */
    g->detune = 1.0f + (rng_float(&inst->rng_state) - 0.5f) * 0.009f;  /* ±~8 cents */

    /* Shorten grains for extreme pitch shifts to sound musical rather than "tape slowing down" */
    if (speed > 1.5f || speed < 0.75f) {
        float speed_factor = (speed > 1.0f) ? speed : (1.0f / speed);
        int adjusted = (int)((float)g->length / speed_factor);
        if (adjusted < 441) adjusted = 441;  /* minimum 10ms */
        g->length = adjusted;
        g->env_inc = 1.0f / (float)g->length;
    }
}

static int count_active_grains(dioramatic_instance_t *inst) {
    int count = 0;
    for (int i = 0; i < MAX_GRAINS; i++) {
        if (inst->grains[i].active) count++;
    }
    return count;
}

/* ============================================================================
 * Algorithm 0: Mosaic - grain trigger
 * ============================================================================ */

static void mosaic_trigger_grain(dioramatic_instance_t *inst) {
    grain_t *g = find_free_grain(inst);
    if (!g) return;

    int sub = inst->subdivision_samples;
    int wp = inst->capture.write_pos;
    int start = (wp - (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
    init_grain_common(g, inst, start, sub, 1.0f);

    /* Speed per variation */
    float r = rng_float(&inst->rng_state);
    switch (inst->variation) {
        case 0:  /* A: normal + octave up shimmer */
            g->speed = (r < 0.6f) ? 1.0f : 2.0f;  /* bias toward normal */
            break;
        case 1:  /* B: normal + octave down */
            g->speed = (r < 0.5f) ? 1.0f : 0.5f;
            break;
        case 2:  /* C: octave up */
            g->speed = 2.0f;
            break;
        case 3:  /* D: octave spread — only octave-aligned ratios stay consonant */
        default: {
            static const float speeds[5] = {0.5f, 1.0f, 1.0f, 2.0f, 2.0f};
            int choice = (int)(r * 5.0f);
            if (choice > 4) choice = 4;
            g->speed = speeds[choice];
            break;
        }
    }
}

static void mosaic_tick(dioramatic_instance_t *inst) {
    /* Dense grain triggering for lush overlap */
    int min_grains = 3 + (int)(inst->activity * 6.0f);  /* 3 to 9 */
    int trigger_interval = inst->subdivision_samples / min_grains;
    if (trigger_interval < 128) trigger_interval = 128;  /* ~3ms minimum */

    inst->trigger_counter++;
    if (inst->trigger_counter >= trigger_interval) {
        inst->trigger_counter = 0;
        mosaic_trigger_grain(inst);

        /* Shimmer layer: every other trigger, add a quiet octave-up grain.
           This creates the "sparkly sprinkles" quality — a delicate halo
           of high harmonics floating above the main effect. */
        if (inst->trigger_counter == 0 && (rng_next(&inst->rng_state) & 1)) {
            grain_t *shimmer = find_free_grain(inst);
            if (shimmer) {
                int sub = inst->subdivision_samples;
                int wp = inst->capture.write_pos;
                int start = (wp - (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                init_grain_common(shimmer, inst, start, sub, 2.0f);
                shimmer->amplitude *= 0.2f;  /* very quiet — felt not heard */
                shimmer->pan = (rng_float(&inst->rng_state) - 0.5f) * 0.8f;  /* wider stereo for shimmer */
            }
        }
    }
}

/* ============================================================================
 * Algorithm 1: Seq - Sequenced Rearrangement
 * ============================================================================ */

static void seq_shuffle(dioramatic_instance_t *inst) {
    /* Fisher-Yates shuffle */
    for (int i = 7; i > 0; i--) {
        int j = (int)(rng_float(&inst->rng_state) * (float)(i + 1));
        if (j > i) j = i;
        int tmp = inst->seq_order[i];
        inst->seq_order[i] = inst->seq_order[j];
        inst->seq_order[j] = tmp;
    }
}

static void seq_capture_slices(dioramatic_instance_t *inst) {
    int wp = inst->capture.write_pos;
    int sub = inst->subdivision_samples;
    for (int i = 0; i < 8; i++) {
        inst->seq_positions[i] = (wp - sub * (8 - i) + CAPTURE_SAMPLES * 8) % CAPTURE_SAMPLES;
    }
    seq_shuffle(inst);
}

static void seq_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;

    inst->trigger_counter++;
    if (inst->trigger_counter >= sub) {
        inst->trigger_counter = 0;

        /* Every 8 steps, re-capture and re-shuffle */
        if (inst->seq_step % 8 == 0) {
            seq_capture_slices(inst);
        }

        int idx = inst->seq_order[inst->seq_step % 8];
        int start = inst->seq_positions[idx];
        int step = inst->seq_step;

        switch (inst->variation) {
            case 0: { /* A: Normal speed, random amplitude mod */
                grain_t *g = find_free_grain(inst);
                if (g) {
                    init_grain_common(g, inst, start, sub, 1.0f);
                    /* Random amplitude modulation scaled by activity */
                    float amp_mod = 1.0f - inst->activity * rng_float(&inst->rng_state) * 0.5f;
                    g->amplitude *= amp_mod;
                }
                break;
            }
            case 1: { /* B: Alternating normal/half-speed, optional pad grain */
                grain_t *g = find_free_grain(inst);
                if (g) {
                    float speed = 1.0f - 0.5f * inst->activity * ((step % 2 == 0) ? 1.0f : 0.0f);
                    init_grain_common(g, inst, start, sub, speed);
                }
                /* At max activity, add a long quiet pad grain */
                if (inst->activity > 0.8f) {
                    grain_t *pad = find_free_grain(inst);
                    if (pad) {
                        int pad_len = sub * 4;
                        if (pad_len > CAPTURE_SAMPLES - 1) pad_len = CAPTURE_SAMPLES - 1;
                        init_grain_common(pad, inst, start, pad_len, 0.5f);
                        pad->amplitude = 0.2f;
                    }
                }
                break;
            }
            case 2: { /* C: Overlapping layers */
                int layers = 1 + (int)(inst->activity * 3.0f);
                for (int l = 0; l < layers; l++) {
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        int offset = (int)(rng_float(&inst->rng_state) * 200.0f) - 100;
                        int s = (start + offset + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                        init_grain_common(g, inst, s, sub, 1.0f);
                    }
                }
                break;
            }
            case 3: { /* D: Bit-crush at high activity */
                grain_t *g = find_free_grain(inst);
                if (g) {
                    float speed = (inst->activity > 0.7f) ? 0.5f : 1.0f;
                    init_grain_common(g, inst, start, sub, speed);
                    if (inst->activity > 0.7f) {
                        g->amplitude *= (0.5f + 0.5f * inst->activity);
                    }
                }
                break;
            }
        }

        inst->seq_step++;
    }
}

/* ============================================================================
 * Algorithm 2: Glide - Pitch Portamento
 * ============================================================================ */

static void glide_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;
    int grain_len = (int)((float)sub * (0.5f + inst->repeats * 1.5f));
    if (grain_len > CAPTURE_SAMPLES - 1) grain_len = CAPTURE_SAMPLES - 1;
    if (grain_len < 441) grain_len = 441;

    /* Trigger overlapping grains at subdivision intervals */
    inst->trigger_counter++;
    if (inst->trigger_counter >= sub) {
        inst->trigger_counter = 0;

        int wp = inst->capture.write_pos;
        int start = (wp - sub + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;

        grain_t *g = find_free_grain(inst);
        if (g) {
            float start_speed = 1.0f;
            float target_speed = 1.0f;

            switch (inst->variation) {
                case 0: /* A: Ascending */
                    target_speed = 1.0f + inst->activity * 1.0f;
                    break;
                case 1: /* B: Descending */
                    target_speed = 1.0f - inst->activity * 0.5f;
                    break;
                case 2: /* C: Alternating up/down */
                    if (inst->glide_toggle)
                        target_speed = 1.0f + inst->activity * 1.0f;
                    else
                        target_speed = 1.0f - inst->activity * 0.5f;
                    inst->glide_toggle = !inst->glide_toggle;
                    break;
                case 3: { /* D: Random octave-safe target */
                    static const float glide_targets[4] = {0.5f, 1.0f, 1.0f, 2.0f};
                    int gi = (int)(rng_float(&inst->rng_state) * 4.0f);
                    if (gi > 3) gi = 3;
                    target_speed = glide_targets[gi];
                    break;
                }
            }

            init_grain_common(g, inst, start, grain_len, start_speed);
            g->speed_target = target_speed;
            g->speed_glide_rate = 0.0001f;
        }
    }
}

/* ============================================================================
 * Algorithm 3: Haze - Granular Cloud
 * ============================================================================ */

static void haze_tick(dioramatic_instance_t *inst) {
    /* Trigger interval: from ~100ms down to ~0.66ms based on activity */
    int trigger_interval = (int)(441.0f / (1.0f + inst->activity * 14.0f));
    if (trigger_interval < 1) trigger_interval = 1;

    inst->haze_counter++;
    if (inst->haze_counter >= trigger_interval) {
        inst->haze_counter = 0;

        grain_t *g = find_free_grain(inst);
        if (!g) return;

        int wp = inst->capture.write_pos;

        /* Grain length: 10-50ms for A, 10-30ms for others adjusted per variation */
        int base_min = 441;
        int base_range = 1764;

        float speed = 1.0f;
        int spread = 22050;  /* 500ms default spread */

        switch (inst->variation) {
            case 0: /* A: Speed 1.0, short grains (10-30ms) */
                base_range = 882;  /* 441 to 1323 */
                break;
            case 1: /* B: Many overlapping, full 2-second spread */
                spread = CAPTURE_SAMPLES;
                break;
            case 2: /* C: Octave shimmer - mix of 1.0 and 2.0 */
                speed = (rng_float(&inst->rng_state) < 0.5f) ? 1.0f : 2.0f;
                break;
            case 3: /* D: Darker - mix of 1.0 and 0.5 */
                speed = (rng_float(&inst->rng_state) < 0.5f) ? 1.0f : 0.5f;
                break;
        }

        int length = base_min + (int)(rng_float(&inst->rng_state) * (float)base_range);
        int start = (wp - (int)(rng_float(&inst->rng_state) * (float)spread) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;

        init_grain_common(g, inst, start, length, speed);
        g->amplitude = 0.25f + inst->repeats * 0.5f;
    }
}

/* ============================================================================
 * Algorithm 4: Tunnel - Drone Generator
 * ============================================================================ */

static void tunnel_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;
    int drone_len = sub * 4;
    if (drone_len > CAPTURE_SAMPLES - 1) drone_len = CAPTURE_SAMPLES - 1;
    int wp = inst->capture.write_pos;

    /* Trigger a new drone when no active grains remain */
    int active = count_active_grains(inst);
    if (active == 0) {
        grain_t *g = find_free_grain(inst);
        if (g) {
            int start = (wp - sub + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
            init_grain_common(g, inst, start, drone_len, 1.0f);
            g->amplitude = 0.3f + inst->repeats * 0.7f;

            switch (inst->variation) {
                case 0: /* A: Smooth sweep */
                    break;
                case 1: /* B: Normal drone (overtones triggered below) */
                    break;
                case 2: /* C: Normal drone (chorus triggered below) */
                    break;
                case 3: /* D: Normal drone (decay layers below) */
                    break;
            }
        }
    }

    /* Overlay grains at longer intervals if activity > 0.3 */
    inst->tunnel_sub_counter++;
    if (inst->tunnel_sub_counter >= sub * 4 && inst->activity > 0.3f) {
        inst->tunnel_sub_counter = 0;

        int start = (wp - (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
        int overlay_count = 1 + (int)(inst->activity * 2.0f);

        switch (inst->variation) {
            case 0: /* A: Extra overlays */
                for (int i = 0; i < overlay_count; i++) {
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        init_grain_common(g, inst, start, drone_len / 2, 1.0f);
                        g->amplitude *= 0.5f;
                    }
                }
                break;
            case 1: { /* B: Overtone grains at 2x and 3x */
                grain_t *g2 = find_free_grain(inst);
                if (g2) {
                    init_grain_common(g2, inst, start, drone_len / 2, 2.0f);
                    g2->amplitude *= 0.3f * inst->activity;
                }
                grain_t *g3 = find_free_grain(inst);
                if (g3) {
                    init_grain_common(g3, inst, start, drone_len / 3, 3.0f);
                    g3->amplitude *= 0.2f * inst->activity;
                }
                break;
            }
            case 2: { /* C: Chorus — same position with slight offset */
                grain_t *g2 = find_free_grain(inst);
                if (g2) {
                    int offset = 20 + (int)(rng_float(&inst->rng_state) * 80.0f);
                    int s = (start + offset) % CAPTURE_SAMPLES;
                    init_grain_common(g2, inst, s, drone_len, 1.0f);
                    g2->amplitude *= 0.4f;
                }
                break;
            }
            case 3: { /* D: Progressively lower amplitude layers */
                for (int i = 0; i < overlay_count; i++) {
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        init_grain_common(g, inst, start, drone_len / 2, 1.0f);
                        g->amplitude *= 0.5f / (float)(i + 1);
                    }
                }
                break;
            }
        }
    }
}

/* ============================================================================
 * Algorithm 5: Strum - Onset Chain
 * ============================================================================ */

static void strum_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;
    int wp = inst->capture.write_pos;

    /* Onset detection — read from the sample just written (one behind write_pos) */
    int prev_wp = (wp - 1 + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
    float rms = fabsf(inst->capture.buffer[prev_wp].l) + fabsf(inst->capture.buffer[prev_wp].r);
    if (inst->onset_debounce > 0) {
        inst->onset_debounce--;
    } else if (rms > 0.1f && inst->onset_prev_rms < 0.05f) {
        inst->onset_positions[inst->onset_head] = prev_wp;
        inst->onset_head = (inst->onset_head + 1) % 8;
        if (inst->onset_count < 8) inst->onset_count++;
        inst->strum_cascade_step = 0;
        inst->strum_cascade_timer = 0;
        inst->strum_cascade_total = 2 + (int)(inst->repeats * 6.0f);
        inst->onset_debounce = 4410;  /* ~100ms debounce */
    }
    inst->onset_prev_rms = rms;

    if (inst->onset_count == 0) return;

    /* Strum cascade logic */
    int cascade_interval = sub / (4 + (int)(inst->activity * 12.0f));
    if (cascade_interval < 1) cascade_interval = 1;

    switch (inst->variation) {
        case 0: { /* A: Most recent onset, repeated at subdivision intervals */
            inst->trigger_counter++;
            if (inst->trigger_counter >= sub) {
                inst->trigger_counter = 0;
                int recent = (inst->onset_head - 1 + 8) % 8;
                grain_t *g = find_free_grain(inst);
                if (g) {
                    init_grain_common(g, inst, inst->onset_positions[recent], sub, 1.0f);
                }
            }
            break;
        }
        case 1: { /* B: Many overlapping copies of most recent onset (phasing) */
            inst->trigger_counter++;
            if (inst->trigger_counter >= sub) {
                inst->trigger_counter = 0;
                int recent = (inst->onset_head - 1 + 8) % 8;
                int copies = 2 + (int)(inst->activity * 4.0f);
                for (int c = 0; c < copies; c++) {
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        int offset = (int)(rng_float(&inst->rng_state) * 100.0f) - 50;
                        int s = (inst->onset_positions[recent] + offset + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                        init_grain_common(g, inst, s, sub, 1.0f);
                    }
                }
            }
            break;
        }
        case 2: /* C: Cascading chain through all stored onsets */
        case 3: { /* D: Like C but with double-speed grains interleaved */
            if (inst->strum_cascade_step < inst->strum_cascade_total) {
                /* Active cascade */
                inst->strum_cascade_timer++;
                if (inst->strum_cascade_timer >= cascade_interval) {
                    inst->strum_cascade_timer = 0;
                    int oi = (inst->onset_head - 1 - (inst->strum_cascade_step % inst->onset_count) + 80) % 8;
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        init_grain_common(g, inst, inst->onset_positions[oi], sub, 1.0f);
                    }
                    if (inst->variation == 3) {
                        grain_t *g2 = find_free_grain(inst);
                        if (g2) {
                            init_grain_common(g2, inst, inst->onset_positions[oi], sub / 2, 2.0f);
                            g2->amplitude *= 0.6f;
                        }
                    }
                    inst->strum_cascade_step++;
                }
            } else {
                /* Cascade finished — re-trigger periodically from stored onsets */
                inst->trigger_counter++;
                if (inst->trigger_counter >= sub * 2) {
                    inst->trigger_counter = 0;
                    inst->strum_cascade_step = 0;
                    inst->strum_cascade_timer = 0;
                }
            }
            break;
        }
    }
}

/* ============================================================================
 * Algorithm 6: Blocks - Glitch Stutters
 * ============================================================================ */

static void blocks_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;
    int wp = inst->capture.write_pos;

    inst->trigger_counter++;
    if (inst->trigger_counter >= sub) {
        inst->trigger_counter = 0;

        float prob;
        int grain_len;
        int start = (wp - sub / 2 + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;

        switch (inst->variation) {
            case 0: { /* A: Regular, predictable stutters */
                prob = 1.0f;
                grain_len = sub / (2 + (int)(inst->activity * 6.0f));
                if (grain_len < 64) grain_len = 64;
                if (rng_float(&inst->rng_state) < prob) {
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        init_grain_common(g, inst, start, grain_len, 1.0f);
                        g->amplitude = 0.8f + inst->repeats * 0.2f;
                    }
                }
                break;
            }
            case 1: { /* B: Random bursts — probabilistic */
                prob = 0.2f + inst->activity * 0.8f;
                grain_len = sub / 4 + (int)(rng_float(&inst->rng_state) * ((float)sub / 4.0f));
                if (grain_len < 64) grain_len = 64;
                if (rng_float(&inst->rng_state) < prob) {
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        init_grain_common(g, inst, start, grain_len, 1.0f);
                        g->amplitude = 0.8f + inst->repeats * 0.2f;
                    }
                }
                break;
            }
            case 2: { /* C: Pitch-shifted bursts */
                prob = 0.2f + inst->activity * 0.8f;
                grain_len = sub / 4 + (int)(rng_float(&inst->rng_state) * ((float)sub / 4.0f));
                if (grain_len < 64) grain_len = 64;
                if (rng_float(&inst->rng_state) < prob) {
                    static const float block_speeds[4] = {0.5f, 1.0f, 1.5f, 2.0f};
                    int si = (int)(rng_float(&inst->rng_state) * 4.0f);
                    if (si > 3) si = 3;
                    grain_t *g = find_free_grain(inst);
                    if (g) {
                        init_grain_common(g, inst, start, grain_len, block_speeds[si]);
                        g->amplitude = 0.8f + inst->repeats * 0.2f;
                    }
                }
                break;
            }
            case 3: { /* D: Multiple rapid-fire short grains */
                prob = 0.2f + inst->activity * 0.8f;
                if (rng_float(&inst->rng_state) < prob) {
                    int count = 2 + (int)(inst->activity * 4.0f);
                    for (int c = 0; c < count; c++) {
                        grain_t *g = find_free_grain(inst);
                        if (g) {
                            grain_len = sub / 8 + (int)(rng_float(&inst->rng_state) * ((float)sub / 8.0f));
                            if (grain_len < 64) grain_len = 64;
                            int s = (start + (int)(rng_float(&inst->rng_state) * ((float)sub / 2.0f)) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                            init_grain_common(g, inst, s, grain_len, 1.0f);
                            g->amplitude = (0.8f + inst->repeats * 0.2f) * (0.3f + rng_float(&inst->rng_state) * 0.7f);
                        }
                    }
                }
                break;
            }
        }
    }
}

/* ============================================================================
 * Algorithm 7: Interrupt - Dry + Glitch Bursts
 * ============================================================================ */

static void interrupt_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;
    int wp = inst->capture.write_pos;

    /* Count down active interrupt */
    if (inst->interrupt_active) {
        inst->interrupt_remaining--;
        if (inst->interrupt_remaining <= 0) {
            inst->interrupt_active = 0;
        }
    }

    inst->trigger_counter++;
    if (inst->trigger_counter >= sub) {
        inst->trigger_counter = 0;

        float prob = inst->repeats * 0.5f;
        if (rng_float(&inst->rng_state) < prob) {
            inst->interrupt_active = 1;
            inst->interrupt_remaining = sub / 2;

            int start = (wp - sub + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
            int count = 1 + (int)(inst->activity * 2.0f);

            for (int c = 0; c < count; c++) {
                grain_t *g = find_free_grain(inst);
                if (!g) break;

                switch (inst->variation) {
                    case 0: { /* A: Rearranged at speed 1.0 */
                        int s = (start + (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                        init_grain_common(g, inst, s, sub / 2, 1.0f);
                        break;
                    }
                    case 1: { /* B: Pitch-shifted 0.5 or 2.0 */
                        float speed = (rng_float(&inst->rng_state) < 0.5f) ? 0.5f : 2.0f;
                        int s = (start + (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                        init_grain_common(g, inst, s, sub / 2, speed);
                        break;
                    }
                    case 2: { /* C: Smooth sweep feel */
                        int s = (start + (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                        init_grain_common(g, inst, s, sub / 2, 1.0f);
                        break;
                    }
                    case 3: { /* D: Very short, loud */
                        int grain_len = sub / 8;
                        if (grain_len < 64) grain_len = 64;
                        int s = (start + (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                        init_grain_common(g, inst, s, grain_len, 1.0f);
                        g->amplitude = 1.0f;
                        break;
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * Algorithm 8: Arp - Granular Arpeggio
 * ============================================================================ */

static const float arp_intervals[] = {
    1.0f,       /* unison */
    1.12246f,   /* major 2nd */
    1.25992f,   /* major 3rd */
    1.33484f,   /* perfect 4th */
    1.49831f,   /* perfect 5th */
    2.0f,       /* octave */
};

static void arp_tick(dioramatic_instance_t *inst) {
    int sub = inst->subdivision_samples;
    int step_interval = sub / (2 + (int)(inst->activity * 6.0f));
    if (step_interval < 1) step_interval = 1;

    inst->arp_step_timer++;
    if (inst->arp_step_timer >= step_interval) {
        inst->arp_step_timer = 0;

        int wp = inst->capture.write_pos;
        int start = (wp - sub + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;

        /* Determine interval index based on variation and step */
        int idx = 0;
        int pattern_len = 4;

        switch (inst->variation) {
            case 0: /* A: Ascending I, III, V, VIII */
                {
                    static const int asc_pat[4] = {0, 2, 4, 5};
                    idx = asc_pat[inst->arp_step % 4];
                }
                break;
            case 1: /* B: Descending VIII, V, III, I */
                {
                    static const int desc_pat[4] = {5, 4, 2, 0};
                    idx = desc_pat[inst->arp_step % 4];
                }
                break;
            case 2: /* C: Up-down 0,2,4,5,4,2 */
                pattern_len = 6;
                {
                    static const int updn_pat[6] = {0, 2, 4, 5, 4, 2};
                    idx = updn_pat[inst->arp_step % 6];
                }
                break;
            case 3: /* D: Random from all 6 */
                idx = (int)(rng_float(&inst->rng_state) * 6.0f);
                if (idx > 5) idx = 5;
                break;
        }

        float speed = arp_intervals[idx];

        grain_t *g = find_free_grain(inst);
        if (g) {
            int grain_len = sub;
            if (grain_len > CAPTURE_SAMPLES - 1) grain_len = CAPTURE_SAMPLES - 1;
            init_grain_common(g, inst, start, grain_len, speed);

            /* Fade amplitude over cycles */
            int max_cycles = 1 + (int)(inst->repeats * 4.0f);
            int current_cycle = inst->arp_step / pattern_len;
            if (max_cycles > 0 && current_cycle < max_cycles) {
                g->amplitude *= 1.0f - (float)current_cycle / (float)(max_cycles + 1);
            } else if (current_cycle >= max_cycles) {
                g->amplitude *= 0.1f;
            }
        }

        inst->arp_step++;
        /* Reset cycle tracking */
        if (inst->variation == 2) {
            if (inst->arp_step % 6 == 0) inst->arp_cycle++;
        } else {
            if (inst->arp_step % 4 == 0) inst->arp_cycle++;
        }
    }
}

/* ============================================================================
 * Delay engine tap configuration (algorithms 9-10)
 * ============================================================================ */

static void pattern_configure_taps(dioramatic_instance_t *inst) {
    delay_engine_t *de = &inst->delay;
    int sub = inst->subdivision_samples;
    int var = inst->variation;

    de->active_taps = 1 + (int)(inst->activity * 7.0f);
    if (de->active_taps > MAX_DELAY_TAPS) de->active_taps = MAX_DELAY_TAPS;

    float feedback = inst->repeats * 0.85f;

    for (int t = 0; t < de->active_taps; t++) {
        delay_tap_t *tap = &de->taps[t];
        int delay_samp = (int)(pattern_tap_ratios[var][t] * (float)sub);
        if (delay_samp < 1) delay_samp = 1;
        if (delay_samp > DELAY_BUFFER_SAMPLES - 1) delay_samp = DELAY_BUFFER_SAMPLES - 1;
        tap->delay_samples = delay_samp;
        tap->feedback = feedback;
        tap->level = 1.0f / (1.0f + (float)t * 0.3f);
        tap->lp_coeff = 0.0f;
        tap->speed = 1.0f;
    }
}

static void warp_configure_taps(dioramatic_instance_t *inst) {
    delay_engine_t *de = &inst->delay;
    int sub = inst->subdivision_samples;
    int var = inst->variation;

    de->active_taps = 1 + (int)(inst->activity * 7.0f);
    if (de->active_taps > MAX_DELAY_TAPS) de->active_taps = MAX_DELAY_TAPS;

    float feedback = inst->repeats * 0.85f;

    for (int t = 0; t < de->active_taps; t++) {
        delay_tap_t *tap = &de->taps[t];
        /* Use Pattern A spacing as base */
        int delay_samp = (int)(pattern_tap_ratios[0][t] * (float)sub);
        if (delay_samp < 1) delay_samp = 1;
        if (delay_samp > DELAY_BUFFER_SAMPLES - 1) delay_samp = DELAY_BUFFER_SAMPLES - 1;
        tap->delay_samples = delay_samp;
        tap->feedback = feedback;
        tap->level = 1.0f / (1.0f + (float)t * 0.3f);
        tap->lp_coeff = 0.0f;
        tap->speed = 1.0f;

        switch (var) {
            case 0: /* A: Ascending pitch per tap */
                tap->speed = powf(2.0f, inst->activity * 2.0f * (float)t / 12.0f);
                break;
            case 1: /* B: Progressive LP filtering */
                tap->lp_coeff = 0.1f + (float)t * inst->activity * 0.1f;
                if (tap->lp_coeff > 0.99f) tap->lp_coeff = 0.99f;
                break;
            case 2: /* C: Combined pitch + filter */
                tap->speed = powf(2.0f, inst->activity * 2.0f * (float)t / 12.0f);
                tap->lp_coeff = 0.1f + (float)t * inst->activity * 0.1f;
                if (tap->lp_coeff > 0.99f) tap->lp_coeff = 0.99f;
                break;
            case 3: /* D: Reverse read (negative speed marker, handled in processing) */
                tap->speed = -1.0f;
                break;
        }
    }
}

static void delay_configure_taps_if_dirty(dioramatic_instance_t *inst) {
    if (inst->algorithm != 9 && inst->algorithm != 10) return;

    /* Check if any relevant parameters changed */
    if (inst->algorithm != inst->delay_prev_algorithm ||
        inst->variation != inst->delay_prev_variation ||
        inst->activity != inst->delay_prev_activity ||
        inst->repeats != inst->delay_prev_repeats ||
        inst->subdivision_samples != inst->delay_prev_subdivision ||
        inst->delay_taps_dirty) {

        if (inst->algorithm == 9) {
            pattern_configure_taps(inst);
        } else {
            warp_configure_taps(inst);
        }

        inst->delay_prev_algorithm = inst->algorithm;
        inst->delay_prev_variation = inst->variation;
        inst->delay_prev_activity = inst->activity;
        inst->delay_prev_repeats = inst->repeats;
        inst->delay_prev_subdivision = inst->subdivision_samples;
        inst->delay_taps_dirty = 0;
    }
}

/* ============================================================================
 * Algorithm 11: Ethereal - unified crystal/angel/smear effect
 * Layers micro-grain clouds, overtone drones, and crystal sparkles.
 * ============================================================================ */

static void ethereal_tick(dioramatic_instance_t *inst) {
    float activity = inst->activity;
    float repeats = inst->repeats;
    int sub = inst->subdivision_samples;
    int wp = inst->capture.write_pos;

    /* Variation balance multipliers */
    float cloud_density = 1.0f, drone_amp = 1.0f, sparkle_prob = 1.0f;
    switch (inst->variation) {
        case 0: break; /* A: balanced */
        case 1: cloud_density = 1.5f; sparkle_prob = 0.3f; break; /* B: cloud-heavy */
        case 2: drone_amp = 1.5f; cloud_density = 0.6f; break; /* C: drone-heavy */
        case 3: sparkle_prob = 2.5f; cloud_density = 0.7f; break; /* D: sparkle-heavy */
    }

    /* --- Layer 1: Micro-grain cloud (the "wash") --- */
    int cloud_interval = (int)(882.0f / (1.0f + activity * 8.0f));
    cloud_interval = (int)((float)cloud_interval / cloud_density);
    if (cloud_interval < 64) cloud_interval = 64;

    inst->ethereal_cloud_counter++;
    if (inst->ethereal_cloud_counter >= cloud_interval) {
        inst->ethereal_cloud_counter = 0;
        grain_t *g = find_free_grain(inst);
        if (g) {
            /* Grain length: 441 to 2205 samples (10-50ms) */
            int grain_len = 441 + (int)(rng_float(&inst->rng_state) * 1764.0f);
            /* Start position: random within last 1 second */
            int start = (wp - (int)(rng_float(&inst->rng_state) * (float)SAMPLE_RATE) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
            /* Speed: mostly 1.0, occasional 2.0 (octave up shimmer) */
            float speed = (rng_float(&inst->rng_state) < 0.85f) ? 1.0f : 2.0f;
            init_grain_common(g, inst, start, grain_len, speed);
            g->direction = -1;  /* reverse for subtle time-smearing */
            g->amplitude = 0.2f + repeats * 0.3f;
        }
    }

    /* --- Layer 2: Overtone drone (the "angels") --- */
    int drone_interval = sub * 2;
    if (drone_interval < 1) drone_interval = 1;

    inst->ethereal_drone_counter++;
    if (inst->ethereal_drone_counter >= drone_interval) {
        inst->ethereal_drone_counter = 0;
        int start = (wp - sub + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;

        /* Fundamental grain: speed 1.0 */
        grain_t *g1 = find_free_grain(inst);
        if (g1) {
            int len1 = sub * 3;
            if (len1 > CAPTURE_SAMPLES - 1) len1 = CAPTURE_SAMPLES - 1;
            init_grain_common(g1, inst, start, len1, 1.0f);
            g1->direction = 1;  /* forward — drones should be stable */
            g1->amplitude = (0.3f + repeats * 0.4f) * drone_amp;
        }

        /* Octave up grain: speed 2.0 */
        grain_t *g2 = find_free_grain(inst);
        if (g2) {
            int len2 = sub * 2;
            if (len2 > CAPTURE_SAMPLES - 1) len2 = CAPTURE_SAMPLES - 1;
            init_grain_common(g2, inst, start, len2, 2.0f);
            g2->direction = 1;
            g2->amplitude = (0.15f + repeats * 0.2f) * drone_amp;
        }

        /* Octave down grain: speed 0.5, only at higher activity */
        if (activity > 0.5f) {
            grain_t *g3 = find_free_grain(inst);
            if (g3) {
                int len3 = sub * 4;
                if (len3 > CAPTURE_SAMPLES - 1) len3 = CAPTURE_SAMPLES - 1;
                init_grain_common(g3, inst, start, len3, 0.5f);
                g3->direction = 1;
                g3->amplitude = (0.1f + repeats * 0.15f) * drone_amp;
            }
        }
    }

    /* --- Layer 3: Crystal sparkles (the "dynamism") --- */
    inst->ethereal_sparkle_counter++;
    if (inst->ethereal_sparkle_counter >= sub) {
        inst->ethereal_sparkle_counter = 0;
        float prob = (0.02f + activity * 0.08f) * sparkle_prob;
        if (rng_float(&inst->rng_state) < prob) {
            grain_t *g = find_free_grain(inst);
            if (g) {
                /* Very short grain: 220-660 samples (5-15ms) */
                int grain_len = 220 + (int)(rng_float(&inst->rng_state) * 440.0f);
                int start = (wp - (int)(rng_float(&inst->rng_state) * (float)sub) + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                init_grain_common(g, inst, start, grain_len, 2.0f);
                g->amplitude = 0.6f;
                g->pan = (rng_float(&inst->rng_state) - 0.5f) * 0.8f;  /* wide: +/-0.4 */
            }
        }
    }
}

/* ============================================================================
 * Algorithm tick dispatcher (called per sample)
 * ============================================================================ */

static void algorithm_tick(dioramatic_instance_t *inst) {
    switch (inst->algorithm) {
        case 0: mosaic_tick(inst); break;
        case 1: seq_tick(inst); break;
        case 2: glide_tick(inst); break;
        case 3: haze_tick(inst); break;
        case 4: tunnel_tick(inst); break;
        case 5: strum_tick(inst); break;
        case 6: blocks_tick(inst); break;
        case 7: interrupt_tick(inst); break;
        case 8: arp_tick(inst); break;
        case 11: ethereal_tick(inst); break;
        default: break;  /* algorithms 9-10 use delay engine, handled separately */
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
    inst->reverb_mode = 0;      /* Bright */
    inst->reverse = 0;          /* Off */
    inst->hold = 0;             /* Off */

    /* Grain engine defaults */
    inst->bpm = 120.0f;
    inst->rng_state = 12345;
    inst->lfo_phase = 0.0f;
    init_envelope_tables(inst->env_tables);
    recalculate_subdivision(inst);

    /* Initialize algorithm state */
    for (int i = 0; i < 8; i++) inst->seq_order[i] = i;

    /* Initialize delay engine */
    inst->delay_taps_dirty = 1;
    inst->delay_prev_algorithm = -1;

    /* Initialize hold state */
    inst->hold_prev = 0;

    /* Initialize crossfade */
    inst->crossfade_active = 0;
    inst->crossfade_level = 1.0f;
    inst->pending_algorithm = 0;
    inst->pending_variation = 0;

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

    /* Configure delay taps if needed (algorithms 9-10) */
    delay_configure_taps_if_dirty(inst);

    /* Handle crossfade midpoint switch */
    /* (crossfade_level goes from 1.0 down to 0.0, then the pending params
       are applied and it goes back up to 1.0) */

    /* Handle hold edge detection */
    if (inst->hold != inst->hold_prev) {
        if (inst->hold == 1 && inst->hold_prev == 0) {
            /* Hold turned ON: capture recent audio into hold buffer */
            int len = inst->subdivision_samples;
            if (len > CAPTURE_SAMPLES) len = CAPTURE_SAMPLES;
            if (len < 64) len = 64;
            inst->hold_state.length = len;
            int cap_wp = inst->capture.write_pos;
            for (int h = 0; h < len; h++) {
                int src = (cap_wp - len + h + CAPTURE_SAMPLES) % CAPTURE_SAMPLES;
                inst->hold_state.buffer[h] = inst->capture.buffer[src];
            }
            inst->hold_state.read_pos = 0;
            inst->hold_state.active = 1;
            inst->hold_state.fade = 0.0f;
            inst->hold_state.fade_dir = 1;
        } else if (inst->hold == 0 && inst->hold_prev == 1) {
            /* Hold turned OFF: start fade-out */
            inst->hold_state.fade_dir = -1;
        }
        inst->hold_prev = inst->hold;
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

        /* 2b. Hold/Freeze: feed hold buffer into capture buffer so grains re-process it */
        if (inst->hold_state.active) {
            float hold_l = inst->hold_state.buffer[inst->hold_state.read_pos].l;
            float hold_r = inst->hold_state.buffer[inst->hold_state.read_pos].r;
            inst->hold_state.read_pos = (inst->hold_state.read_pos + 1) % inst->hold_state.length;

            float fade = inst->hold_state.fade;
            /* Blend hold content into capture buffer (replacing dry input for grains) */
            inst->capture.buffer[wp].l = hold_l * fade + dry_l * (1.0f - fade);
            inst->capture.buffer[wp].r = hold_r * fade + dry_r * (1.0f - fade);

            /* Advance fade (~50ms = 2205 samples) */
            inst->hold_state.fade += (float)inst->hold_state.fade_dir / 2205.0f;
            if (inst->hold_state.fade > 1.0f) inst->hold_state.fade = 1.0f;
            if (inst->hold_state.fade <= 0.0f) {
                inst->hold_state.fade = 0.0f;
                if (inst->hold_state.fade_dir == -1) inst->hold_state.active = 0;
            }
        }

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

        /* 5. Accumulate wet signal — delay engine or grain engine */
        float wet_l = 0.0f, wet_r = 0.0f;

        if (inst->algorithm == 9 || inst->algorithm == 10) {
            /* ---- Delay engine path (Pattern / Warp) ---- */
            delay_engine_t *de = &inst->delay;

            /* Write input + feedback to delay buffer */
            float fb_l = 0.0f, fb_r = 0.0f;
            for (int t = 0; t < de->active_taps; t++) {
                delay_tap_t *tap = &de->taps[t];
                int rp = (de->write_pos - tap->delay_samples + DELAY_BUFFER_SAMPLES) % DELAY_BUFFER_SAMPLES;
                fb_l += de->buffer[rp].l * tap->feedback * tap->level;
                fb_r += de->buffer[rp].r * tap->feedback * tap->level;
            }
            de->buffer[de->write_pos].l = dry_l + fb_l * 0.5f;
            de->buffer[de->write_pos].r = dry_r + fb_r * 0.5f;

            /* Read from taps */
            for (int t = 0; t < de->active_taps; t++) {
                delay_tap_t *tap = &de->taps[t];
                int rp;

                if (tap->speed < 0.0f) {
                    /* Warp variation D: reverse read — mirror the tap position.
                       Normal read: write_pos - delay. Reverse: read from the
                       "other side" — same distance but we want recently written audio
                       played in reverse order. Use half the delay as read point. */
                    rp = (de->write_pos - tap->delay_samples / 2 + DELAY_BUFFER_SAMPLES) % DELAY_BUFFER_SAMPLES;
                } else {
                    rp = (de->write_pos - tap->delay_samples + DELAY_BUFFER_SAMPLES) % DELAY_BUFFER_SAMPLES;
                }

                float tap_l = de->buffer[rp].l * tap->level;
                float tap_r = de->buffer[rp].r * tap->level;

                /* Per-tap lowpass (one-pole) */
                if (tap->lp_coeff > 0.001f) {
                    tap->lp_state_l += tap->lp_coeff * (tap_l - tap->lp_state_l);
                    tap_l = tap->lp_state_l;
                    tap->lp_state_r += tap->lp_coeff * (tap_r - tap->lp_state_r);
                    tap_r = tap->lp_state_r;
                }

                wet_l += tap_l;
                wet_r += tap_r;
            }

            /* Scale delay output to prevent buildup with many taps */
            float tap_scale = 1.0f / (1.0f + (float)de->active_taps * 0.15f);
            wet_l *= tap_scale;
            wet_r *= tap_scale;

            de->write_pos = (de->write_pos + 1) % DELAY_BUFFER_SAMPLES;
        } else {
            /* ---- Grain engine path (algorithms 0-8) ---- */
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

                /* Glide: lerp speed toward target */
                if (gr->speed_target != 0.0f) {
                    gr->speed += (gr->speed_target - gr->speed) * gr->speed_glide_rate;
                }

                /* Advance grain with pitch modulation + per-grain detune */
                gr->position += gr->speed * gr->detune * pitch_mod * (float)gr->direction;
                gr->env_phase += gr->env_inc;

                if (gr->env_phase >= 1.0f) {
                    gr->active = 0;
                }
            }
        }

        /* 5b. Algorithm crossfade envelope */
        if (inst->crossfade_active) {
            #define CROSSFADE_HALF_SAMPLES 1102  /* ~25ms at 44100 */
            inst->crossfade_counter++;
            if (inst->crossfade_counter <= CROSSFADE_HALF_SAMPLES) {
                /* Fading out old algorithm */
                inst->crossfade_level = 1.0f - (float)inst->crossfade_counter / (float)CROSSFADE_HALF_SAMPLES;
            } else if (inst->crossfade_counter == CROSSFADE_HALF_SAMPLES + 1) {
                /* Midpoint: switch to new algorithm/variation */
                inst->algorithm = inst->pending_algorithm;
                inst->variation = inst->pending_variation;
                /* Clear all grains for clean start */
                for (int g = 0; g < MAX_GRAINS; g++) inst->grains[g].active = 0;
                /* Mark delay taps dirty */
                inst->delay_taps_dirty = 1;
                delay_configure_taps_if_dirty(inst);
                inst->crossfade_level = 0.0f;
            } else {
                /* Fading in new algorithm */
                int fade_in_pos = inst->crossfade_counter - CROSSFADE_HALF_SAMPLES - 1;
                inst->crossfade_level = (float)fade_in_pos / (float)CROSSFADE_HALF_SAMPLES;
                if (inst->crossfade_level >= 1.0f) {
                    inst->crossfade_level = 1.0f;
                    inst->crossfade_active = 0;
                }
            }
            wet_l *= inst->crossfade_level;
            wet_r *= inst->crossfade_level;
        }

        /* 5c. Shape LFO modulation — modulates wet signal volume rhythmically */
        if (inst->shape < 0.95f) {
            float shape_lfo_val;
            float phase = inst->shape_lfo_phase;

            if (inst->shape < 0.25f) {
                /* Zone 1: Square wave — on/off gating */
                shape_lfo_val = (phase < 0.5f) ? 1.0f : 0.0f;
            } else if (inst->shape < 0.50f) {
                /* Zone 2: Ramp — gradual rise, abrupt cut */
                shape_lfo_val = phase;
            } else if (inst->shape < 0.75f) {
                /* Zone 3: Triangle — smooth fade in and out */
                shape_lfo_val = (phase < 0.5f) ? (phase * 2.0f) : (2.0f - phase * 2.0f);
            } else {
                /* Zone 4: Saw — abrupt rise, gradual fall */
                shape_lfo_val = 1.0f - phase;
            }

            /* Full-depth modulation — shape knob IS the modulator.
               At zone center: 100% depth (wet goes to silence at LFO trough).
               Nearer to 0.95 (off): depth fades out smoothly. */
            float depth_fade = 1.0f - (inst->shape / 0.95f);  /* 1.0 at shape=0, 0.0 at shape=0.95 */
            if (depth_fade > 1.0f) depth_fade = 1.0f;
            float mod_depth = 0.7f + 0.3f * depth_fade;  /* 70-100% depth */

            float shape_mod = 1.0f - mod_depth * (1.0f - shape_lfo_val);
            if (shape_mod < 0.0f) shape_mod = 0.0f;
            wet_l *= shape_mod;
            wet_r *= shape_mod;

            /* Advance shape LFO — 1 cycle per subdivision */
            inst->shape_lfo_phase += 1.0f / (float)inst->subdivision_samples;
            if (inst->shape_lfo_phase >= 1.0f) inst->shape_lfo_phase -= 1.0f;
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

        /* 8. Mix dry/wet (Interrupt algorithm forces 100% wet during events) */
        float mix = inst->mix;
        if (inst->algorithm == 7 && inst->interrupt_active) {
            mix = 1.0f;
        }
        float out_l = dry_l * (1.0f - mix) + wet_l * mix;
        float out_r = dry_r * (1.0f - mix) + wet_r * mix;

        /* 9. Soft clip with tanh for musical saturation (bypass at mix=0 for clean passthrough) */
        if (mix > 0.001f) {
            out_l = tanhf(out_l * 1.5f) * 0.667f;
            out_r = tanhf(out_r * 1.5f) * 0.667f;
        }

        audio_inout[i * 2]     = (int16_t)(out_l * 32767.0f);
        audio_inout[i * 2 + 1] = (int16_t)(out_r * 32767.0f);
    }
}

static void v2_set_param(void *instance, const char *key, const char *val) {
    if (!instance || !key || !val) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    if (strcmp(key, "algorithm") == 0) {
        int idx = find_enum_index(val, algorithm_names, NUM_ALGORITHMS);
        if (idx >= 0 && idx != inst->algorithm) {
            /* Crossfade to new algorithm */
            inst->pending_algorithm = idx;
            inst->pending_variation = inst->variation;
            inst->crossfade_active = 1;
            inst->crossfade_counter = 0;
            inst->crossfade_level = 1.0f;
        }
    } else if (strcmp(key, "variation") == 0) {
        int idx = find_enum_index(val, variation_names, NUM_VARIATIONS);
        if (idx >= 0 && idx != inst->variation) {
            /* Crossfade to new variation */
            inst->pending_algorithm = inst->algorithm;
            inst->pending_variation = idx;
            inst->crossfade_active = 1;
            inst->crossfade_counter = 0;
            inst->crossfade_level = 1.0f;
        }
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
        /* Mark delay taps dirty after state restore */
        inst->delay_taps_dirty = 1;
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
            "{\"key\":\"algorithm\",\"name\":\"Algorithm\",\"type\":\"enum\",\"options\":[\"Mosaic\",\"Seq\",\"Glide\",\"Haze\",\"Tunnel\",\"Strum\",\"Blocks\",\"Interrupt\",\"Arp\",\"Pattern\",\"Warp\",\"Ethereal\"]},"
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
            "{\"key\":\"reverb_mode\",\"name\":\"Reverb Mode\",\"type\":\"enum\",\"options\":[\"Bright\",\"Dark\",\"Hall\",\"Ambient\"]},"
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
    if (!instance || !msg || len < 1) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;
    uint8_t status = msg[0];

    if (status == 0xF8) {
        /* MIDI timing clock: 24 ppqn */
        inst->midi_clock_tick_count++;
        inst->midi_clock_sample_counter += 128; /* approximate: 1 tick per process block */

        /* Every 24 ticks = 1 beat, derive BPM */
        if (inst->midi_clock_tick_count >= 24) {
            if (inst->midi_clock_sample_counter > 0) {
                float beat_seconds = (float)inst->midi_clock_sample_counter / (float)SAMPLE_RATE;
                float derived_bpm = 60.0f / beat_seconds;
                /* Sanity check: 30-300 BPM */
                if (derived_bpm >= 30.0f && derived_bpm <= 300.0f) {
                    inst->bpm = derived_bpm;
                }
            }
            inst->midi_clock_tick_count = 0;
            inst->midi_clock_sample_counter = 0;
        }
    }

    (void)source;
    (void)len;
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
