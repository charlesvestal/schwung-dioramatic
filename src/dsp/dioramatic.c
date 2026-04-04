/*
 * Dioramatic — Granular Shimmer Reverb
 *
 * A shimmer reverb where the pitch shifter in the feedback loop is GRANULAR.
 * This creates a reverb tail made of sparkly crystalline reflections that
 * cascade upward with each cycle. The Valhalla/Eno/Lanois architecture.
 *
 * 8 controls: Space, Shimmer, Smear, Warmth, Drift, Sustain, Scatter, Mix
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_fx_api_v2.h"

#define SR 44100
#define FDN_SIZE 8192
#define FDN_LINES 4
#define SHIM_BUF 8192         /* shimmer granular pitch shift buffer */
#define SHIM_MAX_GRAINS 16    /* grains inside the pitch shifter */
#define AP_STAGES 4
#define AP_MAX 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================ */

typedef struct { float ic1, ic2; } svf_t;

/* A grain inside the shimmer pitch shifter (lives in the feedback loop) */
typedef struct {
    int active;
    float read_pos;   /* fractional position in shimmer buffer */
    float speed;      /* 2.0 = octave up */
    float env_phase;
    float env_inc;
    float amp;
    float pan_l, pan_r;
    float detune;
} shim_grain_t;

typedef struct {
    /* 8 musical params */
    float space, shimmer, smear, warmth, drift, sustain, scatter, mix;

    /* FDN reverb delay lines */
    float lines[FDN_LINES][FDN_SIZE];
    int line_pos[FDN_LINES];
    float line_lp[FDN_LINES];   /* per-line damping state */
    float line_mod[FDN_LINES];  /* modulation phase */

    /* Input allpass diffusers */
    float ap_buf[AP_STAGES][AP_MAX];
    int ap_pos[AP_STAGES];

    /* Feedback allpass diffusers (for density in the tail) */
    float fb_ap_buf[2][512];
    int fb_ap_pos[2];
    float fb_ap_mod[2];

    /* Shimmer: granular pitch shifter in the feedback loop */
    float shim_buf_l[SHIM_BUF];
    float shim_buf_r[SHIM_BUF];
    int shim_write;
    shim_grain_t shim_grains[SHIM_MAX_GRAINS];
    int shim_trigger_timer;

    /* Stereo decorrelation */
    float stereo_buf[1024];
    int stereo_pos;

    /* Sparkle shelf */
    float sparkle_l, sparkle_r;

    /* SVF output filter */
    svf_t svf_l, svf_r;

    /* DC blocker */
    float dc_in_l, dc_in_r, dc_out_l, dc_out_r;

    /* RNG */
    uint32_t rng;
} dioramatic_instance_t;

static const host_api_v1_t *g_host = NULL;
static audio_fx_api_v2_t g_api;

/* ============================================================================ */

static inline float rngf(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)(*s >> 8) / 16777216.0f;
}

static int json_num(const char *j, const char *k, float *o) {
    char s[64]; snprintf(s, 64, "\"%s\":", k);
    const char *p = strstr(j, s);
    if (!p) return -1;
    p += strlen(s); while (*p == ' ') p++;
    if (*p == '"') p++;
    *o = (float)atof(p); return 0;
}

/* Trigger a shimmer grain (inside the pitch shifter) */
static void shim_trigger(dioramatic_instance_t *inst, float grain_size_ms) {
    for (int i = 0; i < SHIM_MAX_GRAINS; i++) {
        shim_grain_t *g = &inst->shim_grains[i];
        if (g->active) continue;

        int grain_len = (int)(SR * grain_size_ms / 1000.0f);
        if (grain_len < 128) grain_len = 128;
        if (grain_len > SHIM_BUF / 2) grain_len = SHIM_BUF / 2;

        /* Start reading from a random recent position in the shimmer buffer */
        int offset = (int)(rngf(&inst->rng) * (float)(SHIM_BUF / 2));
        g->read_pos = (float)((inst->shim_write - grain_len - offset + SHIM_BUF) & (SHIM_BUF - 1));
        g->speed = 2.0f;  /* octave up */
        g->detune = 1.0f + (rngf(&inst->rng) - 0.5f) * 0.012f;  /* ±10 cents */
        g->env_phase = 0.0f;
        g->env_inc = 1.0f / (float)grain_len;
        g->amp = 0.5f + rngf(&inst->rng) * 0.3f;
        float pan = (rngf(&inst->rng) - 0.5f) * inst->scatter * 0.7f;
        g->pan_l = sqrtf(0.5f - pan * 0.5f);
        g->pan_r = sqrtf(0.5f + pan * 0.5f);
        g->active = 1;
        return;
    }
}

/* ============================================================================ */

static void *create(const char *dir, const char *cfg) {
    (void)dir; (void)cfg;
    dioramatic_instance_t *inst = calloc(1, sizeof(dioramatic_instance_t));
    if (!inst) return NULL;
    inst->space = 0.55f; inst->shimmer = 0.35f; inst->smear = 0.40f;
    inst->warmth = 0.45f; inst->drift = 0.35f; inst->sustain = 0.50f;
    inst->scatter = 0.40f; inst->mix = 0.60f;
    inst->rng = 12345;
    if (g_host && g_host->log) g_host->log("dioramatic: created");
    return inst;
}

static void destroy(void *inst) { if (inst) free(inst); }

static void process(void *instance, int16_t *audio, int frames) {
    if (!instance || !audio) return;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    /* === DERIVE DSP PARAMS FROM KNOBS === */

    /* Space: reverb delay line lengths */
    int lengths[FDN_LINES] = {
        1087 + (int)(inst->space * 3000.0f),
        1283 + (int)(inst->space * 3500.0f),
        1447 + (int)(inst->space * 3800.0f),
        1663 + (int)(inst->space * 4200.0f)
    };
    for (int i = 0; i < FDN_LINES; i++)
        if (lengths[i] >= FDN_SIZE) lengths[i] = FDN_SIZE - 1;

    /* Sustain: reverb feedback. Exponential curve — top end approaches unity */
    float feedback = 1.0f - 0.35f * powf(1.0f - inst->sustain, 2.0f);
    if (feedback > 0.998f) feedback = 0.998f;

    /* Shimmer: how much octave-up feeds back. Tapered so high values
       don't overwhelm the reverb — the direct output carries the sparkle. */
    float shim_level = inst->shimmer * 0.3f;

    /* Smear: grain size in the pitch shifter. Small=sparkly, large=smooth */
    float grain_ms = 5.0f + inst->smear * 95.0f;  /* 5ms to 100ms */

    /* Scatter: number of simultaneous shimmer grains + trigger rate */
    int shim_trigger_interval = (int)(4410.0f / (1.0f + inst->scatter * 8.0f));
    if (shim_trigger_interval < 128) shim_trigger_interval = 128;

    /* Warmth: damping in feedback (inverted: more warmth = more HF absorption) */
    float damping = (0.1f + inst->warmth * 0.6f) * (1.0f - inst->sustain * 0.5f);

    /* Drift: modulation depth on delay lines */
    float mod_depth = 4.0f + inst->drift * 24.0f;
    static const float mod_rates[FDN_LINES] = {0.37f, 0.47f, 0.31f, 0.53f};

    /* SVF filter cutoff from Warmth */
    float cut = 200.0f * powf(100.0f, 1.0f - inst->warmth);
    if (cut > 20000.0f) cut = 20000.0f;
    float svfg = tanf((float)M_PI * cut / (float)SR);
    float svfk = 2.0f - 0.19f;
    float a1 = 1.0f / (1.0f + svfg * (svfg + svfk));
    float a2 = svfg * a1;
    float a3 = svfg * a2;

    /* Reverb send: Space controls how much goes in */
    float rev_send = 0.3f + inst->space * 0.7f;

    /* Allpass diffuser lengths */
    static const int ap_lens[AP_STAGES] = {241, 173, 419, 313};
    static const int fb_ap_lens[2] = {347, 461};

    /* Pitch mod LFO */
    float lfo_rate = 0.05f + inst->drift * 0.4f;
    float lfo_depth = inst->drift * 0.12f;

    for (int i = 0; i < frames; i++) {
        float dry_l = (float)audio[i * 2] / 32768.0f;
        float dry_r = (float)audio[i * 2 + 1] / 32768.0f;

        /* === INPUT DIFFUSION === */
        float in_mono = (dry_l + dry_r) * 0.5f * rev_send;
        float diff = in_mono;
        for (int a = 0; a < AP_STAGES; a++) {
            float del = inst->ap_buf[a][inst->ap_pos[a]];
            float out = del - 0.5f * diff;
            inst->ap_buf[a][inst->ap_pos[a]] = diff + 0.5f * out;
            diff = out;
            inst->ap_pos[a] = (inst->ap_pos[a] + 1) % ap_lens[a];
        }

        /* === READ FDN DELAY LINES (modulated) === */
        float taps[FDN_LINES];
        for (int l = 0; l < FDN_LINES; l++) {
            float mod = sinf(2.0f * (float)M_PI * inst->line_mod[l]) * mod_depth;
            float fd = (float)lengths[l] + mod;
            if (fd < 1.0f) fd = 1.0f;
            if (fd >= (float)(FDN_SIZE - 1)) fd = (float)(FDN_SIZE - 2);
            int ri = (int)fd;
            float rf = fd - (float)ri;
            int p0 = (inst->line_pos[l] - ri + FDN_SIZE) & (FDN_SIZE - 1);
            int p1 = (p0 - 1 + FDN_SIZE) & (FDN_SIZE - 1);
            taps[l] = inst->lines[l][p0] * (1.0f - rf) + inst->lines[l][p1] * rf;
            inst->line_mod[l] += mod_rates[l] / (float)SR;
            if (inst->line_mod[l] >= 1.0f) inst->line_mod[l] -= 1.0f;
        }

        /* === HADAMARD MIX === */
        float mx[FDN_LINES];
        mx[0] = 0.5f * (taps[0] + taps[1] + taps[2] + taps[3]);
        mx[1] = 0.5f * (taps[0] - taps[1] + taps[2] - taps[3]);
        mx[2] = 0.5f * (taps[0] + taps[1] - taps[2] - taps[3]);
        mx[3] = 0.5f * (taps[0] - taps[1] - taps[2] + taps[3]);

        /* === GRANULAR SHIMMER PITCH SHIFTER (in the feedback loop) ===
           Write the mixed feedback into the shimmer buffer. Then read it
           back with multiple overlapping grains at 2x speed (octave up).
           This is what creates the cascading crystalline harmonics. */

        float fb_mono = (mx[0] + mx[1] + mx[2] + mx[3]) * 0.25f;
        inst->shim_buf_l[inst->shim_write] = (taps[0] + taps[2]) * 0.5f;
        inst->shim_buf_r[inst->shim_write] = (taps[1] + taps[3]) * 0.5f;
        inst->shim_write = (inst->shim_write + 1) & (SHIM_BUF - 1);

        /* Trigger new shimmer grains at a rate controlled by Scatter */
        inst->shim_trigger_timer++;
        if (inst->shim_trigger_timer >= shim_trigger_interval) {
            inst->shim_trigger_timer = 0;
            shim_trigger(inst, grain_ms);
        }

        /* Read shimmer grains — the sparkly octave-up layer */
        float shim_l = 0.0f, shim_r = 0.0f;
        for (int g = 0; g < SHIM_MAX_GRAINS; g++) {
            shim_grain_t *gr = &inst->shim_grains[g];
            if (!gr->active) continue;

            int ri = (int)gr->read_pos;
            float rf = gr->read_pos - (float)ri;
            int i0 = ri & (SHIM_BUF - 1);
            int i1 = (ri + 1) & (SHIM_BUF - 1);
            float sl = inst->shim_buf_l[i0] * (1.0f - rf) + inst->shim_buf_l[i1] * rf;
            float sr = inst->shim_buf_r[i0] * (1.0f - rf) + inst->shim_buf_r[i1] * rf;

            /* Hann envelope */
            float env = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * gr->env_phase));

            float a = gr->amp * env;
            shim_l += sl * a * gr->pan_l;
            shim_r += sr * a * gr->pan_r;

            gr->read_pos += gr->speed * gr->detune;
            /* Wrap */
            while (gr->read_pos >= (float)SHIM_BUF) gr->read_pos -= (float)SHIM_BUF;
            while (gr->read_pos < 0) gr->read_pos += (float)SHIM_BUF;

            gr->env_phase += gr->env_inc;
            if (gr->env_phase >= 1.0f) gr->active = 0;
        }

        /* === FEEDBACK ALLPASS DIFFUSION (for density) === */
        float fb_diff = fb_mono;
        for (int a = 0; a < 2; a++) {
            float mod = sinf(2.0f * (float)M_PI * inst->fb_ap_mod[a]) * 3.0f;
            int rd = (inst->fb_ap_pos[a] - fb_ap_lens[a] - (int)mod + 512) & 511;
            float del = inst->fb_ap_buf[a][rd];
            float out = del - 0.45f * fb_diff;
            inst->fb_ap_buf[a][inst->fb_ap_pos[a]] = fb_diff + 0.45f * out;
            fb_diff = out;
            inst->fb_ap_pos[a] = (inst->fb_ap_pos[a] + 1) & 511;
            inst->fb_ap_mod[a] += ((a == 0) ? 0.23f : 0.29f) / (float)SR;
            if (inst->fb_ap_mod[a] >= 1.0f) inst->fb_ap_mod[a] -= 1.0f;
        }

        /* === WRITE BACK TO DELAY LINES ===
           feedback = mixed × feedback_coeff + shimmer + diffusion
           The shimmer grains create the cascading octave harmonics.
           Each cycle through the reverb shifts them up another octave. */
        for (int l = 0; l < FDN_LINES; l++) {
            float fb = mx[l] * feedback;
            /* Damping: one-pole LP in feedback */
            inst->line_lp[l] += damping * (fb - inst->line_lp[l]);
            fb = inst->line_lp[l];
            /* Add shimmer (granular octave-up) */
            fb += ((l < 2) ? shim_l : shim_r) * shim_level;
            /* Add diffusion */
            fb += fb_diff * 0.2f;
            /* Soft limit — transparent at normal levels */
            if (fb > 0.7f || fb < -0.7f)
                fb = tanhf(fb) * 0.98f;
            /* Write */
            inst->lines[l][inst->line_pos[l]] = diff + fb;
            inst->line_pos[l] = (inst->line_pos[l] + 1) & (FDN_SIZE - 1);
        }

        /* === OUTPUT === */
        float rev_l = (taps[0] + taps[2]) * 0.6f;
        float rev_r = (taps[1] + taps[3]) * 0.6f;

        /* Add shimmer grains directly to output — heard as distinct sparkles
           on top of the reverb wash, not just buried inside the feedback.
           This is the "trail of grains careening into space." */
        rev_l += shim_l * inst->shimmer * 0.4f;
        rev_r += shim_r * inst->shimmer * 0.4f;

        /* Stereo decorrelation on right (~8ms) */
        inst->stereo_buf[inst->stereo_pos] = rev_r;
        int srd = (inst->stereo_pos - 353 + 1024) & 1023;
        rev_r = inst->stereo_buf[srd];
        inst->stereo_pos = (inst->stereo_pos + 1) & 1023;

        /* Sparkle shelf: gentle HF boost */
        float hf_l = rev_l - inst->sparkle_l;
        inst->sparkle_l += 0.15f * hf_l;
        float hf_r = rev_r - inst->sparkle_r;
        inst->sparkle_r += 0.15f * hf_r;
        rev_l += hf_l * 0.15f;
        rev_r += hf_r * 0.15f;

        /* SVF lowpass on output */
        {
            float v3 = rev_l - inst->svf_l.ic2;
            float v1 = a1 * inst->svf_l.ic1 + a2 * v3;
            float v2 = inst->svf_l.ic2 + a2 * inst->svf_l.ic1 + a3 * v3;
            inst->svf_l.ic1 = 2.0f * v1 - inst->svf_l.ic1;
            inst->svf_l.ic2 = 2.0f * v2 - inst->svf_l.ic2;
            rev_l = v2;
        }
        {
            float v3 = rev_r - inst->svf_r.ic2;
            float v1 = a1 * inst->svf_r.ic1 + a2 * v3;
            float v2 = inst->svf_r.ic2 + a2 * inst->svf_r.ic1 + a3 * v3;
            inst->svf_r.ic1 = 2.0f * v1 - inst->svf_r.ic1;
            inst->svf_r.ic2 = 2.0f * v2 - inst->svf_r.ic2;
            rev_r = v2;
        }

        /* Mix */
        float out_l = dry_l * (1.0f - inst->mix) + rev_l * inst->mix;
        float out_r = dry_r * (1.0f - inst->mix) + rev_r * inst->mix;

        /* DC blocker */
        float dc_l = out_l - inst->dc_in_l + 0.9975f * inst->dc_out_l;
        float dc_r = out_r - inst->dc_in_r + 0.9975f * inst->dc_out_r;
        inst->dc_in_l = out_l; inst->dc_in_r = out_r;
        inst->dc_out_l = dc_l; inst->dc_out_r = dc_r;

        /* Soft clip */
        out_l = tanhf(dc_l * 1.5f) * 0.667f;
        out_r = tanhf(dc_r * 1.5f) * 0.667f;

        audio[i * 2]     = (int16_t)(out_l * 32767.0f);
        audio[i * 2 + 1] = (int16_t)(out_r * 32767.0f);
    }
}

/* ============================================================================ */

static void set_param(void *instance, const char *key, const char *val) {
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
        if (json_num(val, "space", &f) == 0) inst->space = f;
        if (json_num(val, "shimmer", &f) == 0) inst->shimmer = f;
        if (json_num(val, "smear", &f) == 0) inst->smear = f;
        if (json_num(val, "warmth", &f) == 0) inst->warmth = f;
        if (json_num(val, "drift", &f) == 0) inst->drift = f;
        if (json_num(val, "sustain", &f) == 0) inst->sustain = f;
        if (json_num(val, "scatter", &f) == 0) inst->scatter = f;
        if (json_num(val, "mix", &f) == 0) inst->mix = f;
    }
}

static int get_param(void *instance, const char *key, char *buf, int len) {
    if (!instance || !key || !buf || len <= 0) return -1;
    dioramatic_instance_t *inst = (dioramatic_instance_t *)instance;

    if (strcmp(key, "name") == 0) return snprintf(buf, len, "Dioramatic");
    if (strcmp(key, "space") == 0) return snprintf(buf, len, "%.3f", inst->space);
    if (strcmp(key, "shimmer") == 0) return snprintf(buf, len, "%.3f", inst->shimmer);
    if (strcmp(key, "smear") == 0) return snprintf(buf, len, "%.3f", inst->smear);
    if (strcmp(key, "warmth") == 0) return snprintf(buf, len, "%.3f", inst->warmth);
    if (strcmp(key, "drift") == 0) return snprintf(buf, len, "%.3f", inst->drift);
    if (strcmp(key, "sustain") == 0) return snprintf(buf, len, "%.3f", inst->sustain);
    if (strcmp(key, "scatter") == 0) return snprintf(buf, len, "%.3f", inst->scatter);
    if (strcmp(key, "mix") == 0) return snprintf(buf, len, "%.3f", inst->mix);
    if (strcmp(key, "state") == 0)
        return snprintf(buf, len,
            "{\"space\":%.3f,\"shimmer\":%.3f,\"smear\":%.3f,\"warmth\":%.3f,"
            "\"drift\":%.3f,\"sustain\":%.3f,\"scatter\":%.3f,\"mix\":%.3f}",
            inst->space, inst->shimmer, inst->smear, inst->warmth,
            inst->drift, inst->sustain, inst->scatter, inst->mix);
    if (strcmp(key, "chain_params") == 0)
        return snprintf(buf, len,
            "[{\"key\":\"space\",\"name\":\"Space\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"shimmer\",\"name\":\"Shimmer\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"smear\",\"name\":\"Smear\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"warmth\",\"name\":\"Warmth\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"drift\",\"name\":\"Drift\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"sustain\",\"name\":\"Sustain\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"scatter\",\"name\":\"Scatter\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"mix\",\"name\":\"Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01}]");
    if (strcmp(key, "ui_hierarchy") == 0)
        return snprintf(buf, len,
            "{\"modes\":null,\"levels\":{\"root\":{\"children\":null,"
            "\"knobs\":[\"space\",\"shimmer\",\"smear\",\"warmth\",\"drift\",\"sustain\",\"scatter\",\"mix\"],"
            "\"params\":[\"space\",\"shimmer\",\"smear\",\"warmth\",\"drift\",\"sustain\",\"scatter\",\"mix\"]}}}");
    return -1;
}

static void on_midi(void *inst, const uint8_t *msg, int len, int src) {
    (void)inst; (void)msg; (void)len; (void)src;
}

audio_fx_api_v2_t *move_audio_fx_init_v2(const host_api_v1_t *host) {
    g_host = host;
    g_api.api_version = 2;
    g_api.create_instance = create;
    g_api.destroy_instance = destroy;
    g_api.process_block = process;
    g_api.set_param = set_param;
    g_api.get_param = get_param;
    g_api.on_midi = on_midi;
    if (host && host->log) host->log("dioramatic: init");
    return &g_api;
}
