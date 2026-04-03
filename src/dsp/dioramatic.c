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

    if (g_host && g_host->log) {
        g_host->log("dioramatic: instance created (passthrough)");
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
    /* Passthrough — audio passes through unchanged */
    (void)instance;
    (void)audio_inout;
    (void)frames;
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
