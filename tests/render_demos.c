/*
 * Dioramatic Demo Renderer
 *
 * Generates a musical input signal (chord arpeggios + melody) and renders it
 * through various algorithm/preset combinations, saving each as a WAV file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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
 * WAV writer
 * ============================================================================ */

static void write_wav(const char *path, int16_t *samples, int num_samples) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }

    int data_size = num_samples * 2;  /* int16 = 2 bytes */
    int file_size = 36 + data_size;

    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    uint32_t v = file_size; fwrite(&v, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    v = 16; fwrite(&v, 4, 1, f);           /* chunk size */
    uint16_t s16 = 1; fwrite(&s16, 2, 1, f); /* PCM */
    s16 = 2; fwrite(&s16, 2, 1, f);          /* stereo */
    v = 44100; fwrite(&v, 4, 1, f);          /* sample rate */
    v = 44100 * 2 * 2; fwrite(&v, 4, 1, f);  /* byte rate */
    s16 = 4; fwrite(&s16, 2, 1, f);          /* block align */
    s16 = 16; fwrite(&s16, 2, 1, f);         /* bits per sample */

    /* data chunk */
    fwrite("data", 1, 4, f);
    v = data_size; fwrite(&v, 4, 1, f);
    fwrite(samples, 2, num_samples, f);

    fclose(f);
}

/* ============================================================================
 * Musical input generator
 * ============================================================================ */

/* MIDI note to frequency */
static float mtof(int note) {
    return 440.0f * powf(2.0f, (float)(note - 69) / 12.0f);
}

/* Simple polyBLEP sawtooth for richer harmonics */
static float polyblep(float t, float dt) {
    if (t < dt) {
        t /= dt;
        return t + t - t * t - 1.0f;
    } else if (t > 1.0f - dt) {
        t = (t - 1.0f) / dt;
        return t * t + t + t + 1.0f;
    }
    return 0.0f;
}

typedef struct {
    float phase;
    float freq;
    float amp;
    float amp_env;    /* current amplitude envelope */
    float env_target;
    float env_rate;
    int active;
} voice_t;

#define MAX_VOICES 6

typedef struct {
    voice_t voices[MAX_VOICES];
    int sample_pos;
} synth_t;

static void synth_init(synth_t *s) {
    memset(s, 0, sizeof(*s));
}

static void synth_note_on(synth_t *s, int note, float amp) {
    /* Find free voice or steal quietest */
    int best = 0;
    float best_amp = 999.0f;
    for (int i = 0; i < MAX_VOICES; i++) {
        if (!s->voices[i].active) { best = i; break; }
        if (s->voices[i].amp_env < best_amp) { best_amp = s->voices[i].amp_env; best = i; }
    }
    voice_t *v = &s->voices[best];
    v->freq = mtof(note);
    v->amp = amp;
    v->amp_env = 0.001f;
    v->env_target = amp;
    v->env_rate = 0.002f;  /* ~10ms attack */
    v->active = 1;
    v->phase = 0.0f;
}

static void synth_note_off_all(synth_t *s) {
    for (int i = 0; i < MAX_VOICES; i++) {
        if (s->voices[i].active) {
            s->voices[i].env_target = 0.0f;
            s->voices[i].env_rate = 0.0003f;  /* ~75ms release */
        }
    }
}

static float synth_tick(synth_t *s) {
    float out = 0.0f;
    for (int i = 0; i < MAX_VOICES; i++) {
        voice_t *v = &s->voices[i];
        if (!v->active) continue;

        float dt = v->freq / 44100.0f;

        /* Sawtooth with polyBLEP */
        float saw = 2.0f * v->phase - 1.0f;
        saw -= polyblep(v->phase, dt);

        /* Add a softer square component for warmth */
        float sq = (v->phase < 0.5f) ? 1.0f : -1.0f;
        /* polyBLEP for square edges */
        float phase2 = fmodf(v->phase + 0.5f, 1.0f);
        sq += polyblep(v->phase, dt);
        sq -= polyblep(phase2, dt);

        out += (saw * 0.6f + sq * 0.3f) * v->amp_env;

        v->phase += dt;
        if (v->phase >= 1.0f) v->phase -= 1.0f;

        /* Envelope */
        v->amp_env += (v->env_target - v->amp_env) * v->env_rate;
        if (v->env_target < 0.001f && v->amp_env < 0.0001f) {
            v->active = 0;
        }
    }
    return out;
}

/*
 * Generate 8 seconds of musical content:
 * - Bars 1-2: Cmaj arpeggio (C4-E4-G4-C5 repeated)
 * - Bars 3-4: Am arpeggio (A3-C4-E4-A4)
 * - Bars 5-6: F maj chord stabs
 * - Bars 7-8: Melodic line (C D E G A G E D)
 */
static void generate_musical_input(int16_t *buf, int total_frames) {
    synth_t synth;
    synth_init(&synth);

    /* At 120 BPM, 1 beat = 22050 samples, 1 bar = 88200 samples */
    int beat = 22050;

    /* Note sequences (MIDI notes) */
    /* Cmaj arp: C4=60, E4=64, G4=67, C5=72 */
    int cmaj_arp[] = {60, 64, 67, 72, 67, 64, 60, 64};
    /* Am arp: A3=57, C4=60, E4=64, A4=69 */
    int am_arp[] = {57, 60, 64, 69, 64, 60, 57, 60};
    /* F chord stabs: F3=53, A3=57, C4=60 */
    int fchord[] = {53, 57, 60};
    /* Melody */
    int melody[] = {60, 62, 64, 67, 69, 67, 64, 62};

    for (int i = 0; i < total_frames; i++) {
        int bar = i / (beat * 4);
        int beat_in_bar = (i % (beat * 4)) / beat;
        int pos_in_beat = i % beat;

        /* Trigger notes at beat boundaries */
        if (pos_in_beat == 0) {
            synth_note_off_all(&synth);

            if (bar < 2) {
                /* Cmaj arpeggio, 8th notes */
                int eighth = (i % (beat * 4)) / (beat / 2);
                if (eighth < 8) {
                    synth_note_on(&synth, cmaj_arp[eighth % 8], 0.35f);
                }
            } else if (bar < 4) {
                /* Am arpeggio */
                int eighth = (i % (beat * 4)) / (beat / 2);
                if (eighth < 8) {
                    synth_note_on(&synth, am_arp[eighth % 8], 0.35f);
                }
            } else if (bar < 6) {
                /* F chord stabs on beats 1 and 3 */
                if (beat_in_bar == 0 || beat_in_bar == 2) {
                    for (int n = 0; n < 3; n++)
                        synth_note_on(&synth, fchord[n], 0.25f);
                }
            } else {
                /* Melody, quarter notes */
                if (beat_in_bar < 4) {
                    int note_idx = ((bar - 6) * 4 + beat_in_bar) % 8;
                    synth_note_on(&synth, melody[note_idx], 0.4f);
                }
            }

            /* Also trigger 8th note arps within beats for bars 0-3 */
        }

        /* Trigger 8th notes for arp sections */
        if (bar < 4 && pos_in_beat == beat / 2) {
            synth_note_off_all(&synth);
            int eighth = (i % (beat * 4)) / (beat / 2);
            if (bar < 2 && eighth < 8) {
                synth_note_on(&synth, cmaj_arp[eighth % 8], 0.35f);
            } else if (bar >= 2 && eighth < 8) {
                synth_note_on(&synth, am_arp[eighth % 8], 0.35f);
            }
        }

        float sample = synth_tick(&synth);

        /* Soft clip */
        if (sample > 0.95f) sample = 0.95f;
        if (sample < -0.95f) sample = -0.95f;

        int16_t s = (int16_t)(sample * 32767.0f);
        buf[i * 2] = s;
        buf[i * 2 + 1] = s;
    }
}

/* ============================================================================
 * Render a preset through the effect
 * ============================================================================ */

typedef struct {
    const char *name;
    const char *state;   /* JSON state string */
} preset_t;

static void render_preset(audio_fx_api_v2_t *api, const preset_t *preset,
                          int16_t *input, int total_frames, const char *output_dir) {
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "state", preset->state);

    /* Allocate output buffer (stereo interleaved) */
    int total_samples = total_frames * 2;
    int16_t *output = (int16_t *)malloc(total_samples * sizeof(int16_t));

    /* Process in 128-frame blocks */
    int frames_done = 0;
    while (frames_done < total_frames) {
        int chunk = 128;
        if (frames_done + chunk > total_frames) chunk = total_frames - frames_done;

        /* Copy input block */
        memcpy(&output[frames_done * 2], &input[frames_done * 2], chunk * 2 * sizeof(int16_t));

        api->process_block(inst, &output[frames_done * 2], chunk);
        frames_done += chunk;
    }

    /* Write WAV */
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.wav", output_dir, preset->name);
    write_wav(path, output, total_samples);
    printf("  Wrote: %s\n", path);

    free(output);
    api->destroy_instance(inst);
}

/* ============================================================================ */

int main(void) {
    audio_fx_api_v2_t *api = move_audio_fx_init_v2(&th);

    const char *output_dir = "/Users/charlesvestal/Desktop/dioramatic-demos";

    /* Create output directory */
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", output_dir);
    system(cmd);

    /* 8 seconds of audio at 44100 Hz */
    int duration_sec = 8;
    int total_frames = 44100 * duration_sec;
    int16_t *input = (int16_t *)malloc(total_frames * 2 * sizeof(int16_t));

    printf("Generating %d seconds of musical input...\n", duration_sec);
    generate_musical_input(input, total_frames);

    /* Save dry input */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/00-dry-input.wav", output_dir);
        write_wav(path, input, total_frames * 2);
        printf("  Wrote: %s\n", path);
    }

    /* Presets tuned for musical results with wide sweet spots.
     * Key principles:
     * - Mix 0.5-0.7 for blending, 0.8+ for drenched
     * - Space (reverb) 0.3-0.5 smooths grain artifacts
     * - time_div 2 (1x) is the most musical default
     * - Pitch mod 0.1-0.2 adds warmth without wobble
     * - Filter < 0.8 tames harshness
     * - Activity/Repeats in the 0.5-0.8 range sound best
     */
    preset_t presets[] = {
        /* === MOSAIC BANK (the signature sound) === */
        {"01-mosaic-A-shimmer",
         "{\"algorithm\":0,\"variation\":0,\"activity\":0.65,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.65,\"space\":0.4,"
         "\"time_div\":2,\"pitch_mod_depth\":0.15,\"pitch_mod_rate\":0.25,"
         "\"filter_res\":0.1,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        {"02-mosaic-D-fullrange",
         "{\"algorithm\":0,\"variation\":3,\"activity\":0.75,\"repeats\":0.75,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.7,\"space\":0.5,"
         "\"time_div\":2,\"pitch_mod_depth\":0.18,\"pitch_mod_rate\":0.2,"
         "\"filter_res\":0.05,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"03-mosaic-B-dark-octavedown",
         "{\"algorithm\":0,\"variation\":1,\"activity\":0.5,\"repeats\":0.75,"
         "\"shape\":1.0,\"filter\":0.45,\"mix\":0.7,\"space\":0.55,"
         "\"time_div\":2,\"pitch_mod_depth\":0.2,\"pitch_mod_rate\":0.12,"
         "\"filter_res\":0.35,\"reverb_mode\":1,\"reverse\":0,\"hold\":0}"},

        /* === SHAPE LFO DEMOS === */
        {"04-mosaic-square-gate",
         "{\"algorithm\":0,\"variation\":0,\"activity\":0.6,\"repeats\":0.65,"
         "\"shape\":0.12,\"filter\":1.0,\"mix\":0.75,\"space\":0.35,"
         "\"time_div\":0,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        {"05-haze-triangle-swell",
         "{\"algorithm\":3,\"variation\":0,\"activity\":0.7,\"repeats\":0.65,"
         "\"shape\":0.6,\"filter\":0.85,\"mix\":0.7,\"space\":0.45,"
         "\"time_div\":2,\"pitch_mod_depth\":0.1,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.15,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        /* === GRANULES BANK === */
        {"06-haze-A-diffuse",
         "{\"algorithm\":3,\"variation\":0,\"activity\":0.75,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.75,\"mix\":0.6,\"space\":0.45,"
         "\"time_div\":2,\"pitch_mod_depth\":0.12,\"pitch_mod_rate\":0.35,"
         "\"filter_res\":0.2,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        {"07-haze-C-shimmer",
         "{\"algorithm\":3,\"variation\":2,\"activity\":0.8,\"repeats\":0.65,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.65,\"space\":0.55,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"08-tunnel-B-overtones",
         "{\"algorithm\":4,\"variation\":1,\"activity\":0.65,\"repeats\":0.75,"
         "\"shape\":1.0,\"filter\":0.65,\"mix\":0.65,\"space\":0.5,"
         "\"time_div\":2,\"pitch_mod_depth\":0.1,\"pitch_mod_rate\":0.12,"
         "\"filter_res\":0.25,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* === MICRO LOOP BANK === */
        {"09-glide-C-updown",
         "{\"algorithm\":2,\"variation\":2,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.6,\"space\":0.4,"
         "\"time_div\":2,\"pitch_mod_depth\":0.05,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.1,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        {"10-seq-B-halftime",
         "{\"algorithm\":1,\"variation\":1,\"activity\":0.7,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.6,\"space\":0.35,"
         "\"time_div\":2,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        {"11-strum-C-cascade",
         "{\"algorithm\":5,\"variation\":2,\"activity\":0.7,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.65,\"space\":0.4,"
         "\"time_div\":2,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        /* === GLITCH BANK === */
        {"12-blocks-C-pitchglitch",
         "{\"algorithm\":6,\"variation\":2,\"activity\":0.75,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":1.0,\"mix\":0.55,\"space\":0.3,"
         "\"time_div\":0,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        {"13-interrupt-A-subtle",
         "{\"algorithm\":7,\"variation\":0,\"activity\":0.5,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.5,\"space\":0.45,"
         "\"time_div\":2,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        {"14-arp-A-ascending",
         "{\"algorithm\":8,\"variation\":0,\"activity\":0.65,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.65,\"space\":0.35,"
         "\"time_div\":0,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        {"15-arp-C-updown-reverbed",
         "{\"algorithm\":8,\"variation\":2,\"activity\":0.7,\"repeats\":0.65,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.7,\"space\":0.5,"
         "\"time_div\":0,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.1,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        /* === MULTI DELAY BANK === */
        {"16-pattern-A-clean",
         "{\"algorithm\":9,\"variation\":0,\"activity\":0.45,\"repeats\":0.4,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.5,\"space\":0.25,"
         "\"time_div\":2,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        {"17-pattern-B-dotted",
         "{\"algorithm\":9,\"variation\":1,\"activity\":0.5,\"repeats\":0.45,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.55,\"space\":0.35,"
         "\"time_div\":2,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        {"18-warp-C-pitchfilter",
         "{\"algorithm\":10,\"variation\":2,\"activity\":0.5,\"repeats\":0.4,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.5,\"space\":0.35,"
         "\"time_div\":2,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":1,\"reverse\":0,\"hold\":0}"},

        /* === SHOWCASE PRESETS === */
        {"19-ambient-wash",
         "{\"algorithm\":0,\"variation\":3,\"activity\":0.85,\"repeats\":0.85,"
         "\"shape\":1.0,\"filter\":0.55,\"mix\":0.8,\"space\":0.85,"
         "\"time_div\":4,\"pitch_mod_depth\":0.1,\"pitch_mod_rate\":0.08,"
         "\"filter_res\":0.2,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"20-mosaic-A-reverse",
         "{\"algorithm\":0,\"variation\":0,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.65,\"space\":0.45,"
         "\"time_div\":2,\"pitch_mod_depth\":0.12,\"pitch_mod_rate\":0.25,"
         "\"filter_res\":0.1,\"reverb_mode\":2,\"reverse\":1,\"hold\":0}"},
    };

    int num_presets = sizeof(presets) / sizeof(presets[0]);
    printf("\nRendering %d presets through %d seconds of audio...\n\n", num_presets, duration_sec);

    for (int i = 0; i < num_presets; i++) {
        render_preset(api, &presets[i], input, total_frames, output_dir);
    }

    printf("\nDone! %d WAV files saved to %s\n", num_presets + 1, output_dir);

    free(input);
    return 0;
}
