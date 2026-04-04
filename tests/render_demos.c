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
                          int16_t *input, int input_frames, int tail_frames,
                          const char *output_dir) {
    void *inst = api->create_instance(NULL, NULL);
    api->set_param(inst, "state", preset->state);

    int total_frames = input_frames + tail_frames;
    int total_samples = total_frames * 2;
    int16_t *output = (int16_t *)malloc(total_samples * sizeof(int16_t));

    /* Process in 128-frame blocks */
    int frames_done = 0;
    while (frames_done < total_frames) {
        int chunk = 128;
        if (frames_done + chunk > total_frames) chunk = total_frames - frames_done;

        if (frames_done < input_frames) {
            /* Copy input block (signal portion) */
            int avail = input_frames - frames_done;
            if (avail > chunk) avail = chunk;
            memcpy(&output[frames_done * 2], &input[frames_done * 2], avail * 2 * sizeof(int16_t));
            /* Zero-fill if chunk extends past input */
            if (avail < chunk) {
                memset(&output[(frames_done + avail) * 2], 0, (chunk - avail) * 2 * sizeof(int16_t));
            }
        } else {
            /* Tail portion: feed silence to let effects ring out */
            memset(&output[frames_done * 2], 0, chunk * 2 * sizeof(int16_t));
        }

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

    /* 16 seconds of musical input + 10 seconds of tail (silence) */
    int input_sec = 16;
    int tail_sec = 10;
    int input_frames = 44100 * input_sec;
    int tail_frames = 44100 * tail_sec;
    int total_frames = input_frames + tail_frames;
    int16_t *input = (int16_t *)malloc(input_frames * 2 * sizeof(int16_t));

    printf("Generating %d seconds of musical input + %d seconds tail...\n", input_sec, tail_sec);
    generate_musical_input(input, input_frames);

    /* Save dry input (no tail needed) */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/00-dry-input.wav", output_dir);
        write_wav(path, input, input_frames * 2);
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
    /* Presets exploring "space crystal smearing across space-time,
       voices of angels carrying music to a new dimension."
       Built from the user's favorites: Tunnel overtones, Mosaic reverse,
       ambient wash, Haze shimmer, and Blocks pitch-glitch for dynamism. */
    /* All Ethereal algorithm (11) — exploring the parameter space.
       Activity = sparse/delicate ↔ thick/enveloping
       Filter = warm glow ↔ bright crystal
       Repeats = distinct events ↔ continuous wash
       Variation = A balanced, B cloud, C drone, D sparkle */
    preset_t presets[] = {
        /* === THE SWEET SPOT — balanced, musical, immediately beautiful === */
        {"01-ethereal-A-sweetspot",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.7,\"mix\":0.75,\"space\":0.75,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.06,"
         "\"filter_res\":0.15,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* === ACTIVITY SWEEP — sparse to dense === */
        {"02-ethereal-sparse-delicate",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.25,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.75,\"mix\":0.7,\"space\":0.75,"
         "\"time_div\":2,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.05,"
         "\"filter_res\":0.1,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"03-ethereal-dense-enveloping",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.9,\"repeats\":0.85,"
         "\"shape\":1.0,\"filter\":0.65,\"mix\":0.85,\"space\":0.85,"
         "\"time_div\":2,\"pitch_mod_depth\":0.1,\"pitch_mod_rate\":0.05,"
         "\"filter_res\":0.2,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* === FILTER SWEEP — warm amber to bright crystal === */
        {"04-ethereal-warm-glow",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.45,\"mix\":0.75,\"space\":0.8,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.06,"
         "\"filter_res\":0.3,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"05-ethereal-bright-crystal",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.75,\"space\":0.8,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.06,"
         "\"filter_res\":0.05,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* === FOUR VARIATIONS === */
        {"06-ethereal-B-cloud-wash",
         "{\"algorithm\":11,\"variation\":1,\"activity\":0.65,\"repeats\":0.75,"
         "\"shape\":1.0,\"filter\":0.65,\"mix\":0.8,\"space\":0.8,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.06,"
         "\"filter_res\":0.2,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"07-ethereal-C-angel-drone",
         "{\"algorithm\":11,\"variation\":2,\"activity\":0.65,\"repeats\":0.8,"
         "\"shape\":1.0,\"filter\":0.6,\"mix\":0.8,\"space\":0.85,"
         "\"time_div\":4,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.04,"
         "\"filter_res\":0.2,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"08-ethereal-D-crystal-sparkle",
         "{\"algorithm\":11,\"variation\":3,\"activity\":0.7,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.8,\"mix\":0.75,\"space\":0.75,"
         "\"time_div\":2,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.08,"
         "\"filter_res\":0.1,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* === SHOWCASE — pushing the boundaries === */
        {"09-ethereal-event-horizon",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.95,\"repeats\":0.9,"
         "\"shape\":1.0,\"filter\":0.5,\"mix\":0.9,\"space\":0.95,"
         "\"time_div\":4,\"pitch_mod_depth\":0.12,\"pitch_mod_rate\":0.03,"
         "\"filter_res\":0.25,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        {"10-ethereal-hall-bright",
         "{\"algorithm\":11,\"variation\":0,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.8,\"mix\":0.7,\"space\":0.65,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.06,"
         "\"filter_res\":0.1,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},
    };

    int num_presets = sizeof(presets) / sizeof(presets[0]);
    printf("\nRendering %d presets (%ds input + %ds tail = %ds each)...\n\n", num_presets, input_sec, tail_sec, input_sec + tail_sec);

    for (int i = 0; i < num_presets; i++) {
        render_preset(api, &presets[i], input, input_frames, tail_frames, output_dir);
    }

    printf("\nDone! %d WAV files saved to %s\n", num_presets + 1, output_dir);

    free(input);
    return 0;
}
