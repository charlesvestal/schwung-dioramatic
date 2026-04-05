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
    int waveform;     /* 0=saw+square, 1=pure sine, 2=triangle, 3=filtered saw (mellow) */
    float lp_state;   /* one-pole filter for mellow timbres */
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
        float sample = 0.0f;

        switch (v->waveform) {
            case 0: { /* Saw + square — bright, rich harmonics */
                float saw = 2.0f * v->phase - 1.0f;
                saw -= polyblep(v->phase, dt);
                float sq = (v->phase < 0.5f) ? 1.0f : -1.0f;
                float phase2 = fmodf(v->phase + 0.5f, 1.0f);
                sq += polyblep(v->phase, dt);
                sq -= polyblep(phase2, dt);
                sample = saw * 0.6f + sq * 0.3f;
                break;
            }
            case 1: /* Pure sine — clean, bell-like */
                sample = sinf(2.0f * (float)M_PI * v->phase);
                break;
            case 2: /* Triangle — soft, flute-like */
                sample = (v->phase < 0.5f) ? (4.0f * v->phase - 1.0f) : (3.0f - 4.0f * v->phase);
                break;
            case 3: { /* Filtered saw — mellow, guitar-like */
                float saw = 2.0f * v->phase - 1.0f;
                saw -= polyblep(v->phase, dt);
                v->lp_state += 0.15f * (saw - v->lp_state);
                sample = v->lp_state;
                break;
            }
        }

        out += sample * v->amp_env;

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
 * Input A: Soft piano — sparse chords with space between them.
 * Short stabs that decay, letting the reverb tail breathe.
 * Cmaj → Am → Fmaj → G, one chord every 2 beats, quick release.
 */
static void generate_soft_piano(int16_t *buf, int total_frames) {
    synth_t synth;
    synth_init(&synth);

    /* Slower tempo feel — chords every 2 beats at 90 BPM */
    int beat = 44100 * 60 / 90;  /* ~29400 samples per beat */

    /* Chord voicings (piano-like: spread, mid-register) */
    int chords[4][4] = {
        {60, 64, 67, 72},  /* Cmaj */
        {57, 60, 64, 69},  /* Am */
        {53, 57, 60, 65},  /* Fmaj */
        {55, 59, 62, 67},  /* G */
    };

    /* Make the synth sound softer — override envelope rates */
    for (int i = 0; i < total_frames; i++) {
        int chord_period = beat * 2;  /* one chord every 2 beats */
        int pos_in_chord = i % chord_period;
        int chord_idx = (i / chord_period) % 4;

        if (pos_in_chord == 0) {
            /* Trigger chord with soft attack */
            synth_note_off_all(&synth);
            for (int n = 0; n < 4; n++) {
                synth_note_on(&synth, chords[chord_idx][n], 0.2f);
                synth.voices[n].waveform = 3;  /* filtered saw — mellow piano-like */
                synth.voices[n].env_rate = 0.003f;
            }
        }

        /* Release after 1/4 of the chord period — short stab */
        if (pos_in_chord == chord_period / 4) {
            synth_note_off_all(&synth);
            /* Faster release for piano-like decay */
            for (int v = 0; v < MAX_VOICES; v++) {
                if (synth.voices[v].active) synth.voices[v].env_rate = 0.0008f;
            }
        }

        float sample = synth_tick(&synth);

        /* Gentle soft clip */
        if (sample > 0.7f) sample = 0.7f;
        if (sample < -0.7f) sample = -0.7f;

        /* Reduce saw harshness — use more square (rounder) */
        int16_t s = (int16_t)(sample * 32767.0f);
        buf[i * 2] = s;
        buf[i * 2 + 1] = s;
    }
}

/*
 * Input B: Arpeggiated synth — continuous flowing notes.
 */
static void generate_arp_input(int16_t *buf, int total_frames) {
    synth_t synth;
    synth_init(&synth);

    int beat = 22050;  /* 120 BPM */
    int cmaj_arp[] = {60, 64, 67, 72, 67, 64, 60, 64};
    int am_arp[] = {57, 60, 64, 69, 64, 60, 57, 60};

    for (int i = 0; i < total_frames; i++) {
        int bar = i / (beat * 4);
        int eighth = (i % (beat * 4)) / (beat / 2);
        int pos_in_eighth = i % (beat / 2);

        if (pos_in_eighth == 0) {
            synth_note_off_all(&synth);
            int note;
            if ((bar % 2) == 0)
                note = cmaj_arp[eighth % 8];
            else
                note = am_arp[eighth % 8];
            synth_note_on(&synth, note, 0.3f);
            synth.voices[0].waveform = 0;  /* bright saw — synth arp */
        }

        float sample = synth_tick(&synth);
        if (sample > 0.95f) sample = 0.95f;
        if (sample < -0.95f) sample = -0.95f;

        int16_t s = (int16_t)(sample * 32767.0f);
        buf[i * 2] = s;
        buf[i * 2 + 1] = s;
    }
}

/*
 * Input C: Slow melody — single notes with long sustain, like a guitar solo.
 */
static void generate_slow_melody(int16_t *buf, int total_frames) {
    synth_t synth;
    synth_init(&synth);
    int beat = 44100 * 60 / 72;  /* 72 BPM, slow */
    int melody[] = {64, 67, 72, 71, 69, 67, 64, 60, 62, 64, 67, 69, 72, 76, 74, 72};

    for (int i = 0; i < total_frames; i++) {
        int note_period = beat;  /* one note per beat */
        int pos = i % note_period;
        int note_idx = (i / note_period) % 16;

        if (pos == 0) {
            synth_note_off_all(&synth);
            synth_note_on(&synth, melody[note_idx], 0.2f);
            synth.voices[0].waveform = 1;  /* pure sine — bell-like melody */
            synth.voices[0].env_rate = 0.001f;
        }
        if (pos == note_period * 3 / 4) {
            synth_note_off_all(&synth);
            for (int v = 0; v < MAX_VOICES; v++)
                if (synth.voices[v].active) synth.voices[v].env_rate = 0.0004f;
        }
        float sample = synth_tick(&synth);
        if (sample > 0.8f) sample = 0.8f;
        if (sample < -0.8f) sample = -0.8f;
        buf[i*2] = buf[i*2+1] = (int16_t)(sample * 32767.0f);
    }
}

/*
 * Input D: Single sustained chord — plays once and holds for the full duration.
 * Tests how the effect handles a single event ringing out.
 */
static void generate_single_chord(int16_t *buf, int total_frames) {
    synth_t synth;
    synth_init(&synth);
    int chord[] = {60, 64, 67, 72, 76};  /* Cmaj9 */

    for (int i = 0; i < total_frames; i++) {
        if (i == 0) {
            for (int n = 0; n < 5; n++) {
                synth_note_on(&synth, chord[n], 0.12f);  /* quiet — 5 voices sum up */
                synth.voices[n].waveform = 2;  /* triangle — soft pad */
                synth.voices[n].env_rate = 0.0005f;
            }
        }
        /* Release after 3 seconds */
        if (i == 44100 * 3) {
            synth_note_off_all(&synth);
            for (int v = 0; v < MAX_VOICES; v++)
                if (synth.voices[v].active) synth.voices[v].env_rate = 0.0002f;
        }
        float sample = synth_tick(&synth);
        if (sample > 0.7f) sample = 0.7f;
        if (sample < -0.7f) sample = -0.7f;
        buf[i*2] = buf[i*2+1] = (int16_t)(sample * 32767.0f);
    }
}

/*
 * Input E: Rhythmic stabs — short percussive hits with silence between.
 * Tests how the effect handles transients and the space between them.
 */
static void generate_stabs(int16_t *buf, int total_frames) {
    synth_t synth;
    synth_init(&synth);
    int beat = 44100 * 60 / 110;  /* 110 BPM */
    int chords[4][3] = {
        {48, 60, 67},  /* C5 power chord low */
        {53, 60, 65},  /* F */
        {55, 62, 67},  /* G */
        {52, 57, 64},  /* Em */
    };
    /* Rhythm: hit on beats 1, 2-and, 4 */
    int hits[] = {0, 3, 7};  /* in 8th notes */

    for (int i = 0; i < total_frames; i++) {
        int bar_len = beat * 4;
        int pos_in_bar = i % bar_len;
        int eighth = pos_in_bar / (beat / 2);
        int pos_in_eighth = pos_in_bar % (beat / 2);
        int bar = (i / bar_len) % 4;

        int is_hit = 0;
        for (int h = 0; h < 3; h++) if (eighth == hits[h]) is_hit = 1;

        if (is_hit && pos_in_eighth == 0) {
            synth_note_off_all(&synth);
            for (int n = 0; n < 3; n++) {
                synth_note_on(&synth, chords[bar][n], 0.35f);
                synth.voices[n].waveform = 0;  /* bright saw — punchy stabs */
                synth.voices[n].env_rate = 0.01f;
            }
        }
        /* Quick release after 1/8 of a beat */
        if (is_hit && pos_in_eighth == beat / 16) {
            synth_note_off_all(&synth);
            for (int v = 0; v < MAX_VOICES; v++)
                if (synth.voices[v].active) synth.voices[v].env_rate = 0.002f;
        }
        float sample = synth_tick(&synth);
        if (sample > 0.9f) sample = 0.9f;
        if (sample < -0.9f) sample = -0.9f;
        buf[i*2] = buf[i*2+1] = (int16_t)(sample * 32767.0f);
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

    /* Generate all input types */
    #define NUM_INPUTS 5
    int16_t *inputs[NUM_INPUTS];
    const char *input_names[NUM_INPUTS] = {"piano", "arp", "melody", "chord", "stabs"};
    for (int i = 0; i < NUM_INPUTS; i++)
        inputs[i] = (int16_t *)malloc(input_frames * 2 * sizeof(int16_t));

    printf("Generating inputs (%ds each)...\n", input_sec);
    generate_soft_piano(inputs[0], input_frames);
    generate_arp_input(inputs[1], input_frames);
    generate_slow_melody(inputs[2], input_frames);
    generate_single_chord(inputs[3], input_frames);
    generate_stabs(inputs[4], input_frames);

    /* Save dry inputs */
    for (int i = 0; i < NUM_INPUTS; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/00-dry-%s.wav", output_dir, input_names[i]);
        write_wav(path, inputs[i], input_frames * 2);
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
    /* Unified effect — 8 musical knobs sweeping through the parameter space */
    preset_t presets[] = {
        {"default",
         "{\"space\":0.65,\"shimmer\":0.35,\"smear\":0.40,\"warmth\":0.40,\"drift\":0.35,\"sustain\":0.75,\"scatter\":0.40,\"mix\":0.70}"},
        {"sparse-sparkle",
         "{\"space\":0.75,\"shimmer\":0.50,\"smear\":0.10,\"warmth\":0.35,\"drift\":0.25,\"sustain\":0.80,\"scatter\":0.50,\"mix\":0.75}"},
        {"dense-wash",
         "{\"space\":0.85,\"shimmer\":0.40,\"smear\":0.85,\"warmth\":0.45,\"drift\":0.40,\"sustain\":0.85,\"scatter\":0.50,\"mix\":0.80}"},
        {"crystal-cave",
         "{\"space\":0.85,\"shimmer\":0.65,\"smear\":0.30,\"warmth\":0.25,\"drift\":0.40,\"sustain\":0.80,\"scatter\":0.60,\"mix\":0.75}"},
        {"warm-amber",
         "{\"space\":0.70,\"shimmer\":0.25,\"smear\":0.50,\"warmth\":0.70,\"drift\":0.30,\"sustain\":0.80,\"scatter\":0.35,\"mix\":0.70}"},
        {"bright-air",
         "{\"space\":0.70,\"shimmer\":0.45,\"smear\":0.35,\"warmth\":0.10,\"drift\":0.35,\"sustain\":0.75,\"scatter\":0.55,\"mix\":0.70}"},
        {"infinite-drift",
         "{\"space\":0.90,\"shimmer\":0.50,\"smear\":0.60,\"warmth\":0.45,\"drift\":0.80,\"sustain\":0.88,\"scatter\":0.60,\"mix\":0.85}"},
        {"tight-shimmer",
         "{\"space\":0.35,\"shimmer\":0.60,\"smear\":0.20,\"warmth\":0.30,\"drift\":0.20,\"sustain\":0.70,\"scatter\":0.30,\"mix\":0.60}"},
    };

    int num_presets = sizeof(presets) / sizeof(presets[0]);

    /* Render each preset with all inputs */
    printf("\nRendering %d presets x %d inputs (%ds + %ds tail)...\n\n",
           num_presets, NUM_INPUTS, input_sec, tail_sec);

    for (int i = 0; i < num_presets; i++) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            char name[128];
            snprintf(name, sizeof(name), "%02d-%s-%s", i + 1, input_names[j], presets[i].name);
            preset_t p = { name, presets[i].state };
            render_preset(api, &p, inputs[j], input_frames, tail_frames, output_dir);
        }
    }

    printf("\nDone! %d WAV files saved to %s\n", num_presets * NUM_INPUTS + NUM_INPUTS, output_dir);

    for (int i = 0; i < NUM_INPUTS; i++) free(inputs[i]);
    return 0;
}
