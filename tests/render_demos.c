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
                /* Softer attack for piano-like quality */
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
        }

        float sample = synth_tick(&synth);
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

    /* Generate both input types */
    int16_t *piano_input = (int16_t *)malloc(input_frames * 2 * sizeof(int16_t));
    int16_t *arp_input = (int16_t *)malloc(input_frames * 2 * sizeof(int16_t));

    printf("Generating soft piano input (%ds)...\n", input_sec);
    generate_soft_piano(piano_input, input_frames);
    printf("Generating arp input (%ds)...\n", input_sec);
    generate_arp_input(arp_input, input_frames);

    /* Save dry inputs */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/00-dry-piano.wav", output_dir);
        write_wav(path, piano_input, input_frames * 2);
        printf("  Wrote: %s\n", path);
        snprintf(path, sizeof(path), "%s/00-dry-arp.wav", output_dir);
        write_wav(path, arp_input, input_frames * 2);
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
    /* Your 6 favorites from the shimmer reverb version, rendered with
       both piano stabs and arp input. These are the presets that sounded
       good before — Haze diffuse, Haze shimmer, Tunnel overtones,
       Blocks pitch-glitch, Ambient wash, Mosaic reverse. */
    preset_t presets[] = {
        /* #6 Haze A — granular diffuse wash */
        {"haze-diffuse",
         "{\"algorithm\":3,\"variation\":0,\"activity\":0.75,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.75,\"mix\":0.6,\"space\":0.45,"
         "\"time_div\":2,\"pitch_mod_depth\":0.12,\"pitch_mod_rate\":0.35,"
         "\"filter_res\":0.2,\"reverb_mode\":2,\"reverse\":0,\"hold\":0}"},

        /* #7 Haze C — octave shimmer cloud */
        {"haze-shimmer",
         "{\"algorithm\":3,\"variation\":2,\"activity\":0.8,\"repeats\":0.65,"
         "\"shape\":1.0,\"filter\":0.9,\"mix\":0.65,\"space\":0.55,"
         "\"time_div\":2,\"pitch_mod_depth\":0.08,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* #8 Tunnel B — drone with overtone harmonics */
        {"tunnel-overtones",
         "{\"algorithm\":4,\"variation\":1,\"activity\":0.65,\"repeats\":0.75,"
         "\"shape\":1.0,\"filter\":0.65,\"mix\":0.65,\"space\":0.5,"
         "\"time_div\":2,\"pitch_mod_depth\":0.1,\"pitch_mod_rate\":0.12,"
         "\"filter_res\":0.25,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* #12 Blocks C — pitch-shifted glitch sparkles */
        {"blocks-pitchglitch",
         "{\"algorithm\":6,\"variation\":2,\"activity\":0.75,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":1.0,\"mix\":0.55,\"space\":0.3,"
         "\"time_div\":0,\"pitch_mod_depth\":0.0,\"pitch_mod_rate\":0.3,"
         "\"filter_res\":0.0,\"reverb_mode\":0,\"reverse\":0,\"hold\":0}"},

        /* #19 Ambient wash — full Mosaic D through massive reverb */
        {"ambient-wash",
         "{\"algorithm\":0,\"variation\":3,\"activity\":0.85,\"repeats\":0.85,"
         "\"shape\":1.0,\"filter\":0.55,\"mix\":0.8,\"space\":0.85,"
         "\"time_div\":4,\"pitch_mod_depth\":0.1,\"pitch_mod_rate\":0.08,"
         "\"filter_res\":0.2,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* #20 Mosaic A reverse — time-smeared shimmer */
        {"mosaic-reverse",
         "{\"algorithm\":0,\"variation\":0,\"activity\":0.6,\"repeats\":0.7,"
         "\"shape\":1.0,\"filter\":0.85,\"mix\":0.65,\"space\":0.45,"
         "\"time_div\":2,\"pitch_mod_depth\":0.12,\"pitch_mod_rate\":0.25,"
         "\"filter_res\":0.1,\"reverb_mode\":2,\"reverse\":1,\"hold\":0}"},

        /* NEW: Sparse haze — very low activity, mostly reverb tail with
           occasional delicate grain events. The "sparkle" version. */
        {"haze-sparse-sparkle",
         "{\"algorithm\":3,\"variation\":2,\"activity\":0.15,\"repeats\":0.5,"
         "\"shape\":1.0,\"filter\":0.8,\"mix\":0.8,\"space\":0.85,"
         "\"time_div\":2,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.05,"
         "\"filter_res\":0.05,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* NEW: Sparse tunnel — very low activity drone, mostly shimmer reverb */
        {"tunnel-sparse",
         "{\"algorithm\":4,\"variation\":1,\"activity\":0.2,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.7,\"mix\":0.75,\"space\":0.85,"
         "\"time_div\":4,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.04,"
         "\"filter_res\":0.15,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},

        /* NEW: Sparse mosaic reverse — very few grains, heavy reverb */
        {"mosaic-reverse-sparse",
         "{\"algorithm\":0,\"variation\":0,\"activity\":0.15,\"repeats\":0.55,"
         "\"shape\":1.0,\"filter\":0.75,\"mix\":0.8,\"space\":0.85,"
         "\"time_div\":2,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.05,"
         "\"filter_res\":0.1,\"reverb_mode\":3,\"reverse\":1,\"hold\":0}"},

        /* NEW: Ambient wash sparse — very delicate, open, floating */
        {"ambient-sparse",
         "{\"algorithm\":0,\"variation\":3,\"activity\":0.2,\"repeats\":0.6,"
         "\"shape\":1.0,\"filter\":0.7,\"mix\":0.85,\"space\":0.9,"
         "\"time_div\":4,\"pitch_mod_depth\":0.06,\"pitch_mod_rate\":0.04,"
         "\"filter_res\":0.1,\"reverb_mode\":3,\"reverse\":0,\"hold\":0}"},
    };

    int num_presets = sizeof(presets) / sizeof(presets[0]);

    /* Render each preset with both inputs */
    printf("\nRendering %d presets x 2 inputs (%ds + %ds tail)...\n\n", num_presets, input_sec, tail_sec);

    for (int i = 0; i < num_presets; i++) {
        /* Piano version */
        char piano_name[128];
        snprintf(piano_name, sizeof(piano_name), "%02d-piano-%s", i + 1, presets[i].name);
        preset_t piano_preset = { piano_name, presets[i].state };
        render_preset(api, &piano_preset, piano_input, input_frames, tail_frames, output_dir);

        /* Arp version */
        char arp_name[128];
        snprintf(arp_name, sizeof(arp_name), "%02d-arp-%s", i + 1, presets[i].name);
        preset_t arp_preset = { arp_name, presets[i].state };
        render_preset(api, &arp_preset, arp_input, input_frames, tail_frames, output_dir);
    }

    printf("\nDone! %d WAV files saved to %s\n", num_presets * 2 + 2, output_dir);

    free(piano_input);
    free(arp_input);
    return 0;
}
