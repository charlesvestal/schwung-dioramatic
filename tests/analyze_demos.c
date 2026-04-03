/*
 * Analyze rendered demo WAVs for musical quality issues
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int16_t *read_wav(const char *path, int *num_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    /* Skip header (44 bytes) */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 44, SEEK_SET);
    int data_bytes = (int)(fsize - 44);
    *num_samples = data_bytes / 2;
    int16_t *buf = (int16_t *)malloc(data_bytes);
    fread(buf, 1, data_bytes, f);
    fclose(f);
    return buf;
}

typedef struct {
    float rms;
    float peak;
    float stereo_width;    /* RMS of L-R */
    float crest_factor;    /* peak/rms — lower = more compressed/distorted */
    float dc_offset;
    float silence_pct;     /* % of frames below -60dB */
    float clip_pct;        /* % of samples at ±32767 */
    float zero_crossing_rate;
    /* Per-second RMS for temporal analysis */
    float sec_rms[16];
    int num_seconds;
    float temporal_range;  /* max_sec_rms - min_sec_rms */
} wav_analysis_t;

static wav_analysis_t analyze_wav(int16_t *samples, int num_samples) {
    wav_analysis_t a = {0};
    int frames = num_samples / 2;  /* stereo */

    double sum2 = 0, sum_diff2 = 0, sum_dc = 0;
    float peak = 0;
    int silence_frames = 0, clip_count = 0, zc = 0;
    int16_t prev = 0;

    /* Per-second accumulators */
    int samples_per_sec = 44100;
    double sec_sum2[16] = {0};
    int sec_count[16] = {0};

    for (int i = 0; i < frames; i++) {
        float l = (float)samples[i*2] / 32768.0f;
        float r = (float)samples[i*2+1] / 32768.0f;
        float mono = (l + r) * 0.5f;

        sum2 += mono * mono;
        sum_diff2 += (l - r) * (l - r);
        sum_dc += mono;

        float al = fabsf(l), ar = fabsf(r);
        float amax = al > ar ? al : ar;
        if (amax > peak) peak = amax;

        if (al < 0.001f && ar < 0.001f) silence_frames++;
        if (samples[i*2] == 32767 || samples[i*2] == -32768) clip_count++;
        if (samples[i*2+1] == 32767 || samples[i*2+1] == -32768) clip_count++;

        if (i > 0 && ((samples[i*2] > 0) != (prev > 0))) zc++;
        prev = samples[i*2];

        int sec = i / samples_per_sec;
        if (sec < 16) {
            sec_sum2[sec] += mono * mono;
            sec_count[sec]++;
        }
    }

    a.rms = sqrtf((float)(sum2 / frames));
    a.peak = peak;
    a.stereo_width = sqrtf((float)(sum_diff2 / frames));
    a.crest_factor = (a.rms > 0.0001f) ? (peak / a.rms) : 0;
    a.dc_offset = (float)(sum_dc / frames);
    a.silence_pct = 100.0f * (float)silence_frames / (float)frames;
    a.clip_pct = 100.0f * (float)clip_count / (float)(num_samples);
    a.zero_crossing_rate = (float)zc / (float)frames;

    float min_sec = 999, max_sec = 0;
    for (int s = 0; s < 16; s++) {
        if (sec_count[s] > 0) {
            a.sec_rms[s] = sqrtf((float)(sec_sum2[s] / sec_count[s]));
            if (a.sec_rms[s] < min_sec) min_sec = a.sec_rms[s];
            if (a.sec_rms[s] > max_sec) max_sec = a.sec_rms[s];
            a.num_seconds = s + 1;
        }
    }
    a.temporal_range = max_sec - min_sec;

    return a;
}

int main(void) {
    const char *dir = "/Users/charlesvestal/Desktop/dioramatic-demos";
    const char *files[] = {
        "00-dry-input",
        "01-mosaic-A-shimmer",
        "02-mosaic-D-fullrange",
        "03-mosaic-B-dark-octavedown",
        "04-mosaic-square-gate",
        "05-haze-triangle-swell",
        "06-haze-A-diffuse",
        "07-haze-C-shimmer",
        "08-tunnel-B-overtones",
        "09-glide-C-updown",
        "10-seq-B-halftime",
        "11-strum-C-cascade",
        "12-blocks-C-pitchglitch",
        "13-interrupt-A-subtle",
        "14-arp-A-ascending",
        "15-arp-C-updown-reverbed",
        "16-pattern-A-clean",
        "17-pattern-B-dotted",
        "18-warp-C-pitchfilter",
        "19-ambient-wash",
        "20-mosaic-A-reverse",
    };
    int nfiles = sizeof(files)/sizeof(files[0]);

    printf("\n=== Dioramatic Demo Audio Analysis ===\n\n");
    printf("%-30s %6s %6s %6s %6s %6s %6s %6s\n",
           "File", "RMS", "Peak", "Stereo", "Crest", "Sil%", "Clip%", "TmpRng");
    printf("%-30s %6s %6s %6s %6s %6s %6s %6s\n",
           "----", "---", "----", "------", "-----", "----", "-----", "------");

    wav_analysis_t results[21];

    for (int i = 0; i < nfiles; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.wav", dir, files[i]);
        int num_samples;
        int16_t *samples = read_wav(path, &num_samples);
        if (!samples) continue;

        results[i] = analyze_wav(samples, num_samples);
        wav_analysis_t *a = &results[i];

        printf("%-30s %6.4f %6.3f %6.4f %6.1f %5.1f%% %5.1f%% %6.4f\n",
               files[i], a->rms, a->peak, a->stereo_width,
               a->crest_factor, a->silence_pct, a->clip_pct, a->temporal_range);

        free(samples);
    }

    /* Per-second RMS timeline for each effect */
    printf("\n=== Per-Second RMS Timeline ===\n\n");
    for (int i = 0; i < nfiles; i++) {
        wav_analysis_t *a = &results[i];
        printf("%-30s ", files[i]);
        for (int s = 0; s < a->num_seconds; s++) {
            /* Simple bar graph */
            int bars = (int)(a->sec_rms[s] * 50.0f);
            if (bars > 20) bars = 20;
            printf("%4.3f", a->sec_rms[s]);
            if (s < a->num_seconds - 1) printf(" ");
        }
        printf("\n");
    }

    /* Issue detection */
    printf("\n=== Issues Detected ===\n\n");
    int issues = 0;

    for (int i = 1; i < nfiles; i++) {  /* skip dry */
        wav_analysis_t *a = &results[i];
        wav_analysis_t *dry = &results[0];

        /* Too quiet relative to dry */
        if (a->rms < dry->rms * 0.1f) {
            printf("  %s: VERY QUIET (rms=%.4f vs dry=%.4f) — effect barely audible\n",
                   files[i], a->rms, dry->rms);
            issues++;
        }

        /* Too loud / clipping */
        if (a->clip_pct > 0.1f) {
            printf("  %s: CLIPPING (%.1f%% clipped samples)\n", files[i], a->clip_pct);
            issues++;
        }

        /* Excessive silence (>80% silent for non-Interrupt/Blocks) */
        if (a->silence_pct > 80.0f && i != 13 && i != 18) {
            printf("  %s: MOSTLY SILENT (%.0f%% silence)\n", files[i], a->silence_pct);
            issues++;
        }

        /* No stereo width on grain algorithms */
        if (a->stereo_width < 0.001f && i < 15) {
            printf("  %s: MONO (no stereo width)\n", files[i]);
            issues++;
        }

        /* DC offset */
        if (fabsf(a->dc_offset) > 0.01f) {
            printf("  %s: DC OFFSET (%.4f)\n", files[i], a->dc_offset);
            issues++;
        }

        /* Very low crest factor = heavily compressed/distorted */
        if (a->crest_factor < 1.5f && a->rms > 0.01f) {
            printf("  %s: LOW CREST FACTOR (%.1f) — may sound distorted/crushed\n",
                   files[i], a->crest_factor);
            issues++;
        }

        /* No temporal variation (static/boring) */
        if (a->temporal_range < 0.005f && a->rms > 0.01f) {
            printf("  %s: STATIC (temporal range %.4f) — no movement/evolution\n",
                   files[i], a->temporal_range);
            issues++;
        }
    }

    if (issues == 0) printf("  None — all demos sound healthy!\n");
    printf("\n%d issues found\n\n", issues);

    return 0;
}
