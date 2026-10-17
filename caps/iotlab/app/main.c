/* PREREG12: measured execution time and energy of one forecast on IoT-LAB nodes (RIOT).
 * Same predictors, weights and 100 Intel test windows as avrsim (PREREG11). IEEE float32.
 * Part A: per-call DWT cycle count and result bits for every predictor and window (correctness, time).
 * Part B: power phases for the consumption monitor: IDLE (ztimer sleep) and RUN <predictor> (back-to-back
 *         forecasts for PHASE_MS), repeated ROUNDS times; nothing is printed inside a phase. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "cpu.h"
#include "periph_conf.h"
#include "ztimer.h"
#include "models.h"

#define LB 12
#define PHASE_MS 4000
#define IDLE_MS 2000
#define ROUNDS 3

static inline void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t cyc(void) { return DWT->CYCCNT; }

__attribute__((noinline)) float p_empty(const float *w) { (void)w; return 0.0f; }
__attribute__((noinline)) float p_persistence(const float *w) { return w[LB - 1]; }
__attribute__((noinline)) float p_expsm(const float *w) { float s = 0; for (int i = 0; i < LB; i++) s += EW[i] * w[i]; return s; }
static float lin(const float *w, int k) {
    const float *x = w + LB - k; float m = 0, tm = (k - 1) * 0.5f, den = 0, num = 0;
    for (int i = 0; i < k; i++) m += x[i];
    m /= k;
    for (int i = 0; i < k; i++) { float tc = i - tm; num += (x[i] - m) * tc; den += tc * tc; }
    float sl = num / den; return sl * k + (m - sl * tm);
}
__attribute__((noinline)) float p_lin2(const float *w) { return lin(w, 2); }
__attribute__((noinline)) float p_lin12(const float *w) { return lin(w, 12); }
__attribute__((noinline)) float p_light(const float *w) {
    float z[LB], o = L_B2[0];
    for (int i = 0; i < LB; i++) z[i] = (w[i] - MX[i]) * ISX[i];
    for (int j = 0; j < 16; j++) {
        float a = L_B1[j];
        for (int i = 0; i < LB; i++) a += z[i] * L_K1[i * 16 + j];
        if (a > 0) o += a * L_K2[j];
    }
    return o * SY + MY;
}
static float sg(float x) { return 1.0f / (1.0f + expf(-x)); }
__attribute__((noinline)) float p_cnn(const float *w) {
    static float H[LB][32]; float z[LB + 2], h[32], c[32], g[128];
    z[0] = 0; z[LB + 1] = 0;
    for (int i = 0; i < LB; i++) z[i + 1] = (w[i] - MX[i]) * ISX[i];
    for (int t = 0; t < LB; t++) for (int f = 0; f < 32; f++) {
        float a = C_CB[f];
        for (int k = 0; k < 3; k++) a += z[t + k] * C_CK[k * 32 + f];
        H[t][f] = a > 0 ? a : 0;
    }
    memset(h, 0, sizeof h); memset(c, 0, sizeof c);
    for (int t = 0; t < LB; t++) {
        for (int q = 0; q < 128; q++) {
            float a = C_LB[q];
            for (int i = 0; i < 32; i++) a += H[t][i] * C_LK[i * 128 + q] + h[i] * C_LRK[i * 128 + q];
            g[q] = a;
        }
        for (int u = 0; u < 32; u++) {
            float ig = sg(g[u]), fg = sg(g[32 + u]), cc = tanhf(g[64 + u]), og = sg(g[96 + u]);
            c[u] = fg * c[u] + ig * cc; h[u] = og * tanhf(c[u]);
        }
    }
    float o = C_DB[0];
    for (int i = 0; i < 32; i++) o += h[i] * C_DK[i];
    return o * SY + MY;
}

typedef float (*pf)(const float *);
static const char *NAMES[6] = { "persistence", "expsm", "lin2", "lin12", "lightweight", "cnn_lstm" };
static const pf FUNCS[6] = { p_persistence, p_expsm, p_lin2, p_lin12, p_light, p_cnn };
static volatile float sink;

static void part_a(uint32_t base) {
    float w[LB];
    for (int p = 0; p < 6; p++) for (int n = 0; n < NWIN; n++) {
        for (int i = 0; i < LB; i++) w[i] = WIN[n * LB + i];
        uint32_t a = cyc(); float r = FUNCS[p](w); uint32_t b = cyc(); sink = r;
        uint32_t bits; memcpy(&bits, &r, 4);
        printf("A %s %lu %08lx\n", NAMES[p], (unsigned long)(b - a - base), (unsigned long)bits);
    }
}

static void idle_phase(int round) {
    printf("B IDLE begin %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    ztimer_sleep(ZTIMER_MSEC, IDLE_MS);
    printf("B IDLE end %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
}

static void run_phase(int p, int round) {
    float w[LB]; uint32_t calls = 0, cycles = 0; int n = 0;
    printf("B RUN %s begin %d t=%lu\n", NAMES[p], round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    uint32_t t0 = ztimer_now(ZTIMER_MSEC);
    while (ztimer_now(ZTIMER_MSEC) - t0 < PHASE_MS) {
        for (int i = 0; i < LB; i++) w[i] = WIN[n * LB + i];
        uint32_t a = cyc(); sink = FUNCS[p](w); cycles += cyc() - a;
        calls++; n = (n + 1) % NWIN;
    }
    uint32_t t1 = ztimer_now(ZTIMER_MSEC);
    printf("B RUN %s end %d t=%lu calls=%lu cycles=%lu wall_ms=%lu\n", NAMES[p], round,
           (unsigned long)t1, (unsigned long)calls, (unsigned long)cycles, (unsigned long)(t1 - t0));
}

int main(void) {
    dwt_init();
    ztimer_sleep(ZTIMER_MSEC, 3000);   /* let the serial aggregator attach */
    float w[LB];
    for (int i = 0; i < LB; i++) w[i] = WIN[i];
    uint32_t a = cyc(); sink = p_empty(w); uint32_t base = cyc() - a;
    printf("HDR board=%s core_hz=%lu base=%lu nwin=%d phase_ms=%d idle_ms=%d rounds=%d\n",
           RIOT_BOARD, (unsigned long)CLOCK_CORECLOCK, (unsigned long)base, NWIN, PHASE_MS, IDLE_MS, ROUNDS);
    part_a(base);
    printf("A done\n");
    for (int r = 0; r < ROUNDS; r++) {
        idle_phase(r);
        for (int p = 0; p < 6; p++) { run_phase(p, r); idle_phase(r); }
    }
    printf("ALL DONE\n");
    while (1) ztimer_sleep(ZTIMER_MSEC, 60000);
    return 0;
}
