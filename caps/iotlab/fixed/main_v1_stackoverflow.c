/* PREREG13: integer (fixed-point) predictors on IoT-LAB M3 (RIOT). Mirrors qemu.py bit for bit.
 * Part A: DWT cycles and result (Q(FI) integer) for 100 Intel windows. Part B: power phases as PREREG12. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cpu.h"
#include "periph_conf.h"
#include "ztimer.h"
#include "models_q.h"

#define LB 12
#define ACT 11
#define PHASE_MS 4000
#define IDLE_MS 2000
#define ROUNDS 3

static inline void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t cyc(void) { return DWT->CYCCNT; }
static inline int64_t rsh(int64_t v, int s) { return s >= 0 ? (v >> s) : v * ((int64_t)1 << (-s)); }
static inline int64_t clip16(int64_t v) { return v < -32768 ? -32768 : (v > 32767 ? 32767 : v); }
static inline int64_t lut(const int16_t *T, int64_t x) {
    if (x < -16384) x = -16384;
    if (x > 16383) x = 16383;
    return T[(x + 16384) >> 6];
}
static inline void norm(const int32_t *w, int64_t *z) {
    for (int i = 0; i < LB; i++) z[i] = clip16(rsh(((int64_t)w[i] - MXQ[i]) * ISXQ[i], FI + SI - ACT));
}
static inline int32_t out(int64_t acc, int scale) {
    int64_t o16 = rsh(acc, scale - 16);
    return (int32_t)(rsh(o16 * SYQ, 16) + MYQ);
}
static int32_t dot(const int32_t *w, const int64_t *c) {
    int64_t a = 0; for (int i = 0; i < LB; i++) a += (int64_t)w[i] * c[i]; return (int32_t)(a >> 16);
}
__attribute__((noinline)) int32_t q_empty(const int32_t *w) { (void)w; return 0; }
__attribute__((noinline)) int32_t q_persistence(const int32_t *w) { return w[LB - 1]; }
__attribute__((noinline)) int32_t q_expsm(const int32_t *w) { return dot(w, EQ); }
__attribute__((noinline)) int32_t q_lin2(const int32_t *w) {
    int64_t a = 0; for (int i = LB - 2; i < LB; i++) a += (int64_t)w[i] * L2Q[i]; return (int32_t)(a >> 16);
}
__attribute__((noinline)) int32_t q_lin12(const int32_t *w) { return dot(w, L12Q); }
__attribute__((noinline)) int32_t q_light(const int32_t *w) {
    int64_t z[LB], a2 = B2Q;
    norm(w, z);
    for (int j = 0; j < 16; j++) {
        int64_t a = B1Q[j];
        for (int i = 0; i < LB; i++) a += z[i] * W1Q[i * 16 + j];
        if (a > 0) a2 += (a >> S1) * W2Q[j];
    }
    return out(a2, ACT + S2);
}
__attribute__((noinline)) int32_t q_cnn(const int32_t *w) {
    static int64_t H[LB][32]; int64_t z[LB], zp[LB + 2], h[32], c[32], g[128];
    norm(w, z); zp[0] = 0; zp[LB + 1] = 0; for (int i = 0; i < LB; i++) zp[i + 1] = z[i];
    for (int t = 0; t < LB; t++) for (int f = 0; f < 32; f++) {
        int64_t a = CBQ[f];
        for (int k = 0; k < 3; k++) a += zp[t + k] * CKQ[k * 32 + f];
        H[t][f] = a > 0 ? (a >> SC) : 0;
    }
    memset(h, 0, sizeof h); memset(c, 0, sizeof c);
    for (int t = 0; t < LB; t++) {
        for (int qq = 0; qq < 128; qq++) {
            int64_t a = LBQ[qq];
            for (int i = 0; i < 32; i++) a += H[t][i] * LKQ[i * 128 + qq] + h[i] * LRKQ[i * 128 + qq];
            g[qq] = a >> SL;
        }
        for (int u = 0; u < 32; u++) {
            int64_t ig = lut(SIGT, g[u]), fg = lut(SIGT, g[32 + u]), cc = lut(TANHT, g[64 + u]), og = lut(SIGT, g[96 + u]);
            c[u] = (fg * c[u] + ig * cc) >> ACT;
            h[u] = (og * lut(TANHT, c[u])) >> ACT;
        }
    }
    int64_t o = DBQ; for (int i = 0; i < 32; i++) o += h[i] * DKQ[i];
    return out(o, ACT + SD);
}

typedef int32_t (*pf)(const int32_t *);
static const char *NAMES[6] = { "persistence", "expsm_q", "lin2_q", "lin12_q", "light_q", "cnn_q" };
static const pf FUNCS[6] = { q_persistence, q_expsm, q_lin2, q_lin12, q_light, q_cnn };
static volatile int32_t sink;

static void part_a(uint32_t base) {
    int32_t w[LB];
    for (int p = 0; p < 6; p++) for (int n = 0; n < NWIN; n++) {
        for (int i = 0; i < LB; i++) w[i] = WINQ[n * LB + i];
        uint32_t a = cyc(); int32_t r = FUNCS[p](w); uint32_t b = cyc(); sink = r;
        printf("A %s %lu %ld\n", NAMES[p], (unsigned long)(b - a - base), (long)r);
    }
}
static void idle_phase(int round) {
    printf("B IDLE begin %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    ztimer_sleep(ZTIMER_MSEC, IDLE_MS);
    printf("B IDLE end %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
}
static void run_phase(int p, int round) {
    int32_t w[LB]; uint32_t calls = 0, cycles = 0; int n = 0;
    printf("B RUN %s begin %d t=%lu\n", NAMES[p], round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    uint32_t t0 = ztimer_now(ZTIMER_MSEC);
    while (ztimer_now(ZTIMER_MSEC) - t0 < PHASE_MS) {
        for (int i = 0; i < LB; i++) w[i] = WINQ[n * LB + i];
        uint32_t a = cyc(); sink = FUNCS[p](w); cycles += cyc() - a;
        calls++; n = (n + 1) % NWIN;
    }
    uint32_t t1 = ztimer_now(ZTIMER_MSEC);
    printf("B RUN %s end %d t=%lu calls=%lu cycles=%lu wall_ms=%lu\n", NAMES[p], round,
           (unsigned long)t1, (unsigned long)calls, (unsigned long)cycles, (unsigned long)(t1 - t0));
}
int main(void) {
    dwt_init();
    ztimer_sleep(ZTIMER_MSEC, 3000);
    int32_t w[LB]; for (int i = 0; i < LB; i++) w[i] = WINQ[i];
    uint32_t a = cyc(); sink = q_empty(w); uint32_t base = cyc() - a;
    printf("HDR board=%s core_hz=%lu base=%lu nwin=%d phase_ms=%d idle_ms=%d rounds=%d fixed=1 fi=%d\n",
           RIOT_BOARD, (unsigned long)CLOCK_CORECLOCK, (unsigned long)base, NWIN, PHASE_MS, IDLE_MS, ROUNDS, FI);
    part_a(base);
    printf("A done\n");
    for (int r = 0; r < ROUNDS; r++) { idle_phase(r); for (int p = 0; p < 6; p++) { run_phase(p, r); idle_phase(r); } }
    printf("ALL DONE\n");
    while (1) ztimer_sleep(ZTIMER_MSEC, 60000);
    return 0;
}
