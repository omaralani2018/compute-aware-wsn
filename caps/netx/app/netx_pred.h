/* NETX predictors, readings and the transmit rule, shared by the firmware (main.c) and the ARM host check (check/harness.c). */
#include <string.h>
#include <stdint.h>
#include <math.h>
#define LB 12
#ifndef NETX_INT
#include "netx_table.h"      /* includes netx_models.h: NETX_R, NETX_NS */
typedef float val_t;
static val_t RB[LB + NETX_R];
static const char *const BUILD = "float";
/* ---- float predictors (same arithmetic as PREREG12, per-stream tables) */
__attribute__((noinline)) static float p_expsm(const netx_stream_t *S, const float *w) { float s = 0; for (int i = 0; i < LB; i++) s += S->EW[i] * w[i]; return s; }
__attribute__((noinline)) static float p_lin(const netx_stream_t *S, const float *w) {
    int k = S->K; const float *x = w + LB - k; float m = 0, tm = (k - 1) * 0.5f, den = 0, num = 0;
    for (int i = 0; i < k; i++) m += x[i];
    m /= k;
    for (int i = 0; i < k; i++) { float tc = i - tm; num += (x[i] - m) * tc; den += tc * tc; }
    float sl = num / den; return sl * k + (m - sl * tm);
}
__attribute__((noinline)) static float p_light(const netx_stream_t *S, const float *w) {
    float z[LB], o = S->L_B2[0];
    for (int i = 0; i < LB; i++) z[i] = (w[i] - S->MX[i]) * S->ISX[i];
    for (int j = 0; j < 16; j++) {
        float a = S->L_B1[j];
        for (int i = 0; i < LB; i++) a += z[i] * S->L_K1[i * 16 + j];
        if (a > 0) o += a * S->L_K2[j];
    }
    return o * S->SY + S->MY;
}
static float sg(float x) { return 1.0f / (1.0f + expf(-x)); }
__attribute__((noinline)) static float p_cnn(const netx_stream_t *S, const float *w) {
    static float H[LB][32], z[LB + 2], h[32], c[32], g[128];
    z[0] = 0; z[LB + 1] = 0;
    for (int i = 0; i < LB; i++) z[i + 1] = (w[i] - S->MX[i]) * S->ISX[i];
    for (int t = 0; t < LB; t++) for (int f = 0; f < 32; f++) {
        float a = S->C_CB[f];
        for (int k = 0; k < 3; k++) a += z[t + k] * S->C_CK[k * 32 + f];
        H[t][f] = a > 0 ? a : 0;
    }
    memset(h, 0, sizeof h); memset(c, 0, sizeof c);
    for (int t = 0; t < LB; t++) {
        for (int q = 0; q < 128; q++) {
            float a = S->C_LB[q];
            for (int i = 0; i < 32; i++) a += H[t][i] * S->C_LK[i * 128 + q] + h[i] * S->C_LRK[i * 128 + q];
            g[q] = a;
        }
        for (int u = 0; u < 32; u++) {
            float ig = sg(g[u]), fg = sg(g[32 + u]), cc = tanhf(g[64 + u]), og = sg(g[96 + u]);
            c[u] = fg * c[u] + ig * cc; h[u] = og * tanhf(c[u]);
        }
    }
    float o = S->C_DB[0];
    for (int i = 0; i < 32; i++) o += h[i] * S->C_DK[i];
    return o * S->SY + S->MY;
}
static val_t forecast(int s, int a, const val_t *w) {
    const netx_stream_t *S = &NS[s];
    switch (a) { case 0: return w[LB - 1]; case 1: return p_expsm(S, w); case 2: return p_lin(S, w); case 3: return p_light(S, w); default: return p_cnn(S, w); }
}
static inline val_t reading(int s, int t) { return NS[s].X[t]; }
static inline int misses(int s, val_t x, val_t p) { return !(fabsf(x - p) / NS[s].UNIT <= 0.1f); }
static inline int ref_arm(int s) { return NS[s].ref; }
#else
#include "netx_models_q.h"
typedef int32_t val_t;
static val_t RB[LB + NETX_R];
static const char *const BUILD = "int";
#define ACT 11
static const char *const NETX_NAME[NETX_NS] = { "taxi", "exchange_cpc", "machine_temp", "cpu" };
static const int REFQ[NETX_NS] = { 2, 1, 1, 1 };          /* checked against netx_plan.json by host_check */
static inline int64_t rsh(int64_t v, int s) { return s >= 0 ? (v >> s) : v * ((int64_t)1 << (-s)); }
static inline int64_t clip16(int64_t v) { return v < -32768 ? -32768 : (v > 32767 ? 32767 : v); }
static inline int64_t lut(const int16_t *T, int64_t x) { if (x < -16384) x = -16384; if (x > 16383) x = 16383; return T[(x + 16384) >> 6]; }
static inline void norm(const netx_q_t *S, const int32_t *w, int64_t *z) {
    for (int i = 0; i < LB; i++) z[i] = clip16(rsh(((int64_t)w[i] - S->MXQ[i]) * S->ISXQ[i], S->FI + S->SI - ACT));
}
static inline int32_t out(const netx_q_t *S, int64_t acc, int scale) { int64_t o16 = rsh(acc, scale - 16); return (int32_t)(rsh(o16 * S->SYQ, 16) + S->MYQ); }
__attribute__((noinline)) static int32_t q_dot(const int64_t *c, const int32_t *w) { int64_t a = 0; for (int i = 0; i < LB; i++) a += (int64_t)w[i] * c[i]; return (int32_t)rsh(a, 16); }
__attribute__((noinline)) static int32_t q_lin(const netx_q_t *S, const int32_t *w) {   /* the last K terms only (the others are 0), as lin2_q */
    int64_t a = 0; for (int i = LB - S->K; i < LB; i++) a += (int64_t)w[i] * S->LQ[i]; return (int32_t)rsh(a, 16);
}
__attribute__((noinline)) static int32_t q_light(const netx_q_t *S, const int32_t *w) {
    int64_t z[LB], a2 = S->B2Q;
    norm(S, w, z);
    for (int j = 0; j < 16; j++) {
        int64_t a = S->B1Q[j];
        for (int i = 0; i < LB; i++) a += z[i] * S->W1Q[i * 16 + j];
        if (a > 0) a2 += (a >> S->S1) * S->W2Q[j];
    }
    return out(S, a2, ACT + S->S2);
}
__attribute__((noinline)) static int32_t q_cnn(const netx_q_t *S, const int32_t *w) {
    static int64_t H[LB][32], z[LB], zp[LB + 2], h[32], c[32], g[128];
    norm(S, w, z); zp[0] = 0; zp[LB + 1] = 0; for (int i = 0; i < LB; i++) zp[i + 1] = z[i];
    for (int t = 0; t < LB; t++) for (int f = 0; f < 32; f++) {
        int64_t a = S->CBQ[f];
        for (int k = 0; k < 3; k++) a += zp[t + k] * S->CKQ[k * 32 + f];
        H[t][f] = a > 0 ? (a >> S->SC) : 0;
    }
    memset(h, 0, sizeof h); memset(c, 0, sizeof c);
    for (int t = 0; t < LB; t++) {
        for (int qq = 0; qq < 128; qq++) {
            int64_t a = S->LBQ[qq];
            for (int i = 0; i < 32; i++) a += H[t][i] * S->LKQ[i * 128 + qq] + h[i] * S->LRKQ[i * 128 + qq];
            g[qq] = rsh(a, S->SL);
        }
        for (int u = 0; u < 32; u++) {
            int64_t ig = lut(SIGT, g[u]), fg = lut(SIGT, g[32 + u]), cc = lut(TANHT, g[64 + u]), og = lut(SIGT, g[96 + u]);
            c[u] = (fg * c[u] + ig * cc) >> ACT;
            h[u] = (og * lut(TANHT, c[u])) >> ACT;
        }
    }
    int64_t o = S->DBQ; for (int i = 0; i < 32; i++) o += h[i] * S->DKQ[i];
    return out(S, o, ACT + S->SD);
}
static val_t forecast(int s, int a, const val_t *w) {
    const netx_q_t *S = &NQ[s];
    switch (a) { case 0: return w[LB - 1]; case 1: return q_dot(S->EQ, w); case 2: return q_lin(S, w); case 3: return q_light(S, w); default: return q_cnn(S, w); }
}
static inline val_t reading(int s, int t) { return NQ[s].XQ[t]; }
static inline int misses(int s, val_t x, val_t p) { int64_t e = (int64_t)x - p; if (e < 0) e = -e; return e > NQ[s].THR; }
static inline int ref_arm(int s) { return REFQ[s]; }
#endif

