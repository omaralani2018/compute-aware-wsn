/* Float32 inference of the paper's predictors on Cortex-M0 (soft-float).
   Same arithmetic as fastfwd.forward / frozen_benchmark.wrap. */
#include <stdint.h>
#include <math.h>
#include "params.h"            /* KIND, weights, MX, ISX, MY, SY, NWIN, WIN[][12] */
#define MARK_ON()  (*(volatile uint32_t*)0x48000818 = 1u)   /* GPIOC_BSRR[0]: start counting */
#define MARK_OFF() (*(volatile uint32_t*)0x48000828 = 1u)   /* GPIOC_BRR[0]: stop, print delta */
static volatile float xin[12];
static float z[12];
#if KIND == 1          /* lightweight: Dense(16,relu) -> Dense(1) */
static float predict(void) {
  float h[16];
  for (int i = 0; i < 12; i++) z[i] = (xin[i] - MX[i]) * ISX[i];
  for (int j = 0; j < 16; j++) { float a = B1[j]; for (int i = 0; i < 12; i++) a += z[i] * K1[i][j]; h[j] = a > 0.0f ? a : 0.0f; }
  float y = B2; for (int j = 0; j < 16; j++) y += h[j] * K2[j];
  return y * SY + MY;
}
#elif KIND == 2        /* CNN-LSTM: Conv1D(32,3,same,relu) -> LSTM(32) -> Dense(1) */
static float H[12][32];
static float predict(void) {
  float zp[14], h[32], c[32], g[128];
  zp[0] = 0.0f; zp[13] = 0.0f;
  for (int i = 0; i < 12; i++) zp[i + 1] = (xin[i] - MX[i]) * ISX[i];
  for (int t = 0; t < 12; t++) for (int f = 0; f < 32; f++) {
    float a = CB[f]; for (int k = 0; k < 3; k++) a += zp[t + k] * CK[k][f]; H[t][f] = a > 0.0f ? a : 0.0f; }
  for (int u = 0; u < 32; u++) { h[u] = 0.0f; c[u] = 0.0f; }
  for (int t = 0; t < 12; t++) {
    for (int q = 0; q < 128; q++) { float a = LB[q];
      for (int j = 0; j < 32; j++) a += H[t][j] * LK[j][q];
      for (int j = 0; j < 32; j++) a += h[j] * LRK[j][q];
      g[q] = a; }
    for (int u = 0; u < 32; u++) {
      float ig = 1.0f / (1.0f + expf(-g[u]));
      float fg = 1.0f / (1.0f + expf(-g[32 + u]));
      float cc = tanhf(g[64 + u]);
      float og = 1.0f / (1.0f + expf(-g[96 + u]));
      c[u] = fg * c[u] + ig * cc;
      h[u] = og * tanhf(c[u]); }
  }
  float y = DB; for (int j = 0; j < 32; j++) y += h[j] * DK[j];
  return y * SY + MY;
}
#elif KIND == 3        /* exponential smoothing over the look-back window */
static float predict(void) { float y = 0.0f; for (int i = 0; i < 12; i++) y += xin[i] * EW[i]; return y; }
#elif KIND == 4        /* linear extrapolation over the last LK values */
static float predict(void) {
  const int k = LINK; float m = 0.0f, s = 0.0f;
  for (int i = 12 - k; i < 12; i++) m += xin[i];
  m *= INVK;
  for (int i = 0; i < k; i++) s += (xin[12 - k + i] - m) * TC[i];
  float sl = s * INVDEN;
  return sl * (float)k + (m - sl * TMEAN);
}
#elif KIND == 0        /* persistence */
static float predict(void) { return xin[11]; }
#endif
static void puthex(uint32_t v){ const char *d="0123456789abcdef"; for(int s=28;s>=0;s-=4) *(volatile uint32_t*)0xE0000000 = d[(v>>s)&15]; *(volatile uint32_t*)0xE0000000='\n'; }
static volatile float yout;
int main(void) {
  for (int w = 0; w < NWIN; w++) {
    for (int i = 0; i < 12; i++) xin[i] = WIN[w][i];
    MARK_ON();
    yout = predict();                          /* volatile store: the result is complete */
    __asm volatile ("" ::: "memory");          /* before the stop marker */
    MARK_OFF();
    float y = yout;
    union { float f; uint32_t u; } cv; cv.f = y; puthex(cv.u);
  }
  return 0;
}
