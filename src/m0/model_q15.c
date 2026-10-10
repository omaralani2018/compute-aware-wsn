/* Fixed-point inference (int16 weights Q12, int16 activations Q12, int32 accumulators,
   sigmoid/tanh by 512-segment look-up tables over [-8, 8) with linear interpolation,
   a table approach also used by CMSIS-NN), the usual way to run a
   network on a Cortex-M0 without a floating-point unit. Bit-exact with qmodel.py. */
#include <stdint.h>
#include "qparams.h"
#define MARK_ON()  (*(volatile uint32_t*)0x48000818 = 1u)
#define MARK_OFF() (*(volatile uint32_t*)0x48000828 = 1u)
static volatile int32_t xin[12];                 /* readings as integers: x * 2^PX (per-stream scale) */
static inline int16_t sat16(int32_t v){ return v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v); }
static inline int32_t lut(const int16_t *T, int32_t x){ /* x in Q12; 513-point table */
  if (x < -32768) x = -32768; if (x > 32767) x = 32767;
  int32_t u = x + 32768, i = u >> 7, fr = u & 127;
  return T[i] + (((T[i + 1] - T[i]) * fr) >> 7); }
#if KIND == 1
static int32_t predict(void) {
  int16_t z[12], h[16];
  for (int i = 0; i < 12; i++) z[i] = sat16(((xin[i] - MXQ[i]) * ISXQ[i]) >> SH);
  for (int j = 0; j < 16; j++) { int32_t a = B1Q[j]; for (int i = 0; i < 12; i++) a += z[i] * K1Q[i][j];
    a >>= 12; h[j] = a > 0 ? sat16(a) : 0; }
  int32_t y = B2Q; for (int j = 0; j < 16; j++) y += h[j] * K2Q[j];
  y >>= 12;
  return MYQ + ((y * SYQ) >> SH2);
}
#elif KIND == 2
static int16_t H[12][32];
static int32_t predict(void) {
  int16_t zp[14], h[32], c[32]; int32_t g[128];
  zp[0] = 0; zp[13] = 0;
  for (int i = 0; i < 12; i++) zp[i + 1] = sat16(((xin[i] - MXQ[i]) * ISXQ[i]) >> SH);
  for (int t = 0; t < 12; t++) for (int f = 0; f < 32; f++) {
    int32_t a = CBQ[f]; for (int k = 0; k < 3; k++) a += zp[t + k] * CKQ[k][f];
    a >>= 12; H[t][f] = a > 0 ? sat16(a) : 0; }
  for (int u = 0; u < 32; u++) { h[u] = 0; c[u] = 0; }
  for (int t = 0; t < 12; t++) {
    for (int q = 0; q < 128; q++) { int32_t a = LBQ[q];
      for (int j = 0; j < 32; j++) a += H[t][j] * LKQ[j][q];
      for (int j = 0; j < 32; j++) a += h[j] * LRKQ[j][q];
      g[q] = a >> 12; }
    for (int u = 0; u < 32; u++) {
      int32_t ig = lut(SIG, g[u]), fg = lut(SIG, g[32 + u]), cc = lut(TNH, g[64 + u]), og = lut(SIG, g[96 + u]);
      int32_t cn = (fg * c[u] + ig * cc) >> 12; c[u] = sat16(cn);
      h[u] = sat16((og * lut(TNH, c[u])) >> 12); }
  }
  int32_t y = DBQ; for (int j = 0; j < 32; j++) y += h[j] * DKQ[j];
  y >>= 12;
  return MYQ + ((y * SYQ) >> SH2);
}
#elif KIND == 3
static int32_t predict(void){ int32_t y = 0; for (int i = 0; i < 12; i++) y += xin[i] * EWQ[i]; return y >> 12; }
#elif KIND == 4
static int32_t predict(void){ /* K = 2: x[11] + (x[11] - x[10]) */ return xin[11] + (xin[11] - xin[10]); }
#elif KIND == 0
static int32_t predict(void){ return xin[11]; }
#endif
static void puthex(uint32_t v){ const char *d="0123456789abcdef"; for(int s=28;s>=0;s-=4) *(volatile uint32_t*)0xE0000000 = d[(v>>s)&15]; *(volatile uint32_t*)0xE0000000='\n'; }
static volatile int32_t yout;
int main(void) {
  for (int w = 0; w < NWIN; w++) {
    for (int i = 0; i < 12; i++) xin[i] = WINQ[w][i];
    MARK_ON();
    yout = predict();                          /* volatile store: the result is complete */
    __asm volatile ("" ::: "memory");          /* before the stop marker */
    MARK_OFF();
    int32_t y = yout;
    puthex((uint32_t)y);
  }
  return 0;
}
