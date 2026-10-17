/* PREREG11: cycle counts of one forecast on ATmega128 (MICA2), IEEE float32 (software FP), weights in flash. */
#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "avr_mcu_section.h"
#include "models.h"
AVR_MCU(7372800, "atmega128");
AVR_MCU_SIMAVR_CONSOLE(&UDR1);
static int uput(char c, FILE *s) { UDR1 = c; return 0; }
static FILE out = FDEV_SETUP_STREAM(uput, NULL, _FDEV_SETUP_WRITE);
volatile uint16_t ovf = 0; ISR(TIMER1_OVF_vect) { ovf++; }
static uint32_t cyc(void) {
  uint8_t s = SREG; cli(); uint16_t t = TCNT1; uint16_t o = ovf;
  if ((TIFR & (1 << TOV1)) && t < 0x8000) o++;
  SREG = s; return ((uint32_t)o << 16) | t; }
#define F(a, i) pgm_read_float(&(a)[i])
#define LB 12
__attribute__((noinline)) float p_empty(const float *w) { return 0.0f; }
__attribute__((noinline)) float p_persistence(const float *w) { return w[LB - 1]; }
__attribute__((noinline)) float p_expsm(const float *w) { float s = 0; for (uint8_t i = 0; i < LB; i++) s += F(EW, i) * w[i]; return s; }
static float lin(const float *w, uint8_t k) {
  const float *x = w + LB - k; float m = 0, tm = (k - 1) * 0.5f, den = 0, num = 0;
  for (uint8_t i = 0; i < k; i++) m += x[i]; m /= k;
  for (uint8_t i = 0; i < k; i++) { float tc = i - tm; num += (x[i] - m) * tc; den += tc * tc; }
  float sl = num / den; return sl * k + (m - sl * tm); }
__attribute__((noinline)) float p_lin2(const float *w) { return lin(w, 2); }
__attribute__((noinline)) float p_lin12(const float *w) { return lin(w, 12); }
__attribute__((noinline)) float p_light(const float *w) {
  float z[LB], o = F(L_B2, 0);
  for (uint8_t i = 0; i < LB; i++) z[i] = (w[i] - F(MX, i)) * F(ISX, i);
  for (uint8_t j = 0; j < 16; j++) { float a = F(L_B1, j); for (uint8_t i = 0; i < LB; i++) a += z[i] * F(L_K1, i * 16 + j); if (a > 0) o += a * F(L_K2, j); }
  return o * SY + MY; }
static float sg(float x) { return 1.0f / (1.0f + expf(-x)); }
__attribute__((noinline)) float p_cnn(const float *w) {
  static float H[LB][32]; float z[LB + 2], h[32], c[32], g[128];
  z[0] = 0; z[LB + 1] = 0;
  for (uint8_t i = 0; i < LB; i++) z[i + 1] = (w[i] - F(MX, i)) * F(ISX, i);
  for (uint8_t t = 0; t < LB; t++) for (uint8_t f = 0; f < 32; f++) {
    float a = F(C_CB, f); for (uint8_t k = 0; k < 3; k++) a += z[t + k] * F(C_CK, k * 32 + f); H[t][f] = a > 0 ? a : 0; }
  memset(h, 0, sizeof h); memset(c, 0, sizeof c);
  for (uint8_t t = 0; t < LB; t++) {
    for (uint8_t q = 0; q < 128; q++) { float a = F(C_LB, q);
      for (uint8_t i = 0; i < 32; i++) a += H[t][i] * F(C_LK, (uint16_t)i * 128 + q) + h[i] * F(C_LRK, (uint16_t)i * 128 + q);
      g[q] = a; }
    for (uint8_t u = 0; u < 32; u++) {
      float ig = sg(g[u]), fg = sg(g[32 + u]), cc = tanhf(g[64 + u]), og = sg(g[96 + u]);
      c[u] = fg * c[u] + ig * cc; h[u] = og * tanhf(c[u]); } }
  float o = F(C_DB, 0); for (uint8_t i = 0; i < 32; i++) o += h[i] * F(C_DK, i);
  return o * SY + MY; }
typedef float (*pf)(const float *);
static void run(const char *name, pf p, uint32_t base) {
  float w[LB]; volatile float r;
  for (uint8_t n = 0; n < NWIN; n++) {
    for (uint8_t i = 0; i < LB; i++) w[i] = F(WIN, n * LB + i);
    uint32_t a = cyc(); r = p(w); uint32_t b = cyc();
    float rv = r; uint32_t bits; memcpy(&bits, &rv, 4);
    printf("%s %lu %08lx\n", name, (unsigned long)(b - a - base), (unsigned long)bits); } }
int main(void) {
  stdout = &out; TCCR1B = (1 << CS10); TIMSK = (1 << TOIE1); sei();
  float w[LB]; for (uint8_t i = 0; i < LB; i++) w[i] = F(WIN, i);
  volatile float r; uint32_t a = cyc(); r = p_empty(w); uint32_t base = cyc() - a; (void)r;
  printf("base %lu\n", (unsigned long)base);
  run("persistence", p_persistence, base); run("expsm", p_expsm, base); run("lin2", p_lin2, base);
  run("lin12", p_lin12, base); run("lightweight", p_light, base); run("cnn_lstm", p_cnn, base);
  printf("done\n"); cli(); sleep_enable(); sleep_cpu(); }
