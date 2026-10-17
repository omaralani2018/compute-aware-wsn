/* ARM host check for PREREG17: the firmware's own predictor and transmit-rule code (app/netx_pred.h), compiled with the
 * firmware's compiler, flags and newlib, run under qemu-arm (Linux user mode). Prints, per stream and arm, the number of
 * transmissions and the decision bitmap in the firmware's format. */
#include "../app/netx_pred.h"
static void sys_write(const char *s, int n) {
    register int r0 __asm__("r0") = 1; register const char *r1 __asm__("r1") = s; register int r2 __asm__("r2") = n;
    register int r7 __asm__("r7") = 4;
    __asm__ volatile("svc 0" : "+r"(r0) : "r"(r1), "r"(r2), "r"(r7) : "memory");
}
static void sys_exit(int c) { register int r0 __asm__("r0") = c; register int r7 __asm__("r7") = 1; __asm__ volatile("svc 0" :: "r"(r0), "r"(r7)); for (;;) {} }
static char ob[256]; static int on;
static void put(const char *s) { while (*s) ob[on++] = *s++; }
static void flush(void) { sys_write(ob, on); on = 0; }
static void putu(unsigned v) { char t[12]; int n = 0; do { t[n++] = '0' + v % 10; v /= 10; } while (v); while (n) ob[on++] = t[--n]; }
static void puthex(const uint8_t *b, int n) { const char *h = "0123456789abcdef"; for (int i = 0; i < n; i++) { ob[on++] = h[b[i] >> 4]; ob[on++] = h[b[i] & 15]; if (on > 200) flush(); } }
static uint8_t bm[(NETX_R + 7) / 8];
static const char *const ARMN[5] = { "persistence", "expsm", "linear_extrap", "lightweight", "cnn_lstm" };
int main(void) {
    put(BUILD); put(" ref"); for (int s = 0; s < NETX_NS; s++) { put(" "); putu(ref_arm(s)); } put("\n"); flush();
    for (int s = 0; s < NETX_NS; s++) for (int a = 0; a < 5; a++) {
        for (int i = 0; i < LB; i++) RB[i] = reading(s, i);
        memset(bm, 0, sizeof bm); unsigned ntx = 0;
        for (int r = 0; r < NETX_R; r++) {
            int t = LB + r; val_t p = forecast(s, a, &RB[t - LB]); val_t x = reading(s, t);
            if (misses(s, x, p)) { RB[t] = x; bm[r >> 3] |= (uint8_t)(1u << (r & 7)); ntx++; } else RB[t] = p;
        }
        put("D "); put(NETX_NAME[s]); put(":"); put(ARMN[a]); put(" "); putu(ntx); put(" "); puthex(bm, sizeof bm); put("\n"); flush();
    }
    sys_exit(0); return 0;
}
__attribute__((naked, noreturn)) void _start(void) { __asm__ volatile("bl main\n b ."); }
