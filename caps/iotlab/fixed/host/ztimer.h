#define ZTIMER_MSEC 0
static inline unsigned long ztimer_now(int z){ (void)z; return 0; }
static inline void ztimer_sleep(int z, int ms){ (void)z; (void)ms; }
