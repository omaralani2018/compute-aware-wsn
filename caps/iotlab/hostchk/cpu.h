#include <stdint.h>
typedef struct { uint32_t DEMCR; } cd_t; typedef struct { uint32_t CYCCNT, CTRL; } dwt_t;
static cd_t cd_s; static dwt_t dwt_s;
#define CoreDebug (&cd_s)
#define DWT (&dwt_s)
#define CoreDebug_DEMCR_TRCENA_Msk 1
#define DWT_CTRL_CYCCNTENA_Msk 1
#define CLOCK_CORECLOCK 1
#define RIOT_BOARD "host"
