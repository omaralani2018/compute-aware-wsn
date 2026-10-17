/* PREREG17 AMENDMENT_17b: sink v2 (not measured). Same radio set-up as the PREREG17 sink. Instead of one line per frame it
 * keeps, per stream and arm, the cumulative number of frames received and an order-independent hash of their contents
 * (sum of mix(stream, arm, round, reading) mod 2^32), and prints them every 10 s with a check value over the line's numbers,
 * so that lines corrupted by the serial path are detected and ignored:  "RXV s a count hash chk"  (hash, chk in hex). */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ztimer.h"
#include "net/gnrc.h"
#include "net/gnrc/netif.h"
#include "net/netopt.h"

#define CHANNEL 22
#define PAN_ID 0x0023
#define ADDR_RX 0x0002
#define PLEN 7

static gnrc_netif_t *nif;
static int set16(netopt_t o, uint16_t v) { return gnrc_netapi_set(nif->pid, o, 0, &v, sizeof(v)); }
static uint32_t fmix(uint32_t h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
static uint32_t chk(uint32_t s, uint32_t a, uint32_t c, uint32_t h) { return fmix(fmix(fmix(fmix(s) ^ a) ^ c) ^ h); }
static msg_t q[16];
static uint32_t cnt[16][5], hsh[16][5];
int main(void) {
    ztimer_sleep(ZTIMER_MSEC, 2000);
    nif = gnrc_netif_iter(NULL);
    if (!nif) { printf("ERROR no network interface\n"); while (1) ztimer_sleep(ZTIMER_MSEC, 60000); }
    int r1 = set16(NETOPT_CHANNEL, CHANNEL), r2 = set16(NETOPT_NID, PAN_ID);
    uint8_t a2[2] = { ADDR_RX >> 8, ADDR_RX & 0xff };
    int r3 = gnrc_netapi_set(nif->pid, NETOPT_ADDRESS, 0, a2, sizeof(a2));
    int r4 = set16(NETOPT_SRC_LEN, 2);
    netopt_state_t st = NETOPT_STATE_IDLE;
    int r5 = gnrc_netapi_set(nif->pid, NETOPT_STATE, 0, &st, sizeof(st));
    printf("HDR netx board=%s role=RX sink=v2 ch=%d pan=0x%04x addr=0x%04x set=%d,%d,%d,%d,%d\n", RIOT_BOARD, CHANNEL, PAN_ID, ADDR_RX, r1, r2, r3, r4, r5);
    msg_init_queue(q, 16);
    gnrc_netreg_entry_t me = GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me);
    uint32_t n = 0, other = 0, last = ztimer_now(ZTIMER_MSEC);
    while (1) {
        msg_t m;
        if (ztimer_msg_receive_timeout(ZTIMER_MSEC, &m, 500) >= 0 && m.type == GNRC_NETAPI_MSG_TYPE_RCV) {
            gnrc_pktsnip_t *p = m.content.ptr; n++;
            const uint8_t *d = p->data;
            if (p->size == PLEN && (d[0] >> 4) < 16 && (d[0] & 15) < 5) {
                uint32_t s = d[0] >> 4, a = d[0] & 15, r = d[1] | ((uint32_t)d[2] << 8), v;
                memcpy(&v, d + 3, 4);
                cnt[s][a]++; hsh[s][a] += fmix(((s << 28) | (a << 24) | r) ^ fmix(v));
            } else other++;
            gnrc_pktbuf_release(p);
        }
        if (ztimer_now(ZTIMER_MSEC) - last >= 10000) {
            last = ztimer_now(ZTIMER_MSEC);
            printf("RX count=%lu other=%lu t=%lu\n", (unsigned long)n, (unsigned long)other, (unsigned long)last);
            for (uint32_t s = 0; s < 16; s++) for (uint32_t a = 0; a < 5; a++) if (cnt[s][a])
                printf("RXV %lu %lu %lu %08lx %08lx\n", (unsigned long)s, (unsigned long)a, (unsigned long)cnt[s][a],
                       (unsigned long)hsh[s][a], (unsigned long)chk(s, a, cnt[s][a], hsh[s][a]));
        }
    }
    return 0;
}
