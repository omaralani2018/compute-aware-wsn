/* PREREG17 (NETX): dual prediction run end to end on IoT-LAB nodes over their own IEEE 802.15.4 radios (RIOT, GNRC link layer).
 * ROLE_TX (sender, measured): for each pass, stream and arm (reference, lightweight, CNN-LSTM):
 *   RUN  R rounds at a fixed period. Each round: forecast from the shared state; if the forecast misses by more than tau,
 *        wake the radio, send one 7-byte unicast frame with acknowledgement {stream<<4|arm, round (2 bytes), reading (4 bytes)},
 *        wait for the end of the transmission, put the radio back to sleep, and store the reading; otherwise store the forecast.
 *        Then sleep (CPU idle, radio asleep) to the next period boundary.
 *   IDLE 5 s, radio asleep, CPU idle, between all RUN phases.
 *   SYNC 12 s with the radio in receive mode, at the start and at the end (clock alignment of power trace and serial log).
 *   Nothing is printed inside a phase. After each RUN: transmissions, send errors, forecast and active cycles, late rounds,
 *   wall time, and the transmit decisions as a bitmap (round r is bit r&7 of byte r>>3), in hex.
 * ROLE_RX (sink, not measured): listens with short address 0x0002 and prints every frame it receives.
 * Build NETX_INT=1: integer predictors and readings (PREREG13 scheme, bit-exact with qemu.py; M3 only). */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "cpu.h"
#include "periph_conf.h"
#include "ztimer.h"
#include "net/gnrc.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netif/hdr.h"
#include "net/gnrc/tx_sync.h"
#include "net/netopt.h"

#define CHANNEL 22
#define PAN_ID 0x0023
#define ADDR_TX 0x0001
#define ADDR_RX 0x0002
#define PLEN 7
#define IDLE_MS 5000
#define SYNC_MS 12000
#ifndef PASSES
#define PASSES 3
#endif

static gnrc_netif_t *nif;
static int set16(netopt_t o, uint16_t v) { return gnrc_netapi_set(nif->pid, o, 0, &v, sizeof(v)); }
static int set_state(netopt_state_t s) { return gnrc_netapi_set(nif->pid, NETOPT_STATE, 0, &s, sizeof(s)); }
static void common_setup(uint16_t addr) {
    nif = gnrc_netif_iter(NULL);
    if (!nif) return;
    int r1 = set16(NETOPT_CHANNEL, CHANNEL), r2 = set16(NETOPT_NID, PAN_ID);
    uint8_t a2[2] = { addr >> 8, addr & 0xff };
    int r3 = gnrc_netapi_set(nif->pid, NETOPT_ADDRESS, 0, a2, sizeof(a2));
    int r4 = set16(NETOPT_SRC_LEN, 2);
    int16_t pw = 0; gnrc_netapi_get(nif->pid, NETOPT_TX_POWER, 0, &pw, sizeof(pw));
    printf("HDR netx board=%s role=%s core_hz=%lu ch=%d pan=0x%04x addr=0x%04x txpower_dbm=%d set=%d,%d,%d,%d\n",
           RIOT_BOARD, addr == ADDR_TX ? "TX" : "RX", (unsigned long)CLOCK_CORECLOCK, CHANNEL, PAN_ID, addr, pw, r1, r2, r3, r4);
}

#ifdef ROLE_RX
static msg_t q[16];
int main(void) {
    ztimer_sleep(ZTIMER_MSEC, 2000);
    common_setup(ADDR_RX);
    if (!nif) { printf("ERROR no network interface\n"); while (1) ztimer_sleep(ZTIMER_MSEC, 60000); }
    printf("RX state_idle=%d\n", set_state(NETOPT_STATE_IDLE));
    msg_init_queue(q, 16);
    gnrc_netreg_entry_t me = GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me);
    uint32_t n = 0, last = ztimer_now(ZTIMER_MSEC);
    while (1) {
        msg_t m;
        if (ztimer_msg_receive_timeout(ZTIMER_MSEC, &m, 1000) >= 0 && m.type == GNRC_NETAPI_MSG_TYPE_RCV) {
            gnrc_pktsnip_t *p = m.content.ptr; n++;
            if (p->size == PLEN) {
                const uint8_t *d = p->data;
                printf("F %d %d %u %02x%02x%02x%02x\n", d[0] >> 4, d[0] & 15, (unsigned)(d[1] | (d[2] << 8)), d[3], d[4], d[5], d[6]);
            }
            gnrc_pktbuf_release(p);
        }
        if (ztimer_now(ZTIMER_MSEC) - last >= 10000) { last = ztimer_now(ZTIMER_MSEC); printf("RX count=%lu t=%lu\n", (unsigned long)n, (unsigned long)last); }
    }
    return 0;
}
#else
static inline void dwt_init(void) { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; }
static inline uint32_t cyc(void) { return DWT->CYCCNT; }
#include "netx_pred.h"

#ifndef PERIOD_FAST_MS
#define PERIOD_FAST_MS 20
#endif
#ifndef PERIOD_CNN_MS
#define PERIOD_CNN_MS 120
#endif
static const char *const ARMN[5] = { "persistence", "expsm", "linear_extrap", "lightweight", "cnn_lstm" };
static uint8_t payload[PLEN], bm[(NETX_R + 7) / 8];

static int send_one(void) {
    gnrc_pktsnip_t *pkt = gnrc_pktbuf_add(NULL, payload, PLEN, GNRC_NETTYPE_UNDEF);
    if (!pkt) return -1;
    uint8_t dst[2] = { ADDR_RX >> 8, ADDR_RX & 0xff };
    gnrc_pktsnip_t *hdr = gnrc_netif_hdr_build(NULL, 0, dst, 2);
    if (!hdr) { gnrc_pktbuf_release(pkt); return -2; }
    pkt = gnrc_pkt_prepend(pkt, hdr);
    gnrc_tx_sync_t sync;
    if (gnrc_tx_sync_append(pkt, &sync)) { gnrc_pktbuf_release(pkt); return -3; }
    if (gnrc_netif_send(nif, pkt) < 1) { gnrc_pktbuf_release(pkt); return -4; }
    gnrc_tx_sync(&sync);
    return 0;
}
static void idle_phase(int pass) {
    printf("B IDLE begin %d t=%lu\n", pass, (unsigned long)ztimer_now(ZTIMER_MSEC));
    ztimer_sleep(ZTIMER_MSEC, IDLE_MS);
    printf("B IDLE end %d t=%lu\n", pass, (unsigned long)ztimer_now(ZTIMER_MSEC));
}
static void listen_phase(int pass) {
    printf("B RUN SYNC begin %d t=%lu\n", pass, (unsigned long)ztimer_now(ZTIMER_MSEC));
    set_state(NETOPT_STATE_IDLE); ztimer_sleep(ZTIMER_MSEC, SYNC_MS); set_state(NETOPT_STATE_SLEEP);
    printf("B RUN SYNC end %d t=%lu\n", pass, (unsigned long)ztimer_now(ZTIMER_MSEC));
}
static void run_phase(int s, int a, int pass) {
    const uint32_t P = a == 4 ? PERIOD_CNN_MS : PERIOD_FAST_MS;
    uint32_t ntx = 0, err = 0, late = 0; uint64_t fcyc = 0, acyc = 0;
    for (int i = 0; i < LB; i++) RB[i] = reading(s, i);
    memset(bm, 0, sizeof bm);
    payload[0] = (uint8_t)((s << 4) | a);
    printf("B RUN %s:%s begin %d t=%lu period_ms=%lu\n", NETX_NAME[s], ARMN[a], pass, (unsigned long)ztimer_now(ZTIMER_MSEC), (unsigned long)P);
    uint32_t last = ztimer_now(ZTIMER_MSEC), t0 = last;
    for (int r = 0; r < NETX_R; r++) {
        int t = LB + r;
        uint32_t c0 = cyc();
        val_t p = forecast(s, a, &RB[t - LB]);
        uint32_t c1 = cyc();
        val_t x = reading(s, t);
        if (misses(s, x, p)) {
            RB[t] = x; bm[r >> 3] |= (uint8_t)(1u << (r & 7)); ntx++;
            payload[1] = (uint8_t)r; payload[2] = (uint8_t)(r >> 8); memcpy(payload + 3, &x, 4);
            set_state(NETOPT_STATE_IDLE);
            if (send_one() != 0) err++;
            set_state(NETOPT_STATE_SLEEP);
        } else RB[t] = p;
        uint32_t c2 = cyc();
        fcyc += c1 - c0; acyc += c2 - c0;
        if (ztimer_now(ZTIMER_MSEC) - last >= P) late++;
        ztimer_periodic_wakeup(ZTIMER_MSEC, &last, P);
    }
    uint32_t t1 = ztimer_now(ZTIMER_MSEC);
    printf("B RUN %s:%s end %d t=%lu tx=%lu send_err=%lu late=%lu fkcyc=%lu akcyc=%lu wall_ms=%lu\n", NETX_NAME[s], ARMN[a], pass,
           (unsigned long)t1, (unsigned long)ntx, (unsigned long)err, (unsigned long)late, (unsigned long)(fcyc / 1000),
           (unsigned long)(acyc / 1000), (unsigned long)(t1 - t0));   /* thousands of cycles: no long long in printf */
    printf("D %s:%s %d ", NETX_NAME[s], ARMN[a], pass);
    for (unsigned i = 0; i < sizeof bm; i++) printf("%02x", bm[i]);
    printf("\n");
}
int main(void) {
    dwt_init();
    ztimer_sleep(ZTIMER_MSEC, 60000);   /* AMENDMENT_15a: time for the serial capture to attach, no reset */
    common_setup(ADDR_TX);
    if (!nif) { printf("ERROR no network interface\n"); while (1) ztimer_sleep(ZTIMER_MSEC, 60000); }
    netopt_enable_t en = NETOPT_ENABLE;
    int ra = gnrc_netapi_set(nif->pid, NETOPT_ACK_REQ, 0, &en, sizeof(en));
    int rs = set_state(NETOPT_STATE_SLEEP);
    printf("TX build=%s R=%d passes=%d period_fast_ms=%d period_cnn_ms=%d ack_req_set=%d sleep_set=%d\n",
           BUILD, NETX_R, PASSES, PERIOD_FAST_MS, PERIOD_CNN_MS, ra, rs);
    idle_phase(-1); listen_phase(-1); idle_phase(-1);
    for (int pass = 0; pass < PASSES; pass++)
        for (int s = 0; s < NETX_NS; s++) {
            const int arms[3] = { ref_arm(s), 3, 4 };
            for (int k = 0; k < 3; k++) { run_phase(s, arms[k], pass); idle_phase(pass); }
        }
    listen_phase(PASSES); idle_phase(PASSES);
    set_state(NETOPT_STATE_IDLE);       /* AMENDMENT_15a: leave the radio awake, so that a later reset can initialise it */
    printf("ALL DONE\n");
    while (1) ztimer_sleep(ZTIMER_MSEC, 60000);
    return 0;
}
#endif
