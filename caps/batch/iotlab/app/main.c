/* PREREG16: energy of batched reports on the IoT-LAB nodes' own IEEE 802.15.4 radios (RIOT, GNRC link layer only).
 * Same radio set-up as PREREG15 (channel 22, PAN 0x0023, broadcast, default power and CSMA-CA).
 * ROLE_TX, after SYNC (as PREREG15), ROUNDS rounds of:
 *   SWEEP: for each payload size in SIZES: RUN SW<size> PHASE_MS of back-to-back broadcast frames, then IDLE.
 *   REPLAY: for each cell (stream x scheme, sched.h) and the NULL cell: one reading every PERIOD_MS; at the readings
 *           listed for the cell the radio is woken, one frame of the listed size is sent, the radio is put to sleep;
 *           between readings the CPU sleeps until the next reading (ztimer_periodic_wakeup). Then IDLE.
 * Nothing is printed inside a phase. Frame tag: payload[0] = phase id, payload[1] = round (counted by the RX node). */
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ztimer.h"
#include "net/gnrc.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netif/hdr.h"
#include "net/gnrc/tx_sync.h"
#include "net/netopt.h"
#include "replay_tab.h"

#define PHASE_MS 4000
#define IDLE_MS 2000
#define PERIOD_MS 20
#define ROUNDS 5
#define CHANNEL 22
#define PAN_ID 0x0023
#define ADDR_TX 0x0001
#define ADDR_RX 0x0002
#define MAXLEN 100
#define NULL_READ 1500
static const int SIZES[] = { 4, 20, 40, 60, 80, 100 };
#define N_SIZES 6
#define N_IDS (N_SIZES + N_CELLS + 1)

static gnrc_netif_t *nif;
static uint8_t payload[MAXLEN];
static int set16(netopt_t o, uint16_t v) { return gnrc_netapi_set(nif->pid, o, 0, &v, sizeof(v)); }
static int set_state(netopt_state_t s) { return gnrc_netapi_set(nif->pid, NETOPT_STATE, 0, &s, sizeof(s)); }

static void common_setup(uint16_t addr) {
    nif = gnrc_netif_iter(NULL);
    if (!nif) return;
    int r1 = set16(NETOPT_CHANNEL, CHANNEL), r2 = set16(NETOPT_NID, PAN_ID);
    uint8_t a2[2] = { addr >> 8, addr & 0xff };
    int r3 = gnrc_netapi_set(nif->pid, NETOPT_ADDRESS, 0, a2, sizeof(a2)), r4 = set16(NETOPT_SRC_LEN, 2);
    int16_t pw = 0; gnrc_netapi_get(nif->pid, NETOPT_TX_POWER, 0, &pw, sizeof(pw));
    printf("HDR board=%s role=%s ch=%d pan=0x%04x addr=0x%04x txpower_dbm=%d set=%d,%d,%d,%d phase_ms=%d idle_ms=%d period_ms=%d rounds=%d cells=%d\n",
           RIOT_BOARD, addr == ADDR_TX ? "TX" : "RX", CHANNEL, PAN_ID, addr, pw, r1, r2, r3, r4, PHASE_MS, IDLE_MS, PERIOD_MS,
           ROUNDS, N_CELLS);
}

#ifdef ROLE_RX
static msg_t q[16];
static uint32_t cnt[N_IDS][ROUNDS];
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
            if (p->size >= 2) {
                uint8_t v = ((uint8_t *)p->data)[0], r = ((uint8_t *)p->data)[1];
                if (v < N_IDS && r < ROUNDS) cnt[v][r]++;
            }
            gnrc_pktbuf_release(p);
        }
        if (ztimer_now(ZTIMER_MSEC) - last >= 10000) {
            last = ztimer_now(ZTIMER_MSEC);
            printf("RX count=%lu t=%lu\n", (unsigned long)n, (unsigned long)last);
            for (int v = 0; v < N_IDS; v++) for (int r = 0; r < ROUNDS; r++)
                if (cnt[v][r]) printf("RXV %d %d %lu\n", v, r, (unsigned long)cnt[v][r]);
        }
    }
    return 0;
}
#else
static int send_one(int len) {
    gnrc_pktsnip_t *pkt = gnrc_pktbuf_add(NULL, payload, len, GNRC_NETTYPE_UNDEF);
    if (!pkt) return -1;
    gnrc_pktsnip_t *hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
    if (!hdr) { gnrc_pktbuf_release(pkt); return -2; }
    pkt = gnrc_pkt_prepend(pkt, hdr);
    ((gnrc_netif_hdr_t *)hdr->data)->flags = GNRC_NETIF_HDR_FLAGS_BROADCAST;
    gnrc_tx_sync_t sync;
    if (gnrc_tx_sync_append(pkt, &sync)) { gnrc_pktbuf_release(pkt); return -3; }
    if (gnrc_netif_send(nif, pkt) < 1) { gnrc_pktbuf_release(pkt); return -4; }
    gnrc_tx_sync(&sync);
    return 0;
}
static int frame(int id, int round, uint32_t k, int len) {
    payload[0] = (uint8_t)id; payload[1] = (uint8_t)round; payload[2] = (uint8_t)(k >> 8); payload[3] = (uint8_t)k;
    set_state(NETOPT_STATE_IDLE);
    int e = send_one(len);
    set_state(NETOPT_STATE_SLEEP);
    return e;
}
static void idle_phase(int round) {
    printf("B IDLE begin %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    ztimer_sleep(ZTIMER_MSEC, IDLE_MS);
    printf("B IDLE end %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
}
static void sweep_phase(int s, int round) {
    uint32_t calls = 0, errs = 0;
    printf("B RUN SW%d begin %d t=%lu\n", SIZES[s], round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    uint32_t t0 = ztimer_now(ZTIMER_MSEC);
    while (ztimer_now(ZTIMER_MSEC) - t0 < PHASE_MS) { if (frame(s, round, calls, SIZES[s])) errs++; calls++; }
    uint32_t t1 = ztimer_now(ZTIMER_MSEC);
    printf("B RUN SW%d end %d t=%lu calls=%lu wall_ms=%lu send_err=%lu\n", SIZES[s], round, (unsigned long)t1,
           (unsigned long)calls, (unsigned long)(t1 - t0), (unsigned long)errs);
}
static void replay_phase(int c, int round) {      /* c == N_CELLS: the NULL cell (no frame) */
    const char *name = c < N_CELLS ? CN[c] : "NULL";
    int nread = c < N_CELLS ? CNR[c] : NULL_READ, nf = c < N_CELLS ? CNF[c] : 0, j = 0;
    uint32_t errs = 0, late = 0;
    printf("B RUN R_%s begin %d t=%lu\n", name, round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    uint32_t t0 = ztimer_now(ZTIMER_MSEC), wake = t0;
    for (int r = 0; r < nread; r++) {
        while (j < nf && CT[c][j] == r) { if (frame(N_SIZES + c, round, j, CL[c][j])) errs++; j++; }
        if (ztimer_now(ZTIMER_MSEC) - wake >= PERIOD_MS) late++;
        ztimer_periodic_wakeup(ZTIMER_MSEC, &wake, PERIOD_MS);
    }
    uint32_t t1 = ztimer_now(ZTIMER_MSEC);
    printf("B RUN R_%s end %d t=%lu calls=%d reads=%d wall_ms=%lu send_err=%lu late=%lu\n", name, round, (unsigned long)t1, j,
           nread, (unsigned long)(t1 - t0), (unsigned long)errs, (unsigned long)late);
}
static void listen_phase(const char *name, int round, uint32_t ms) {
    printf("B RUN %s begin %d t=%lu\n", name, round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    set_state(NETOPT_STATE_IDLE); ztimer_sleep(ZTIMER_MSEC, ms); set_state(NETOPT_STATE_SLEEP);
    printf("B RUN %s end %d t=%lu\n", name, round, (unsigned long)ztimer_now(ZTIMER_MSEC));
}
int main(void) {
    for (int i = 0; i < MAXLEN; i++) payload[i] = (uint8_t)i;
    ztimer_sleep(ZTIMER_MSEC, 60000);   /* as AMENDMENT_15a: time for the serial aggregator to attach */
    common_setup(ADDR_TX);
    if (!nif) { printf("ERROR no network interface\n"); while (1) ztimer_sleep(ZTIMER_MSEC, 60000); }
    printf("TX sleep_set=%d\n", set_state(NETOPT_STATE_SLEEP));
    idle_phase(-1); listen_phase("SYNC", -1, 3 * PHASE_MS); idle_phase(-1);
    for (int r = 0; r < ROUNDS; r++) {
        idle_phase(r);
        for (int s = 0; s < N_SIZES; s++) { sweep_phase(s, r); idle_phase(r); }
        for (int c = 0; c <= N_CELLS; c++) { replay_phase(c, r); idle_phase(r); }
    }
    set_state(NETOPT_STATE_IDLE);
    printf("ALL DONE\n");
    while (1) ztimer_sleep(ZTIMER_MSEC, 60000);
    return 0;
}
#endif
