/* PREREG15: measured energy of one report on the IoT-LAB nodes' own IEEE 802.15.4 radios (RIOT, GNRC link layer only).
 * ROLE_TX: power phases for the consumption monitor, repeated ROUNDS times:
 *   IDLE        radio asleep, CPU idle (ztimer sleep), as the IDLE phase of PREREG12;
 *   RUN <v>     back-to-back reports for PHASE_MS; one report = wake radio, send one frame, wait for the end of the
 *               transmission (gnrc_tx_sync), put the radio back to sleep. Variants v:
 *                 S_BC  4-byte payload, broadcast (no acknowledgement)            -- primary
 *                 S_UC  4-byte payload, unicast to the RX node, acknowledgement requested
 *                 L_BC  100-byte payload, broadcast
 *   LISTEN      radio awake in receive mode, no transmission (idle-listening power, context only).
 *   SYNC        once before the rounds: 12 s with the radio in receive mode, for the clock alignment.
 *   Nothing is printed inside a phase. Counts, cycles and the link-layer success/failure counters are printed after it.
 * ROLE_RX: listens on the same channel and PAN with short address 0x0002 and counts received frames. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cpu.h"
#include "periph_conf.h"
#include "ztimer.h"
#include "net/gnrc.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netif/hdr.h"
#include "net/gnrc/tx_sync.h"
#include "net/netstats.h"
#include "net/netopt.h"

#define PHASE_MS 4000
#define IDLE_MS 2000
#define ROUNDS 3
#define CHANNEL 22
#define PAN_ID 0x0023
#define ADDR_TX 0x0001
#define ADDR_RX 0x0002
#define SMALL 4
#define LARGE 100

static inline void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t cyc(void) { return DWT->CYCCNT; }

static gnrc_netif_t *nif;
static __attribute__((unused)) uint8_t payload[LARGE];

static int set16(netopt_t o, uint16_t v) { return gnrc_netapi_set(nif->pid, o, 0, &v, sizeof(v)); }
static int set_state(netopt_state_t s) { return gnrc_netapi_set(nif->pid, NETOPT_STATE, 0, &s, sizeof(s)); }
static __attribute__((unused)) void stats(uint32_t *ok, uint32_t *fail) {
    netstats_t st;                       /* NETOPT_STATS/NETSTATS_LAYER2 copies the counters (data_len = sizeof) */
    memset(&st, 0, sizeof(st));
    *ok = *fail = 0;
    if (gnrc_netapi_get(nif->pid, NETOPT_STATS, NETSTATS_LAYER2, &st, sizeof(st)) == (int)sizeof(st)) {
        *ok = st.tx_success; *fail = st.tx_failed;
    }
}

static void common_setup(uint16_t addr) {
    nif = gnrc_netif_iter(NULL);
    int r1 = set16(NETOPT_CHANNEL, CHANNEL);
    int r2 = set16(NETOPT_NID, PAN_ID);
    uint8_t a2[2] = { addr >> 8, addr & 0xff };   /* NETOPT_ADDRESS takes the address bytes in network order */
    int r3 = gnrc_netapi_set(nif->pid, NETOPT_ADDRESS, 0, a2, sizeof(a2));
    int r4 = set16(NETOPT_SRC_LEN, 2);
    int16_t pw = 0; gnrc_netapi_get(nif->pid, NETOPT_TX_POWER, 0, &pw, sizeof(pw));
    printf("HDR board=%s role=%s core_hz=%lu ch=%d pan=0x%04x addr=0x%04x txpower_dbm=%d set=%d,%d,%d,%d phase_ms=%d idle_ms=%d rounds=%d small=%d large=%d\n",
           RIOT_BOARD, addr == ADDR_TX ? "TX" : "RX", (unsigned long)CLOCK_CORECLOCK, CHANNEL, PAN_ID, addr, pw,
           r1, r2, r3, r4, PHASE_MS, IDLE_MS, ROUNDS, SMALL, LARGE);
}

#ifdef ROLE_RX
static msg_t q[16];
int main(void) {
    ztimer_sleep(ZTIMER_MSEC, 2000);
    common_setup(ADDR_RX);
    printf("RX state_idle=%d\n", set_state(NETOPT_STATE_IDLE));
    msg_init_queue(q, 16);
    gnrc_netreg_entry_t me = GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me);
    uint32_t n = 0, last = ztimer_now(ZTIMER_MSEC);
    while (1) {
        msg_t m;
        if (ztimer_msg_receive_timeout(ZTIMER_MSEC, &m, 1000) >= 0 && m.type == GNRC_NETAPI_MSG_TYPE_RCV) {
            n++; gnrc_pktbuf_release(m.content.ptr);
        }
        if (ztimer_now(ZTIMER_MSEC) - last >= 10000) {
            last = ztimer_now(ZTIMER_MSEC);
            printf("RX count=%lu t=%lu\n", (unsigned long)n, (unsigned long)last);
        }
    }
    return 0;
}
#else
static int send_one(int len, int unicast) {
    gnrc_pktsnip_t *pkt = gnrc_pktbuf_add(NULL, payload, len, GNRC_NETTYPE_UNDEF);
    if (!pkt) return -1;
    uint8_t dst[2] = { ADDR_RX >> 8, ADDR_RX & 0xff };
    gnrc_pktsnip_t *hdr = unicast ? gnrc_netif_hdr_build(NULL, 0, dst, 2) : gnrc_netif_hdr_build(NULL, 0, NULL, 0);
    if (!hdr) { gnrc_pktbuf_release(pkt); return -2; }
    pkt = gnrc_pkt_prepend(pkt, hdr);
    if (!unicast) ((gnrc_netif_hdr_t *)hdr->data)->flags = GNRC_NETIF_HDR_FLAGS_BROADCAST;
    gnrc_tx_sync_t sync;
    if (gnrc_tx_sync_append(pkt, &sync)) { gnrc_pktbuf_release(pkt); return -3; }
    if (gnrc_netif_send(nif, pkt) < 1) { gnrc_pktbuf_release(pkt); return -4; }
    gnrc_tx_sync(&sync);
    return 0;
}

static void idle_phase(int round) {
    printf("B IDLE begin %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    ztimer_sleep(ZTIMER_MSEC, IDLE_MS);
    printf("B IDLE end %d t=%lu\n", round, (unsigned long)ztimer_now(ZTIMER_MSEC));
}

static const char *VN[] = { "S_BC", "S_UC", "L_BC" };
static const int VLEN[] = { SMALL, SMALL, LARGE }, VUNI[] = { 0, 1, 0 };

static void run_phase(int v, int round) {
    uint32_t ok0, f0, ok1, f1, calls = 0, cycles = 0, errs = 0;
    stats(&ok0, &f0);
    printf("B RUN %s begin %d t=%lu\n", VN[v], round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    uint32_t t0 = ztimer_now(ZTIMER_MSEC);
    while (ztimer_now(ZTIMER_MSEC) - t0 < PHASE_MS) {
        uint32_t a = cyc();
        set_state(NETOPT_STATE_IDLE);
        if (send_one(VLEN[v], VUNI[v]) != 0) errs++;
        set_state(NETOPT_STATE_SLEEP);
        cycles += cyc() - a; calls++;
    }
    uint32_t t1 = ztimer_now(ZTIMER_MSEC);
    stats(&ok1, &f1);
    printf("B RUN %s end %d t=%lu calls=%lu cycles=%lu wall_ms=%lu tx_ok=%lu tx_fail=%lu send_err=%lu\n", VN[v], round,
           (unsigned long)t1, (unsigned long)calls, (unsigned long)cycles, (unsigned long)(t1 - t0),
           (unsigned long)(ok1 - ok0), (unsigned long)(f1 - f0), (unsigned long)errs);
}

static void listen_phase(const char *name, int round, uint32_t ms) {
    printf("B RUN %s begin %d t=%lu\n", name, round, (unsigned long)ztimer_now(ZTIMER_MSEC));
    set_state(NETOPT_STATE_IDLE);
    ztimer_sleep(ZTIMER_MSEC, ms);
    set_state(NETOPT_STATE_SLEEP);
    printf("B RUN %s end %d t=%lu\n", name, round, (unsigned long)ztimer_now(ZTIMER_MSEC));
}

int main(void) {
    dwt_init();
    for (int i = 0; i < LARGE; i++) payload[i] = (uint8_t)i;
    ztimer_sleep(ZTIMER_MSEC, 3000);   /* let the serial aggregator attach */
    common_setup(ADDR_TX);
    netopt_enable_t en = NETOPT_ENABLE;
    int ra = gnrc_netapi_set(nif->pid, NETOPT_ACK_REQ, 0, &en, sizeof(en));
    int rs = set_state(NETOPT_STATE_SLEEP);
    printf("TX ack_req_set=%d sleep_set=%d\n", ra, rs);
    /* SYNC: one long receive-on block before the rounds, so that the clock offset between the power trace and the
     * serial log has a unique best match (the rounds repeat with a period of about 30 s) */
    idle_phase(-1); listen_phase("SYNC", -1, 3 * PHASE_MS); idle_phase(-1);
    for (int r = 0; r < ROUNDS; r++) {
        idle_phase(r);
        for (int v = 0; v < 3; v++) { run_phase(v, r); idle_phase(r); }
        listen_phase("LISTEN", r, PHASE_MS); idle_phase(r);
    }
    printf("ALL DONE\n");
    while (1) ztimer_sleep(ZTIMER_MSEC, 60000);
    return 0;
}
#endif
