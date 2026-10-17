#include "netx_models.h"
typedef struct { const float *X, *MX, *ISX, *EW; float MY, SY, UNIT; int K, ref;
  const float *L_K1, *L_B1, *L_K2, *L_B2, *C_CK, *C_CB, *C_LK, *C_LRK, *C_LB, *C_DK, *C_DB; } netx_stream_t;
static const char *const NETX_NAME[NETX_NS] = {"taxi","exchange_cpc","machine_temp","cpu"};
static const netx_stream_t NS[NETX_NS] = {
  { X0, MX0, ISX0, EW0, MY0, SY0, UNIT0, K0, 2, L_K1_0, L_B1_0, L_K2_0, L_B2_0, C_CK_0, C_CB_0, C_LK_0, C_LRK_0, C_LB_0, C_DK_0, C_DB_0 },
  { X1, MX1, ISX1, EW1, MY1, SY1, UNIT1, K1, 1, L_K1_1, L_B1_1, L_K2_1, L_B2_1, C_CK_1, C_CB_1, C_LK_1, C_LRK_1, C_LB_1, C_DK_1, C_DB_1 },
  { X2, MX2, ISX2, EW2, MY2, SY2, UNIT2, K2, 1, L_K1_2, L_B1_2, L_K2_2, L_B2_2, C_CK_2, C_CB_2, C_LK_2, C_LRK_2, C_LB_2, C_DK_2, C_DB_2 },
  { X3, MX3, ISX3, EW3, MY3, SY3, UNIT3, K3, 1, L_K1_3, L_B1_3, L_K2_3, L_B2_3, C_CK_3, C_CB_3, C_LK_3, C_LRK_3, C_LB_3, C_DK_3, C_DB_3 },
};
