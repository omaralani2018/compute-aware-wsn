APPLICATION = netx
BOARD ?= iotlab-m3
RIOTBASE ?= $(CURDIR)/../../iotlab/RIOT
ROLE ?= TX
CFLAGS += -DROLE_$(ROLE)=1
ifeq ($(NETX_INT),1)
  CFLAGS += -DNETX_INT=1
endif
ifneq ($(PASSES),)
  CFLAGS += -DPASSES=$(PASSES)
endif
ifneq ($(PERIOD_CNN_MS),)
  CFLAGS += -DPERIOD_CNN_MS=$(PERIOD_CNN_MS)
endif
USEMODULE += ztimer_msec
USEMODULE += netdev_default
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc
USEMODULE += gnrc_tx_sync
DEVELHELP ?= 0
QUIET ?= 1
include $(RIOTBASE)/Makefile.include
