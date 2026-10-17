APPLICATION = caps_radio
BOARD ?= iotlab-m3
RIOTBASE ?= $(CURDIR)/../RIOT
ROLE ?= TX
CFLAGS += -DROLE_$(ROLE)=1
USEMODULE += ztimer_msec
USEMODULE += netdev_default
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc
USEMODULE += gnrc_tx_sync
USEMODULE += netstats_l2
DEVELHELP ?= 0
QUIET ?= 1
include $(RIOTBASE)/Makefile.include
