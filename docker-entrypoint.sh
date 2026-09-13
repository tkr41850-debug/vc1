#!/bin/bash
# Gateway entrypoint: prepare host bits the in-process warp supervisor needs,
# then exec the gateway. Warp itself stays lazy — llms boots supervised pools
# for enabled warp providers at startup and heals them across restarts; with
# no warp-cli installed pools stay unhealthy and traffic fails open to direct.
set -euo pipefail
DATA_ROOT="${DATA_DIR:-/data}"
mkdir -p "$DATA_ROOT"
if [ -n "${WARP_NET_MTU:-}" ]; then
  IFACE=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
  if [ -z "$IFACE" ]; then
    IFACE=$(ip route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
  fi
  IFACE="${IFACE:-eth0}"
  ip link set dev "$IFACE" mtu "$WARP_NET_MTU" 2>&1 || echo "WARN: could not set MTU on $IFACE" >&2
  echo "MTU $WARP_NET_MTU on $IFACE"
fi
if command -v warp-cli >/dev/null 2>&1; then
  if [ ! -e /dev/net/tun ]; then
    echo "WARN: /dev/net/tun missing (run with devices: [/dev/net/tun] + NET_ADMIN)" >&2
  fi
else
  echo "note: warp-cli not installed; warp providers stay unhealthy (direct fallback)"
fi
exec uv run uvicorn llms.proxy.main:app --host 0.0.0.0 --port "${ZEN_GATEWAY_PORT:-8789}"
