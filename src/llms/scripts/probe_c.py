import time
from llms.proxy.egress import WarpSocksEgress

# Item 2: no-hole behavior must equal old slots[slot % len]
for ports in ([40001], [40001, 40002, 40003], [44002, 44003]):
    w = WarpSocksEgress("p", num_slots=len(ports))
    w.set_socks_ports(list(ports))
    for slot in range(12):
        old = ports[slot % len(ports)]
        new = w.pick_port(slot)
        assert new == old, (ports, slot, old, new)
    assert w.ready_ports() == list(ports), (ports, w.ready_ports())
    assert all(p is not None for p in w.ready_ports())
print("item2 PASS: no-hole pick_port == old behavior, ready_ports has no None")

w0 = WarpSocksEgress("p0")
w0.set_socks_ports([])
assert w0.pick_port(0) is None and w0.ready_ports() == []
print("item3a PASS: empty spread -> pick None")
try:
    w0.client_for(0, 0)
    print("item3a FAIL: no raise")
except RuntimeError:
    print("item3a PASS: client_for raises RuntimeError on empty")
