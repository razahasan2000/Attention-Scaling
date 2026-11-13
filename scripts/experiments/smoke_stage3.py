"""Smoke test for Stage-3 machinery: gain, qk_norm, softcap, probe, matched init."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common as C

C.set_threads()

# 1. matched init still bit-identical for plain variants
m1 = C.build_matched(1001, ["standard", "dk"], vocab=500, d_model=64, n_heads=2,
                     n_layers=2, d_ff=128, dropout=0.1, max_len=64,
                     task="classification", n_classes=4)
same = all(torch.equal(m1["standard"].state_dict()[k], m1["dk"].state_dict()[k])
           for k in m1["standard"].state_dict())
print("matched init (plain):", same)

# 2. tiny training + probe with gain, and KL law sanity
d = np.load(f"{C.PROC_DIR}/agnews.npz")
Xtr, ytr = d["Xtr"][:512], d["ytr"][:512]
Xva, yva = d["Xva"][:256], d["yva"][:256]
for g in (1.0, 8.0):
    mm = C.build_matched(1001, ["standard"], vocab=12000, d_model=128, n_heads=4,
                         n_layers=2, d_ff=256, dropout=0.1, max_len=64,
                         task="classification", n_classes=4, logit_gain=g)["standard"]
    r = C.train_cls(mm, Xtr, ytr, Xva, yva, 1001, epochs=1, bs=32, lr=3e-4)
    p = C.probe_summary(C.probe(mm, torch.from_numpy(Xva).long()))
    print(f"gain={g}: val_acc={r['val_acc']:.3f} sigma_raw={p['sigma_raw']} "
          f"sigma_post={p.get('sigma_post_own')} ent={p.get('entropy_own')} "
          f"kl_std_dk={p.get('kl_std_dk')} kl_std_temp={p.get('kl_std_temp')}")
    # law check: KL(std||dk) ~= 0.5*(b1-b2)^2 * sigma_raw^2 * (1-1/nbar)
    dk = 32.0
    law = 0.5 * (1/np.sqrt(dk) - 1/dk)**2 * p["sigma_raw"]**2 * (1 - 1/45)
    print(f"   law prediction kl_std_dk ~= {law:.4f}")

# 3. QK-Norm absorbs the gain injection (forward invariance)
for g in (1.0, 8.0):
    mq = C.build_matched(1001, ["standard"], vocab=12000, d_model=128, n_heads=4,
                         n_layers=2, d_ff=256, dropout=0.1, max_len=64,
                         task="classification", n_classes=4, qk_norm=True,
                         logit_gain=g)["standard"]
    ids = torch.from_numpy(Xva[:32]).long()
    out = mq(ids)
    pq = C.probe_summary(C.probe(mq, ids))
    print(f"qknorm gain={g}: fwd std={float(out.std()):.5f} "
          f"sigma_raw={pq['sigma_raw']} kl_std_dk={pq.get('kl_std_dk')}")

# 4. softcap forward sanity
ms = C.build_matched(1001, ["standard"], vocab=12000, d_model=128, n_heads=4,
                     n_layers=2, d_ff=256, dropout=0.1, max_len=64,
                     task="classification", n_classes=4, softcap=3.0,
                     logit_gain=8)["standard"]
ids = torch.from_numpy(Xva[:32]).long()
out = ms(ids)
ps = C.probe_summary(C.probe(ms, ids))
print("softcap(3) gain=8: sigma_raw =", ps["sigma_raw"], "ent =", ps.get("entropy_own"))
print("SMOKE OK")
