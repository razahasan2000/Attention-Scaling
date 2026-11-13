"""Stage 3 theory: the divisor-power law for softmax attention.

Claim (small-logit expansion): for i.i.d. pre-softmax logits X_i ~ N(0, s^2)
over n attended positions, the KL divergence between the attention
distributions induced by two divisors c1 > c2 (beta_i = 1/c_i) is

    KL(P_b1 || P_b2)  ~=  0.5 * (b1 - b2)^2 * s^2 * (1 - 1/n)        (Eq. L1)

to second order in (beta * sigma). This script validates Eq. (L1) by Monte
Carlo and tabulates:
  - exact KL vs sigma for the divisor pairs used in the paper (d_k = 32, 128)
  - attention entropy and expected max weight vs sigma for each divisor
    (the near-uniform regime: H -> log n, max w -> 1/n as sigma -> 0)

The same law predicts the measured operating points: Stage-2 replica runs sit
at sigma_raw ~ 1 (KL between divisors ~ 0.01 nats, inert); the Stage-3 logit
gain g moves the operating point to sigma ~ g (divisor begins to bite).
"""

import json
import os
import math

import numpy as np

RESULTS = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results"))


def softmax_rows(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp_keepdims(A):
    m = A.max(-1, keepdims=True)
    return m + np.log(np.exp(A - m).sum(-1, keepdims=True))


def kl_mc(sigma, n, b1, b2, reps=200_000, rng=None, chunk=50_000):
    """Monte-Carlo KL(P_{b1} || P_{b2}) for iid N(0, sigma^2) logits."""
    rng = rng or np.random.default_rng(0)
    total = 0.0
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        X = rng.standard_normal((m, n)) * sigma
        P = softmax_rows(X * b1)
        logP = X * b1 - logsumexp_keepdims(X * b1)
        logQ = X * b2 - logsumexp_keepdims(X * b2)
        total += float((P * (logP - logQ)).sum())
        done += m
    return total / reps


def kl_law(sigma, n, b1, b2):
    return 0.5 * (b1 - b2) ** 2 * sigma ** 2 * (1.0 - 1.0 / n)


def main():
    rng = np.random.default_rng(0)
    out = {"law": "KL(P_b1||P_b2) ~= 0.5*(b1-b2)^2*sigma^2*(1-1/n)",
           "grid": {}}

    for dk in (32, 128):
        for n in (64, 128):
            betas = {"standard": 1.0 / math.sqrt(dk),
                     "temp": 1.0 / (2.0 * math.sqrt(dk)),
                     "dk": 1.0 / float(dk)}
            sigmas = [0.5, 1, 2, 3, 4, 6, 8, 12, 16]
            rows = []
            for s in sigmas:
                row = {"sigma": s}
                for a, b in (("standard", "dk"), ("standard", "temp"),
                             ("dk", "temp")):
                    key = f"{a}_vs_{b}"
                    row[f"kl_{key}"] = round(kl_mc(s, n, betas[a], betas[b], rng=rng), 5)
                    row[f"law_{key}"] = round(kl_law(s, n, betas[a], betas[b]), 5)
                rows.append(row)
            out["grid"][f"dk{dk}_n{n}"] = {"betas": betas, "rows": rows}

    # Entropy / max-weight table: d_k = 32, n = 64
    dk, n = 32, 64
    betas = {"standard": 1.0 / math.sqrt(dk), "temp": 1.0 / (2.0 * math.sqrt(dk)),
             "dk": 1.0 / float(dk)}
    ent_rows = []
    for s in [0.5, 1, 2, 4, 8, 16]:
        row = {"sigma": s}
        X = rng.standard_normal((200_000, n)) * s
        for name, b in betas.items():
            P = softmax_rows(X * b)
            H = -(P * np.log(P + 1e-12)).sum(-1)
            row[f"H_{name}"] = round(float(H.mean()), 3)
            row[f"maxw_{name}"] = round(float(P.max(-1).mean()), 4)
        row["H_uniform"] = round(math.log(n), 3)
        row["maxw_uniform"] = round(1.0 / n, 4)
        ent_rows.append(row)
    out["entropy_n64_dk32"] = ent_rows

    # Key thresholds quoted in the paper (n = 64, d_k = 32)
    b_std, b_dk, b_temp = betas["standard"], betas["dk"], betas["temp"]
    out["thresholds"] = {
        "delta_beta_std_dk": round(b_std - b_dk, 5),
        "delta_beta_std_temp": round(b_std - b_temp, 5),
        "sigma_for_KL_0.01_std_dk": round(math.sqrt(0.02 / ((b_std - b_dk) ** 2 * (1 - 1 / 64))), 3),
        "sigma_for_KL_0.1_std_dk": round(math.sqrt(0.2 / ((b_std - b_dk) ** 2 * (1 - 1 / 64))), 3),
        "sigma_for_KL_0.01_std_temp": round(math.sqrt(0.02 / ((b_std - b_temp) ** 2 * (1 - 1 / 64))), 3),
        "sigma_for_KL_0.1_std_temp": round(math.sqrt(0.2 / ((b_std - b_temp) ** 2 * (1 - 1 / 64))), 3),
    }

    with open(f"{RESULTS}/stage3_theory.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["thresholds"], indent=1))
    print("\nValidation (dk=32, n=64), KL(std||dk):")
    for row in out["grid"]["dk32_n64"]["rows"]:
        print(f"  sigma={row['sigma']:>5}  MC={row['kl_standard_vs_dk']:.5f}  "
              f"law={row['law_standard_vs_dk']:.5f}  ratio="
              f"{row['kl_standard_vs_dk']/max(row['law_standard_vs_dk'],1e-9):.3f}")
    print("\nEntropy table (d_k=32, n=64):")
    for row in ent_rows:
        print(f"  sigma={row['sigma']:>5}  H std/temp/dk = "
              f"{row['H_standard']}/{row['H_temp']}/{row['H_dk']}  "
              f"maxw std/temp/dk = {row['maxw_standard']}/{row['maxw_temp']}/{row['maxw_dk']}")


if __name__ == "__main__":
    main()
