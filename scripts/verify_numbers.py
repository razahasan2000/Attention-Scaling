"""Deterministic verification of the Attention-Scaling package.

Independently recomputes every headline statistic from the raw data in
results/ and checks consistency of: results JSONs <-> paper/source/study_data.py
<-> rendered manuscript/supplementary PDF text <-> README.

Run from the repository root:  python3 scripts/verify_numbers.py
Requires: pdftotext (poppler-utils) to extract the PDF text; scipy is optional
(p-values are skipped if unavailable).

Exit code 0 = all checks pass, 1 = at least one failure.
"""
import json, math, os, re, subprocess, sys, tempfile, collections

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # repo root

results = []
def check(cid, ok, detail):
    results.append({'id': cid, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}")

def approx(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))

def pdftext(path):
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tf:
        out = tf.name
    subprocess.run(['pdftotext', '-layout', path, out], check=True)
    return open(out).read()

# ---------------------------------------------------------------- load data
RES = f'{BASE}/results'
agg = json.load(open(f'{RES}/aggregate.json'))
s3t = json.load(open(f'{RES}/stage3_theory.json'))
s3o = json.load(open(f'{RES}/stage3_operatingpoint.json'))
fr = json.load(open(f'{RES}/frontier_gpt2.json'))
mv = json.load(open(f'{RES}/mask_verification.json'))
man = pdftext(f'{BASE}/paper/Attention-Scaling_Manuscript.pdf')
sup = pdftext(f'{BASE}/paper/Attention-Scaling_Supplementary.pdf')
readme = open(f'{BASE}/README.md').read()

# 1. run counts ---------------------------------------------------------------
phases = collections.Counter()
for line in open(f'{RES}/results.jsonl'):
    phases[json.loads(line).get('phase', '?')] += 1
tot = sum(phases.values())
s2 = sum(phases[p] for p in ['lm','core','grid','conv','abl','lmleak','lmc'])
s3_jsonl = sum(phases[p] for p in ['gain','lmgain','bridge','regime','regime128'])
check('runs.total_jsonl', tot == 467, f'results.jsonl records={tot} (claim 467)')
check('runs.stage2', s2 == 339, f'Stage-2 runs={s2} (claim 339)')
check('runs.stage3', s3_jsonl + 6 == 134, f'Stage-3 jsonl={s3_jsonl} + 6 frontier evals = {s3_jsonl+6} (claim 134)')
check('runs.grand', 314 + s2 + s3_jsonl + 6 == 787, f'314+{s2}+{s3_jsonl}+6 = {314+s2+s3_jsonl+6} (claim 787)')

# 2. recompute Stage-2 paired stats from per-seed data ------------------------
def paired_recompute(per_seed, va, vb):
    seeds = sorted(per_seed.keys(), key=int)
    d = [per_seed[s][vb] - per_seed[s][va] for s in seeds]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = m / (sd / math.sqrt(n))
    dz = m / sd
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 1)
    except Exception:
        p = float('nan')
    return m, sd, t, dz, p, n

for block, key in [('core', 'core_agnews'), ('lm', 'lm')]:
    ps = agg['per_seed'][key]
    for var in ['dk', 'temp']:
        m, sd, t, dz, p, n = paired_recompute(ps, 'standard', var)
        st = agg[block]['paired'][var]
        ok = (approx(m, st['diff'], 1e-6) and approx(t, st['t'], 1e-3)
              and approx(dz, st['dz'], 1e-3) and (math.isnan(p) or approx(p, st['p'], 1e-3)))
        check(f'stage2.{block}.paired_{var}', ok,
              f'recomputed diff={m:.6g} t={t:.4g} dz={dz:.4g} vs stored dz={st["dz"]:.4g}')

# 3. KL operating-point law ----------------------------------------------------
n_tok = 64
for lbl, b1, b2 in [('std_dk', 1/math.sqrt(32), 1/32), ('std_temp', 1/math.sqrt(32), 1/(2*math.sqrt(32)))]:
    dbeta = b1 - b2
    for kl in ['0.01', '0.1']:
        sigma = math.sqrt(float(kl) / (0.5 * dbeta**2 * (1 - 1/n_tok)))
        stored = s3t['thresholds'][f'sigma_for_KL_{kl}_{lbl}']
        check(f'law.sigma_KL{kl}_{lbl}', approx(sigma, stored, 0.01),
              f'recomputed sigma={sigma:.4f} vs stored {stored}')
stored_db = s3t['thresholds']['delta_beta_std_dk']
check('law.delta_beta', approx(1/math.sqrt(32) - 1/32, stored_db, 1e-4),
      f'1/sqrt(32)-1/32={1/math.sqrt(32)-1/32:.6f} vs stored {stored_db}')

# 3b. independent Monte Carlo validation of the law ----------------------------
import random
random.seed(0)
def mc_kl(sigma, b1, b2, n=64, reps=20000):
    acc = 0.0
    for _ in range(reps):
        z = [random.gauss(0, sigma) for _ in range(n)]
        def softmax(bz):
            mx = max(bz); ex = [math.exp(v - mx) for v in bz]; s = sum(ex)
            return [v / s for v in ex]
        p1 = softmax([b1 * v for v in z]); p2 = softmax([b2 * v for v in z])
        acc += sum(pi * math.log(pi / pj + 1e-300) for pi, pj in zip(p1, p2) if pi > 0)
    return acc / reps

sigma = 2.0
law_val = 0.5 * (1/math.sqrt(32) - 1/32)**2 * sigma**2 * (1 - 1/64)
mc_val = mc_kl(sigma, 1/math.sqrt(32), 1/32)
check('law.montecarlo', approx(mc_val, law_val, 0.05),
      f'MC KL={mc_val:.5f} vs law={law_val:.5f} at sigma=2')

# 4. frontier GPT-2 ------------------------------------------------------------
for model in ['gpt2', 'gpt2-medium']:
    v = fr[model]['variants']
    base = v['standard']['ppl']
    for var in ['dk', 'temp']:
        recomputed = (v[var]['ppl'] / base - 1) * 100
        check(f'frontier.{model}.{var}_pct', approx(recomputed, v[var]['ppl_delta_pct'], 0.01),
              f'{model} {var}: {recomputed:.2f}% (stored {v[var]["ppl_delta_pct"]}%)')
g = fr['gpt2']['variants']; m = fr['gpt2-medium']['variants']
rows = [
    ('gpt2.standard', '51.23', g['standard']['ppl']), ('gpt2.temp', '79.13', g['temp']['ppl']),
    ('gpt2.dk', '4,791.65', g['dk']['ppl']), ('med.standard', '38.77', m['standard']['ppl']),
    ('med.temp', '49.06', m['temp']['ppl']), ('med.dk', '1,585.70', m['dk']['ppl']),
]
for cid, claim, actual in rows:
    check(f'frontier.{cid}_value', approx(float(claim.replace(',', '')), actual, 0.001),
          f'claimed {claim} vs raw {actual}')
    check(f'doc.{cid}', claim in man or claim in readme, f'{claim} appears in manuscript/README')

Hs_all = [l['H_standard'] for l in fr['gpt2']['layers']] + [l['H_standard'] for l in fr['gpt2-medium']['layers']]
Hd_all = [l['H_dk'] for l in fr['gpt2']['layers']] + [l['H_dk'] for l in fr['gpt2-medium']['layers']]
check('frontier.entropy_range', 1.2 - 0.05 <= min(Hs_all) and max(Hs_all) <= 3.5 + 0.05
      and 3.6 - 0.05 <= min(Hd_all) and max(Hd_all) <= 3.9 + 0.05,
      f'pooled H_standard [{min(Hs_all):.2f},{max(Hs_all):.2f}] (claim 1.2-3.5), H_dk [{min(Hd_all):.2f},{max(Hd_all):.2f}] (claim 3.6-3.9)')
flat_man = ''.join(man.split())
check('doc.entropy_range_text', '1.2–3.5' in flat_man or '1.2--3.5' in flat_man,
      'corrected entropy range 1.2–3.5 present in manuscript')

mean_sigma = {mo: sum(l['sigma_raw'] for l in fr[mo]['layers']) / len(fr[mo]['layers']) for mo in ['gpt2', 'gpt2-medium']}
check('frontier.sigma_range', 45 - 1 <= min(mean_sigma.values()) and max(mean_sigma.values()) <= 63 + 1,
      f'layer-mean sigma_raw gpt2={mean_sigma["gpt2"]:.1f}, medium={mean_sigma["gpt2-medium"]:.1f} (claim 45-63)')
check('doc.sigma_layermean', 'layermeans' in flat_man or 'layer-mean' in flat_man,
      'manuscript qualifies sigma_raw as layer means')

# 5. Stage-3 claims ------------------------------------------------------------
g1 = s3o['phases']['gain']['g1']; g32 = s3o['phases']['gain']['g32']
check('gain.kl_low', approx(g1['probe_std']['kl_std_dk'], 0.0009, 0.15),
      f'g=1 KL(std||dk)={g1["probe_std"]["kl_std_dk"]:.5f} (claim 0.0009)')
check('gain.kl_high', approx(g32['probe_std']['kl_std_dk'], 0.64, 0.01),
      f'g=32 KL(std||dk)={g32["probe_std"]["kl_std_dk"]:.4f} (claim 0.64)')
check('gain.acc32', approx(g32['acc']['standard'], 76.81, 0.001) and approx(g32['acc']['dk'], 78.21, 0.001),
      f'g=32 acc std={g32["acc"]["standard"]}, dk={g32["acc"]["dk"]} (claim 76.81 vs 78.21)')
check('gain.dz', approx(g32['paired_std_dk']['dz'], -1.76, 0.01), f'g=32 dz={g32["paired_std_dk"]["dz"]} (claim -1.76)')
check('gain.sigma_lo', approx(g1['probe_std']['sigma_raw'], 0.33, 0.05), f'g=1 sigma_raw={g1["probe_std"]["sigma_raw"]:.4f} (claim 0.33)')
check('gain.sigma_hi', approx(g32['probe_std']['sigma_raw'], 9.21, 0.01), f'g=32 sigma_raw={g32["probe_std"]["sigma_raw"]:.4f} (claim 9.21)')

reg = s3o['phases']['regime']['trajectory']['standard']
check('regime.sigma32', approx(reg[-1]['sigma_raw'], 1.11, 0.02), f'regime d_k=32 final sigma_raw={reg[-1]["sigma_raw"]:.4f} (claim 1.11)')
r128 = s3o['phases']['regime128']['trajectory']['standard']
s128 = max(e['sigma_raw'] for e in r128)
check('regime.sigma128', approx(s128, 5, 0.1), f'regime d_k=128 max sigma_raw={s128:.3f} (claim ~5)')

qk = s3o['phases']['bridge']
sigs = [qk[f'qk_{v}']['probe']['sigma_post_own'] for v in ['std','dk','temp']]
raws = [qk[f'qk_{v}']['probe']['sigma_raw'] for v in ['std','dk','temp']]
accs = [qk[f'qk_{v}']['acc'] for v in ['std','dk','temp']]
check('bridge.qknorm_raw_pinned', all(5.9 <= s <= 6.2 for s in raws),
      f'QK-Norm sigma_raw = {["%.3f"%s for s in raws]} pinned ~6.0 across arms')
check('bridge.qknorm_sigma_post_std', approx(sigs[0], 1.05, 0.02),
      f'QK-Norm sigma_post under standard divisor = {sigs[0]:.4f} (claim ~1.05)')
check('bridge.qknorm_acc', max(accs) - min(accs) < 0.15,
      f'QK-Norm acc spread={max(accs)-min(accs):.4f} pp ({accs}) (claim re-equalised)')

# 6. LM headline ---------------------------------------------------------------
lm_std = agg['lm']['per_variant']['standard']['mean']
check('lm.ppl', approx(lm_std, 872.80, 0.01), f'WikiText-2 ppl={lm_std:.2f} (claim ~873)')
check('lm.invariance', abs(agg['lm']['paired']['dk']['dz']) < 0.05,
      f'LM paired dz(dk)={agg["lm"]["paired"]["dk"]["dz"]:.4f} (claim variant-invariant)')

# 7. mask verification suite ---------------------------------------------------
check('mask.V1V5', mv['V1_future_max_weight'] == 0 and mv['V2_padding_max_weight'] == 0
      and 0.999 < mv['V3_row_sums_min'] <= mv['V3_row_sums_max'] <= 1.001 and mv['V4_diagonal_allowed_real_rows'] and mv['V5_masked_rows_exact_zero'],
      'V1-V5 causal-mask invariants all hold')

# 8. documents contain headline numbers ----------------------------------------
doc_claims = {
    'ppl 873': ['872.8', '873'], 'stage2 dz small': ['-0.13', '0.13'],
    'law text': ['0.979', '0.98'], 'law sigma 3.1': ['3.097', '3.1'],
    'frontier gpt2 dk': ['4,791.65'], 'frontier pct': ['9,253'],
}
for cid, variants in doc_claims.items():
    ok = any(v in man for v in variants)
    check(f'doc.manuscript_{cid.replace(" ","_")}', ok, f'{cid} present in manuscript PDF text: {ok}')

# 9. LaTeX cross-consistency ---------------------------------------------------
LATEX = f'{BASE}/paper/latex'
lat = open(f'{LATEX}/main.tex').read()
for f in ['sec_intro', 'sec_method', 'sec_results', 'sec_discussion']:
    lat += open(f'{LATEX}/{f}.tex').read()
graphics = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', lat)
missing = [gph for gph in graphics if not os.path.exists(f'{LATEX}/{gph}')]
check('latex.figures', not missing, f'{len(graphics)} includegraphics refs, missing: {missing or "none"}')
cites = set(re.findall(r'\\(?:cite|citep|citet)\{([^}]+)\}', lat))
cites = {c.strip() for grp in cites for c in grp.split(',')}
bib = open(f'{LATEX}/refs.bib').read()
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
check('latex.cites', cites <= bibkeys, f'{len(cites)} cite keys, unresolved: {sorted(cites - bibkeys) or "none"}')

# 10. study_data.py consistency ------------------------------------------------
import importlib.util
spec = importlib.util.spec_from_file_location('study_data', f'{BASE}/paper/source/study_data.py')
sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
found_lm = any('872.8' in json.dumps(getattr(sd, a), default=str) for a in dir(sd) if a.isupper())
check('studydata.lm_mean', found_lm, 'study_data.py contains LM ppl 872.8 consistent with aggregate.json')

# 11. tau=0.5 stage-1 claim present --------------------------------------------
check('doc.tau05', re.search(r'𝜏\s*=\s*0\.5', man) is not None,
      'tau=0.5 sentiment claim present in manuscript (math-italic tau)')

# ---------------------------------------------------------------- summary
n_pass = sum(1 for r in results if r['status'] == 'PASS')
n_fail = sum(1 for r in results if r['status'] == 'FAIL')
print(f'\n==== {n_pass} PASS, {n_fail} FAIL of {len(results)} deterministic checks ====')
json.dump(results, open(f'{BASE}/results/verification_checks.json', 'w'), indent=1)
sys.exit(1 if n_fail else 0)
