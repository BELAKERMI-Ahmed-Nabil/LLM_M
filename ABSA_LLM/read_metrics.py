import json, sys
sys.stdout.reconfigure(encoding='utf-8')

files = {
    '3b_zero':  r'C:\ABSA_LLM\results\zero_shot_results.json',
    '3b_one':   r'C:\ABSA_LLM\results\one_shot_results.json',
    '3b_few10': r'C:\ABSA_LLM\results\few_shot_v2_10ex_results.json',
    '8b_zero':  r'C:\ABSA_LLM_8b\results\zero_shot_results.json',
    '8b_one':   r'C:\ABSA_LLM_8b\results\one_shot_results.json',
    '8b_few10': r'C:\ABSA_LLM_8b\results\few_shot_v2_10ex_results.json',
    '8b_few50': r'C:\ABSA_LLM_8b\results\few_shot_v2_50ex_results.json',
}

for name, path in files.items():
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    n = d.get('n_reviews', '?')
    mp = d.get('metrics_partial', d.get('metrics', {}))
    me = d.get('metrics_exact', d.get('metrics', {}))
    quad_p = mp.get('quad', mp) if isinstance(mp, dict) and 'quad' in mp else mp
    quad_e = me.get('quad', me) if isinstance(me, dict) and 'quad' in me else me
    pf1 = quad_p.get('f1', '?') if isinstance(quad_p, dict) else '?'
    pp  = quad_p.get('precision', '?') if isinstance(quad_p, dict) else '?'
    pr  = quad_p.get('recall', '?') if isinstance(quad_p, dict) else '?'
    ef1 = quad_e.get('f1', '?') if isinstance(quad_e, dict) else '?'
    print(f"{name} (n={n}): Exact={ef1}  Partial P={pp} R={pr} F1={pf1}")

# SetFitQuad results
import os
sf_dir = r'C:\SetFitQuad-Code-main\results'
print("\n--- SetFitQuad EXP1 best results ---")
best_pf1 = 0
best_name = ''
for fname in os.listdir(sf_dir):
    if fname.endswith('_5fold_average_results.json'):
        with open(os.path.join(sf_dir, fname), encoding='utf-8') as f:
            d = json.load(f)
        pf1 = d.get('avg_partial_f1', d.get('partial_f1', '?'))
        ef1 = d.get('avg_exact_f1', d.get('exact_f1', '?'))
        print(f"  {fname[:35]}: Partial={pf1}  Exact={ef1}")
        if isinstance(pf1, float) and pf1 > best_pf1:
            best_pf1 = pf1
            best_name = fname
print(f"\nBest SetFitQuad: {best_name} -> {best_pf1}")
