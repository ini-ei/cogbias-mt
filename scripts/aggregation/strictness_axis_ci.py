# -*- coding: utf-8 -*-
"""Per-axis evaluator strictness (cross-model T2) + problem-level bootstrap 95% CI."""
import json, os
import random

FR=["gpt-5","claude-opus-4.7","gemini-3.1-pro","llama-4-maverick"]
OP_JA=["qwen3.5-27b","qwen3.5-122b","gemma-4-31b"]
JA="results/phase5_v2"; EN="results/phase5_v2_en"

def load(p):
    if not os.path.exists(p): return None
    rs=[r for r in json.load(open(p)) if isinstance(r,dict) and r.get("problem_id")
        and (r.get("turn2_bias_confidence") is not None or r.get("turn4_bias_confidence") is not None)]
    return {r["problem_id"]:r for r in rs}

def merged(base, d, subj, patch=None):
    m=load(f"{base}/{d}/{subj}_results.json")
    if m is None: return None
    if patch:
        pm=load(f"{base}/{patch}/{subj}_results.json")
        if pm: m.update(pm)
    return m

CFG={
 ("gpt-5","JA"):dict(fr=(JA,"cross_eval_gpt5_n80",None),op=(JA,"cross_eval_gpt5_open",None)),
 ("claude-opus-4.7","JA"):dict(fr=(JA,"cross_eval_claude_opus_n80",None),op=(JA,"cross_eval_claude_opus_n80",None)),
 ("gemini-3.1-pro","JA"):dict(fr=(JA,"cross_eval_gemini_n80",None),op=(JA,"cross_eval_gemini_open",None)),
 ("llama-4-maverick","JA"):dict(fr=(JA,"cross_eval_llama_n80",None),op=(JA,"cross_eval_llama_n80",None)),
 ("gpt-5","EN"):dict(fr=(EN,"cross_eval_gpt5_en","cross_eval_gpt5_en_patch"),op=(EN,"cross_eval_gpt5_en_open",None)),
 ("claude-opus-4.7","EN"):dict(fr=(EN,"cross_eval_claude_opus_en",None),op=(EN,"cross_eval_claude_opus_en",None)),
 ("gemini-3.1-pro","EN"):dict(fr=(EN,"cross_eval_gemini_en","cross_eval_gemini_en_patch"),op=(EN,"cross_eval_gemini_en_open",None)),
 ("llama-4-maverick","EN"):dict(fr=(EN,"cross_eval_llama_en",None),op=(EN,"cross_eval_llama_en",None)),
}
rng=random.Random(42)
out={}
for (ev,lang),cfg in CFG.items():
    for grp,subjects in (("Frontier",[s for s in FR if s!=ev]),("Open",OP_JA)):
        b,d,patch=cfg["fr"] if grp=="Frontier" else cfg["op"]
        data={}
        for s_ in subjects:
            m=merged(b,d,s_,patch)
            assert m is not None, (ev,lang,grp,s_)
            data[s_]={pid:1 if r.get("turn2_bias_detected") else 0 for pid,r in m.items()}
        pids=sorted(set().union(*[set(v) for v in data.values()]))
        point=100*sum(sum(v.values()) for v in data.values())/sum(len(v) for v in data.values())
        # 問題レベル bootstrap: 80問復元抽出、被験は全保持
        stats=[]
        for _ in range(10000):
            sample=[pids[rng.randrange(len(pids))] for _ in range(len(pids))]
            num=den=0
            for v in data.values():
                for pid in sample:
                    if pid in v: num+=v[pid]; den+=1
            stats.append(100*num/den)
        stats.sort()
        lo,hi=stats[249],stats[9749]
        out[f"{ev}|{lang}|{grp}"]=dict(point=round(point,1),ci=[round(lo,1),round(hi,1)])
        print(f"{ev:20s} {lang} {grp:9s} {point:5.1f} [{lo:5.1f},{hi:5.1f}]")
json.dump(out, open("results/rebuttal_checks/strictness_ci.json","w"), indent=1)
print("saved results/rebuttal_checks/strictness_ci.json")
