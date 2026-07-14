# -*- coding: utf-8 -*-
"""n=7 の Spearman exact two-sided p (7! 全列挙)。results/rebuttal_checks/cross_eval_rho.json を更新。"""
import json, itertools
from scipy.stats import spearmanr, rankdata
path="results/rebuttal_checks/cross_eval_rho.json"
d=json.load(open(path)); ps=d["per_subject"]; subs=sorted(ps.keys())
def exact_p(x,y):
    rho=spearmanr(x,y).correlation; rx=rankdata(x); ry=list(rankdata(y))
    hits=sum(1 for perm in itertools.permutations(ry)
             if abs(spearmanr(rx,perm).correlation)>=abs(rho)-1e-12)
    return rho, hits/5040
for view,key in (("self","self_eval"),("cross","cross_eval")):
    x=[ps[m][f"{view}_ja"] for m in subs]; y=[ps[m][f"{view}_en"] for m in subs]
    rho,p=exact_p(x,y)
    d[f"{key}_exact_p"]=round(p,4)
    print(view, round(rho,4), round(p,4))
d["note_exact"]="exact two-sided p by full enumeration of 7! rank permutations (ties handled by midranks); asymptotic p retained for reference"
json.dump(d,open(path,"w"),indent=1)
print("updated", path)
