import json,sys,os
M="/dev/shm/big-small-ouro/runs/"
names=sys.argv[1:]
wins=[(2e6,4e6),(5e6,7e6),(9e6,11e6),(14e6,16e6),(19e6,21e6),(29e6,31e6),(38e6,40e6)]
print("run".ljust(22)+"".join(f"{int(a/1e6)}-{int(b/1e6)}M".rjust(9) for a,b in wins))
for n in names:
    p=M+n+"/train.jsonl"
    if not os.path.exists(p): continue
    rs=[json.loads(l) for l in open(p)]
    out=[]
    for a,b in wins:
        sel=[r["kl"] for r in rs if a<=r.get("kl_tokens",0)<=b]
        out.append(f"{sum(sel)/len(sel):.3f}" if sel else "  -  ")
    print(n.ljust(22)+"".join(x.rjust(9) for x in out))
