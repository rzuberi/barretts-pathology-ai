import pandas as pd, os, h5py, glob
A="/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_panakeia"
c=pd.read_csv(A+"/cohort.csv"); c["exists"]=c.slide_path.map(os.path.exists)
def mpp(p):
    f=p.replace("/slides/","/features/20x_224px/features_uni_v2/").replace(".svs",".h5")
    if not os.path.exists(f): return None
    with h5py.File(f) as h: return round(float(h.attrs["mpp_used"]),2)
s=c[(c.cohort=="stad")&c.exists].head(40).copy(); s["m"]=s.slide_path.map(mpp)
pick=list(s[s.m<0.3].patient[:2])+list(s[s.m>0.4].patient[:1])+list(c[c.cohort=="esca"].patient[:1])+list(c[~c.exists].sort_values("file_size").patient[:1])
open(A+"/code/smoke_pids.txt","w").write("\n".join(pick)+"\n"); print(pick)
with open(A+"/code/op_smoke_prep.tsv","w") as f:
    for p in pick: f.write(f"{p}\tvirchow2\tpython {A}/code/prep_slide.py {p}\n")
