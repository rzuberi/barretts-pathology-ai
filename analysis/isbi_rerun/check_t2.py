import json, glob, numpy as np, pandas as pd
A="/mnt/scratche/slow/fmlab/zuberi01/analysis/isbi_rerun"; R=json.load(open(A+"/results.json"))
print("== n_slides / class_counts per arm-set (same y for all arms by construction?)")
for hold in ["patient","report"]:
    for task in ["bin","six"]:
        for T in ["T1","T2","T3"]:
            r=R[hold][task][T]; print(hold,task,T,"n=",r.get("n_slides"),"pats=",r.get("n_patients"),r.get("class_counts"))
print("== T2 six confusion dict orientation + row/col sums")
for hold in ["patient","report"]:
    cm=R[hold]["six"]["T2"]["confusion_truth_rows_pred_cols"]
    for arm in "ABC":
        df=pd.DataFrame(cm[arm])  # to_dict() gives {col: {row: v}} -> DataFrame columns=pred, index=truth
        print(hold,arm,"index(truth) sums:",df.sum(1).to_dict()); print(hold,arm,"column(pred) sums:",df.sum(0).to_dict())
print("== holdout_idx identity across arms/seeds/folds (per hold, task)")
for hold in ["patient","report"]:
    for task in ["bin","six"]:
        sets={}
        for f in sorted(glob.glob(A+f"/preds/{hold}_*_{task}_seed*_fold*.npz")):
            z=np.load(f); sets[f.split("/")[-1]]=(frozenset(z["holdout_idx"].tolist()), len(z["test_idx"]))
        hs=set(v[0] for v in sets.values()); print(hold,task,"files",len(sets),"distinct holdout sets:",len(hs),"size",[len(h) for h in hs])
        # test idx: union per arm identical?
        for arm in "ABC":
            u=set()
            for k,v in sets.items():
                if f"_{arm}_" in k: u|=set(np.load(A+"/preds/"+k)["test_idx"].tolist())
            print("  ",arm,"union test idx",len(u))
print("== timing from train_log (patient holdout, arm B)")
tl=pd.DataFrame(R["train_log"]); tl["hold"]=tl.tag.str.split("_").str[0]; tl["arm"]=tl.tag.str.split("_").str[1]; tl["task"]=tl.tag.str.split("_").str[2]
print(tl[(tl.hold=="patient")&(tl.arm=="B")].groupby("task")[["n_train","sec"]].agg(["mean","min","max"]).round(1))
print("total sec all 180:", tl.sec.sum(), "per fold min:", tl.groupby(tl.tag.str.extract(r"fold(\d)")[0]).sec.sum().round(0).to_dict())
s=pd.read_csv(A+"/slides.csv",dtype={"idx":int}).fillna("")
excl=set(pd.read_csv(A+"/qc/exclusions.csv").idx); s=s[~s.idx.isin(excl)]
print("== patients per fold (non-human):", s[s.fold>=0].groupby("fold").anon_id.nunique().to_dict(), "total", s[s.fold>=0].anon_id.nunique())
tr=s[(s.fold>=0)&(s.worst_grade!="")]
print("train patients per fold-job (4 folds), bin-eligible:", {f:int(tr[(tr.fold!=f)&tr.worst_grade.isin(["NORMAL_OTHER","NDBE","LGD","HGD","CANCER"])].anon_id.nunique()) for f in range(5)})
print("slides per patient (train pool):", round(len(tr)/tr.anon_id.nunique(),2))
print("human: patients",s.human_patient.sum()>0 and s[s.human_patient].anon_id.nunique(),"T2 graded slides",(s.human_grade!="").sum())
