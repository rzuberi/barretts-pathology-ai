"""EXECUTION_PLAN 2.1: ERIN slide<->label join.

Chain: feature h5 (named by slide-dir UUID) -> WSIExport manifest (Case Folder Name
-> Case Identifier) -> report CaseName -> jury label + patient. Output: erin_master.csv
(one row per slide with features + train-eligible label + patient id + date).
"""
import glob, os, re
import pandas as pd

ERIN = "/mnt/scratche/fast/fmlab/datasets/imaging/ERIN"
FEAT = ERIN + "/features/20x_224px/features_uni_v2"
LBL = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/labeller/erin_labels_jury_final.csv"
OUT = os.environ.get("OUTDIR", os.path.dirname(LBL))

mans = glob.glob(ERIN + "/slides/WSIExport_Result*.csv") + glob.glob(ERIN + "/raw/WSIExport_Result*.csv")
wx = pd.concat([pd.read_csv(f, dtype=str) for f in mans], ignore_index=True)
wx.columns = [c.strip() for c in wx.columns]
wx = wx[["Case Folder Name", "Case Identifier"]].dropna().drop_duplicates()
wx.columns = ["uuid", "case_id"]
wx["uuid"] = wx["uuid"].str.strip().str.lower()
wx["case_norm"] = wx["case_id"].str.extract(r"([A-Za-z]{2}\s?\d{2}[.\-/]?\d{3,6})")[0].str.upper()
conf = wx.groupby("uuid")["case_norm"].nunique()
print(f"uuids mapping to >1 normalised case: {(conf > 1).sum()} (excluded)")
wx = wx[~wx["uuid"].isin(conf[conf > 1].index)].drop_duplicates("uuid")
print(f"manifest rows: {len(wx)}")

feats = pd.DataFrame({"h5": sorted(glob.glob(os.path.join(FEAT, "*.h5")))})
feats["uuid"] = feats["h5"].map(lambda p: os.path.basename(p)[:-3].lower())

lab = pd.read_csv(LBL, dtype=str)
def norm_case(s):  # PSyy-nnnnn / PSyy.nnnnn / psyynnnnn -> PSyy-nnnn canonical
    m = re.search(r"PS\s?(\d{2})[.\-/ ]?0*(\d{3,6})", str(s).upper())
    return f"PS{m.group(1)}-{int(m.group(2))}" if m else str(s).strip().upper()
lab["case_key"] = lab["CaseName"].map(norm_case)
wx["case_key"] = wx["case_id"].map(norm_case)

m = feats.merge(wx, on="uuid", how="left").merge(
    lab[["case_key", "anon_id", "CaseName", "CollectedOrOrdered",
         "final_label", "label_status", "jury_frac"]].drop_duplicates("case_key"), on="case_key", how="left")
assert len(m) == len(feats), f"join inflated: {len(m)} vs {len(feats)}"
m.to_csv(os.path.join(OUT, "erin_master.csv"), index=False)
print(f"slides: {len(feats)} | with case link: {m['case_id'].notna().sum()} | "
      f"with label: {m['final_label'].notna().sum()} | "
      f"train-eligible: {(m['label_status'].isin(['train_eligible','adjudicated'])).sum()} | "
      f"patients: {m['anon_id'].nunique()}")
print(m["final_label"].value_counts(dropna=False).to_dict())
