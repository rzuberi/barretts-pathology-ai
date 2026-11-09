"""Two papers, agent BCN (docs/two_papers_BCN.md): GPU phase. NOT RUN YET (needs weights). Reads only the uint8 tile arrays written by tp_bcn_tiles.py
(no slide pixels), embeds them with BEACON's fine-tuned REMEDIS backbone and scores them with the released DACOR checkpoint.

TASK=embed_<V>_<k>  (V = A | B; one slide per task): tiles/<V>/s<k>.npy, keep mask from coords/s<k>.npz (tp_bcn_tiles.keep) ->
    feats/<V>/pt_files/s<k>.pt (torch float32, n_keep x 1024) and feats/<V>/h5_files/s<k>.h5 (features, coords): their extract_features_fp.py layout.
    Encoder = their models/model_foundation.foundationModel: tf.keras.Sequential(hub.KerasLayer(<FOUNDATION_MODEL_ROOT>/finetuned/448_ds2/<ARCH>/saved_backbone_model)),
    input float32 0-255, 224 x 224 x 3, no normalisation (extract_features_fp.py defaults: --transforms normalisation -> no op, --normalisation none).
TASK=score_<V>: CLAM_MB(size_arg="big", n_classes=2, input_L=1024, dropout=True) + s_16_checkpoint.pt (their eval_utils.initiate_model key cleaning) on every
    slide bag -> dacor/<V>/slide_scores.csv (k, n_tiles, p_abnormal = Y_prob[0, 1]) and dacor/<V>/attention/s<k>.npy. No outcome is read here.
Env: phd/beacon/env (TF 2.20 + tf-keras, torch 2.10 cu126). TF_USE_LEGACY_KERAS=1 so tensorflow_hub's KerasLayer loads under Keras 2 semantics.
Paths: FOUNDATION_MODEL_ROOT (REMEDIS; fine-tuned backbone under finetuned/448_ds2/), DACOR_CKPT (Zenodo models/dacor_mil/s_16_checkpoint.pt), ARCH."""
import os, sys, glob, numpy as np
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from tp_bcn_tiles import OD, MIL, keep
FO = os.environ.get("BCN_OUT", OD)   # feats/ and dacor/ root (a scratch dir for the dummy-weight plumbing check)
TASK = os.environ["TASK"]; ARCH = os.environ.get("ARCH", "path-152x2-remedis-m")  # preprint Methods; their fine-tune notebook says path-50x1-remedis-s

if TASK.startswith("embed_"):
    _, V, k = TASK.split("_"); k = int(k); FD = f"{FO}/feats/{V}"; out_pt = f"{FD}/pt_files/s{k:03d}.pt"
    if os.path.exists(out_pt): raise SystemExit(0)
    import tensorflow as tf, tensorflow_hub as hub, torch, h5py
    for g in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(g, True)
    root = os.environ["FOUNDATION_MODEL_ROOT"]; hub_path = os.path.join(root, "finetuned/448_ds2", ARCH, "saved_backbone_model")
    model = tf.keras.Sequential(hub.KerasLayer(hub_path, trainable=False)); model.build(input_shape=(None, 224, 224, 3))
    T = np.load(f"{OD}/tiles/{V}/s{k:03d}.npy", mmap_mode="r"); z = np.load(f"{OD}/coords/s{k:03d}.npz"); m = keep(z[f"stats{V}"]); idx = np.flatnonzero(m)
    F = np.concatenate([model(tf.convert_to_tensor(np.asarray(T[idx[i:i + 256]]), tf.float32), training=False).numpy() for i in range(0, len(idx), 256)]) if len(idx) else np.zeros((0, 1024), np.float32)
    assert F.ndim == 2 and F.shape[1] == 1024, F.shape   # DACOR's first layer is 1024 -> 512 (Supplementary Table 2)
    for d in ("pt_files", "h5_files"): os.makedirs(f"{FD}/{d}", exist_ok=True)
    with h5py.File(f"{FD}/h5_files/s{k:03d}.h5", "w") as h: h["features"] = F; h["coords"] = z[V][idx]
    torch.save(torch.from_numpy(F.astype(np.float32)), out_pt + ".tmp"); os.replace(out_pt + ".tmp", out_pt); print("TPBCN EMBED", V, k, F.shape, flush=True)

elif TASK.startswith("score_"):
    V = TASK.split("_")[1]; import torch, pandas as pd; sys.path.insert(0, MIL); from models.model_clam import CLAM_MB
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLAM_MB(size_arg="big", n_classes=2, input_L=1024, dropout=True)   # their eval: --model_type clam_mb --model_size big --drop_out
    ck = torch.load(os.environ["DACOR_CKPT"], map_location=dev); model.load_state_dict({k_.replace(".module", ""): v for k_, v in ck.items() if "instance_loss_fn" not in k_}, strict=True)
    model.relocate(); model.eval(); AD = f"{FO}/dacor/{V}/attention"; os.makedirs(AD, exist_ok=True); rows = []
    for f in sorted(glob.glob(f"{FO}/feats/{V}/pt_files/s*.pt")):
        k = int(os.path.basename(f)[1:4]); x = torch.load(f).to(dev)
        if len(x) == 0: rows.append({"k": k, "n_tiles": 0, "p_abnormal": np.nan}); continue
        with torch.no_grad(): logits, Y_prob, Y_hat, A, _ = model(x)
        rows.append({"k": k, "n_tiles": len(x), "p_abnormal": float(Y_prob[0, 1])}); np.save(f"{AD}/s{k:03d}.npy", A.cpu().numpy())
    pd.DataFrame(rows).to_csv(f"{FO}/dacor/{V}/slide_scores.csv", index=False); print("TPBCN SCORE", V, len(rows), flush=True)
