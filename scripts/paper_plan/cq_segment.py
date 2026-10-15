"""CNV-quiet morphology (docs/paper_cnvquiet_morphology.md), phase 2 (GPU): HoVer-Net (fast mode, PanNuke weights, TIAToolbox 2.1.3
`hovernet_fast-pannuke`) on the saved 0.25 um/px tiles of cq_extract.py. Never reads slides. Per tile: reflect-pad by 46 px (+ to a multiple of 164),
256-px input / 164-px output sliding window with stride 164, maps assembled and cropped back to the tile, then TIAToolbox's own HoVer-Net
post-processing (HoVerNet._proc_np_hv watershed + get_instance_info) in a CPU pool. Per nucleus: tile, centroid, area (instance pixel count),
perimeter (closed contour length, cv2.arcLength), type, type probability, edge flag (bounding box within 2 px of the tile edge).
usage: python cq_segment.py MANIFEST.csv SHARD NSHARDS [--overlay N]
Row-level output (cluster only): cnvquiet_morphology/nuclei/<slide_key>.npz (+ overlays/<slide_key>_tNNN.png when --overlay)."""
import os, sys, json, time, numpy as np, pandas as pd, cv2, torch
from multiprocessing import get_context
from PIL import Image
W = "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm/cnvquiet_morphology"
IN, OUT, M = 256, 164, 46; BATCH = 64; TD = os.environ.get("CQ_TILEDIR", "tiles"); ND = os.environ.get("CQ_NUCDIR", "nuclei")
def postproc(args):
    k, npm, hvm, tpm = args
    from tiatoolbox.models.architecture.hovernet import HoVerNet
    tpm = np.around(tpm[..., 0]).astype(np.uint8); inst = HoVerNet._proc_np_hv(npm, hvm); info = HoVerNet.get_instance_info(inst, tpm, verbose=False); H, Wd = inst.shape
    cnt = np.bincount(inst.ravel()); rows = []
    for iid, v in info.items():
        x0, y0, x1, y1 = v["box"]; c = np.asarray(v["contours"], dtype=np.int32).reshape(-1, 1, 2)
        rows.append((k, v["centroid"][0], v["centroid"][1], cnt[iid] if iid < len(cnt) else 0, cv2.arcLength(c, True), int(v["type"] or 0), float(v["prob"] or 0),
                     x0 <= 2 or y0 <= 2 or x1 >= Wd - 3 or y1 >= H - 3, iid))
    return k, rows, inst
def main():
    man, shard, nsh = pd.read_csv(sys.argv[1], dtype=str), int(sys.argv[2]), int(sys.argv[3]); n_ov = int(sys.argv[sys.argv.index("--overlay") + 1]) if "--overlay" in sys.argv else 0
    from tiatoolbox.models.architecture import get_pretrained_model
    from tiatoolbox.models.architecture.hovernet import HoVerNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, torch.__version__, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    model, _ = get_pretrained_model("hovernet_fast-pannuke"); model = model.to(dev).eval()
    os.makedirs(f"{W}/{ND}", exist_ok=True); os.makedirs(f"{W}/overlays", exist_ok=True)
    pool = get_context("spawn").Pool(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1))
    for r in man.iloc[shard::nsh].itertuples():
        out = f"{W}/{ND}/{r.slide_key}.npz"
        if os.path.exists(out): print("skip", r.slide_key, flush=True); continue
        meta = json.load(open(f"{W}/{TD}/{r.slide_key}/meta.json")); nt = meta["n_tiles"]; t0 = time.time(); jobs = []
        for k in range(nt):
            img = np.asarray(Image.open(f"{W}/{TD}/{r.slide_key}/t{k:03d}.jpg").convert("RGB")); h, w = img.shape[:2]
            nh, nw = -(-h // OUT) * OUT, -(-w // OUT) * OUT
            pad = np.pad(img, ((M, nh - h + M), (M, nw - w + M), (0, 0)), mode="reflect")
            ys, xs = range(0, nh, OUT), range(0, nw, OUT); patches = np.stack([pad[y:y + IN, x:x + IN] for y in ys for x in xs])
            outs = [HoVerNet.infer_batch(model, torch.from_numpy(patches[i:i + BATCH]), device=dev) for i in range(0, len(patches), BATCH)]
            npm, hvm, tpm = [np.concatenate([o[j] for o in outs]) for j in range(3)]
            def assemble(a):
                full = np.zeros((nh, nw, a.shape[-1]), np.float32); i = 0
                for y in ys:
                    for x in xs: full[y:y + OUT, x:x + OUT] = a[i]; i += 1
                return full[:h, :w]
            jobs.append(pool.apply_async(postproc, ((k, assemble(npm), assemble(hvm), assemble(tpm)),)))
        res = sorted([j.get() for j in jobs], key=lambda t: t[0]); rows = [x for _, rr, _ in res for x in rr]
        a = np.array(rows, dtype=object) if rows else np.zeros((0, 9), dtype=object)
        np.savez_compressed(out + ".tmp.npz", tile=a[:, 0].astype(np.int16), x=a[:, 1].astype(np.float32), y=a[:, 2].astype(np.float32), area_px=a[:, 3].astype(np.float32),
                            perim_px=a[:, 4].astype(np.float32), type=a[:, 5].astype(np.int8), prob=a[:, 6].astype(np.float16), edge=a[:, 7].astype(bool), n_tiles=nt, tile_px=meta["out_px"])
        os.replace(out + ".tmp.npz", out)
        for k in range(min(n_ov, nt)):  # overlay: contours coloured by class (neo red, inflam green, conn blue, dead yellow, epi cyan)
            img = np.asarray(Image.open(f"{W}/{TD}/{r.slide_key}/t{k:03d}.jpg").convert("RGB")).copy(); inst = res[k][2]; col = {1: (255, 0, 0), 2: (0, 200, 0), 3: (0, 0, 255), 4: (255, 255, 0), 5: (0, 255, 255)}
            for (kk, cx, cy, ar, pe, ty, pr, ed, iid) in res[k][1]:
                m = (inst == iid).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, col.get(ty, (0, 0, 0)), 1)
            Image.fromarray(np.concatenate([np.asarray(Image.open(f"{W}/{TD}/{r.slide_key}/t{k:03d}.jpg").convert("RGB")), img], 1)).save(f"{W}/overlays/{r.slide_key}_t{k:03d}.png")
        ty = a[:, 5].astype(int) if len(a) else np.zeros(0, int)
        print("done", r.slide_key, "tiles", nt, "nuclei", len(a), "by type", np.bincount(ty, minlength=6).tolist(), f"{time.time() - t0:.1f}s", flush=True)
    pool.close(); print("CQ SEGMENT DONE", shard, nsh)
if __name__ == "__main__": main()
