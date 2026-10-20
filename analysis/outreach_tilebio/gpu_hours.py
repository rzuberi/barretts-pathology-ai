"""GPU-hours actually allocated to this study (sacct; elapsed x GPUs, every GPU job incl. smoke tests, duplicates that started and
exited in seconds included). Writes results/gpu_hours.json."""
import json, re, subprocess, collections
A = "/mnt/scratche/slow/fmlab/zuberi01/analysis/outreach_tilebio"
PFX = ("tesmk1__", "temb__", "temb1h__", "temb1hb__", "temb_c16r__", "ttrain_tcga__", "ttrain_c16__", "tt_pipesmoke", "tt_chk")
out = subprocess.run(["sacct", "-u", "zuberi01", "-S", "2026-10-19T15:00", "-X", "-n", "-P", "-o", "JobName,Partition,ElapsedRaw,AllocTRES,State"], capture_output=True, text=True).stdout
by = collections.defaultdict(float); n = 0
for line in out.splitlines():
    name, part, el, tres, st = line.split("|")
    if not name.startswith(PFX): continue
    g = re.search(r"gres/gpu(?::[^=]+)?=(\d+)", tres); g = int(g.group(1)) if g else 0
    if g: by[f"{part}"] += g * int(el or 0) / 3600; n += 1
r = {"by_partition_gpu_hours": {k: round(v, 2) for k, v in by.items()}, "total_gpu_hours": round(sum(by.values()), 2), "n_gpu_jobs": n}
json.dump(r, open(f"{A}/results/gpu_hours.json", "w"), indent=1); print(r)
