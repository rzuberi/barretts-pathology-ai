#!/bin/bash
# BCN (docs/two_papers_BCN.md): conda env for BEACON (github.com/caner-ercan/BEACON) at phd/beacon/env. Run as a CPU Slurm job.
# BEACON's requirements.txt pins torch 2.10 + TF 2.20 + Keras 3 for all components together. Deviations: torch from the cu126 index (L40S + H200);
#   tensorflow -> tensorflow[and-cuda] + tf-keras (tensorflow_hub's KerasLayer needs Keras 2 under TF 2.16+); + openslide-python/openslide-bin and
#   opencv-python-headless (imported by wsi_core, not listed); tensorflow_addons (EOL, TF <= 2.15; only the SimCLR fine-tune's LAMB) dropped.
set -x
B=/mnt/scratche/slow/fmlab/zuberi01/phd/beacon; E=$B/env; export PIP_CACHE_DIR=/mnt/scratche/slow/fmlab/zuberi01/pip_cache
source ~/miniforge3/etc/profile.d/conda.sh
[ -d $E ] || conda create -y -p $E python=3.11
conda activate $E
pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
# Their full pin set does not resolve (fastapi==0.128.0 vs starlette==0.52.1; ctypes_snappy==1.03 is not on PyPI): it is the union of DACOR, CellViT's
# viewer/server and SegFormer. This env is for the DACOR path only (tiling, REMEDIS embedding, CLAM scoring), with their pins for those packages.
# CellViT gets its own env later from their 2.1.cell/CellViT-plus-plus/environment.yaml (or the cercan/cellvit image).
grep -E '^(albumentations|h5py|matplotlib|numpy|pandas|Pillow|PyYAML|scikit_learn|scipy|tensorflow_hub|tifffile|tqdm|transformers|timm|einops)==' $B/code/requirements.txt > $B/requirements_bcn.txt
pip install -q -r $B/requirements_bcn.txt "tensorflow[and-cuda]==2.20.0" tf-keras==2.20.1 openslide-python openslide-bin opencv-python-headless "setuptools<81" || echo PIP_FAILED  # tensorflow_hub 0.16.1 imports pkg_resources (gone in setuptools 81)
pip check | tail -5
python - <<'PY'
import os; os.environ["TF_USE_LEGACY_KERAS"] = "1"
import torch, tensorflow as tf, tensorflow_hub as hub, tf_keras, openslide, cv2, h5py, albumentations, pandas, numpy
print("ENV OK torch", torch.__version__, "cuda", torch.version.cuda, "tf", tf.__version__, "hub", hub.__version__, "tf_keras", tf_keras.__version__,
      "openslide", openslide.__library_version__, "cv2", cv2.__version__, "numpy", numpy.__version__)
print("GPU torch", torch.cuda.is_available() and torch.cuda.get_device_name(0), "| tf", tf.config.list_physical_devices("GPU"))
if torch.cuda.is_available(): print("torch matmul ok", float((torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda")).sum()))
if tf.config.list_physical_devices("GPU"): print("tf matmul ok", float(tf.reduce_sum(tf.matmul(tf.ones((64, 64)), tf.ones((64, 64))))))
PY
echo BUILD_DONE
