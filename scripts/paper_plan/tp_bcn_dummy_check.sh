#!/bin/bash
# BCN: plumbing check of tp_bcn_embed.py with DUMMY weights (random TF encoder with the same interface: 224x224x3 float 0-255 -> 1024; random CLAM_MB
# state dict). Proves tile reading, keep mask, output layout and CLAM loading/forward; says nothing about BEACON's numbers. Output in a scratch dir only.
set -e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate /mnt/scratche/slow/fmlab/zuberi01/phd/beacon/env
T=/mnt/scratche/slow/fmlab/zuberi01/phd/thesis; D=/mnt/scratche/slow/fmlab/zuberi01/phd/beacon/dummy_check; mkdir -p $D/finetuned/448_ds2/dummy
export TF_USE_LEGACY_KERAS=1 FOUNDATION_MODEL_ROOT=$D ARCH=dummy DACOR_CKPT=$D/dummy_ckpt.pt BCN_OUT=$D/out
python - <<'PY'
import os, tensorflow as tf, tf_keras as K, torch, sys
D = os.environ["FOUNDATION_MODEL_ROOT"]; tf.random.set_seed(0)
m = K.Sequential([K.layers.Rescaling(1 / 255.), K.layers.Conv2D(8, 7, strides=4), K.layers.GlobalAveragePooling2D(), K.layers.Dense(1024, use_bias=False)]); m.build((None, 224, 224, 3))
tf.saved_model.save(m, D + "/finetuned/448_ds2/dummy/saved_backbone_model")
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/beacon/code/1.aneuploid/mil"); from models.model_clam import CLAM_MB
torch.manual_seed(0); torch.save(CLAM_MB(size_arg="big", n_classes=2, input_L=1024, dropout=True).state_dict(), os.environ["DACOR_CKPT"]); print("DUMMY WEIGHTS OK")
PY
cd $T; TASK=embed_A_600 python scripts/paper_plan/tp_bcn_embed.py; TASK=embed_B_600 python scripts/paper_plan/tp_bcn_embed.py
TASK=score_A python scripts/paper_plan/tp_bcn_embed.py; cat $BCN_OUT/dacor/A/slide_scores.csv
python -c "import numpy as np, sys; sys.path.insert(0,'scripts/paper_plan'); from tp_bcn_tiles import OD, keep; z=np.load(OD+'/coords/s600.npz'); print('keep A', int(keep(z['statsA']).sum()), 'of', len(z['A']))"
