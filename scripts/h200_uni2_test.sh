#!/usr/bin/env bash
# H200 compatibility test: UNI2-h forward pass in the cu128 `virchow2` env.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate virchow2
python - <<'PYEOF'
import torch, timm, time
print(torch.__version__, torch.cuda.get_device_name(0))
m = timm.create_model("vit_giant_patch14_224", pretrained=False, img_size=224,
                      patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                      embed_dim=1536, mlp_ratio=5.33334, num_classes=0,
                      no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                      act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
sd = torch.load("/mnt/scratche/slow/fmlab/zuberi01/phd/UNI2-h/pytorch_model.bin",
                map_location="cpu")
missing, unexpected = m.load_state_dict(sd, strict=False)
m.eval().cuda()
x = torch.randn(64, 3, 224, 224, device="cuda")
t0 = time.time()
with torch.inference_mode():
    o = m(x)
torch.cuda.synchronize()
print("H200 UNI2 OK", tuple(o.shape), round(time.time() - t0, 2), "s",
      "| missing", len(missing), "unexpected", len(unexpected))
PYEOF
