"""Compute the CONCH prompt-ensemble text weights once (CPU, env erin whose transformers has batch_encode_plus) and save them,
so the GPU units (env virchow2, newer transformers) never tokenise. Prompts/templates identical to task_conch_zeroshot.py."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/mnt/scratche/slow/fmlab/zuberi01/phd/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
B = "/mnt/scratche/slow/fmlab/zuberi01"; Z = B + "/analysis/paperB_extras/zeroshot"; os.makedirs(Z, exist_ok=True)
CLASSES = ["NORMAL_OTHER", "NDBE", "IND", "LGD", "HGD", "CANCER"]
PROMPTS = {"NORMAL_OTHER": ["normal squamous oesophageal mucosa", "normal gastric mucosa", "benign squamous epithelium", "unremarkable mucosa without Barrett's"],
           "NDBE": ["Barrett's oesophagus with intestinal metaplasia, negative for dysplasia", "non-dysplastic Barrett's oesophagus", "intestinal metaplasia with goblet cells and no dysplasia", "columnar lined oesophagus without dysplasia"],
           "IND": ["Barrett's oesophagus indefinite for dysplasia", "columnar mucosa with atypia indefinite for dysplasia", "reactive atypia in Barrett's oesophagus, indefinite for dysplasia"],
           "LGD": ["Barrett's oesophagus with low-grade dysplasia", "low-grade glandular dysplasia in Barrett's oesophagus", "columnar epithelium with low-grade dysplasia"],
           "HGD": ["Barrett's oesophagus with high-grade dysplasia", "high-grade glandular dysplasia", "columnar epithelium with high-grade dysplasia and marked nuclear atypia"],
           "CANCER": ["oesophageal adenocarcinoma", "invasive adenocarcinoma arising in Barrett's oesophagus", "intramucosal adenocarcinoma", "adenocarcinoma with invasion"]}
TEMPLATES = ["CLASSNAME.", "a photomicrograph showing CLASSNAME.", "an H&E image of CLASSNAME.", "histopathology image of CLASSNAME.", "CLASSNAME is shown in this image."]
model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=B + "/phd/CONCH/checkpoints/conch/pytorch_model.bin"); model = model.eval(); tok = get_tokenizer()
with torch.inference_mode():
    W = []
    for c in CLASSES:
        e = model.encode_text(tokenize(texts=[t.replace("CLASSNAME", p) for p in PROMPTS[c] for t in TEMPLATES], tokenizer=tok)).float()
        e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
    W = torch.stack(W).numpy(); scale = float(model.logit_scale.exp())
np.save(Z + "/text_weights.npy", W); open(Z + "/logit_scale.txt", "w").write(str(scale)); os.chmod(Z + "/text_weights.npy", 0o600)
print("text weights", W.shape, "logit scale", scale)
