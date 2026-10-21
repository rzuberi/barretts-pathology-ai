from huggingface_hub import model_info
for m in ["thomas-sounack/BioClinical-ModernBERT-base","thomas-sounack/BioClinical-ModernBERT-large","emilyalsentzer/Bio_ClinicalBERT","microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext","answerdotai/ModernBERT-base"]:
    try: print(m, "OK", [s.rfilename for s in model_info(m).siblings][:10])
    except Exception as e: print(m, "FAIL", str(e)[:100])
