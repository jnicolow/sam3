import fnmatch
from sam3.model_builder import build_sam3_image_model

FREEZE_PATTERNS = [
    "backbone.language_backbone.*",   # "ocean" — keep frozen
    "backbone.vision_backbone.*",     # ViT + neck — freeze for head-only
]

def build_sam3_image_model_frozen(**kwargs):
    model = build_sam3_image_model(**kwargs)
    for name, p in model.named_parameters():
        if any(fnmatch.fnmatch(name, pat) for pat in FREEZE_PATTERNS):
            p.requires_grad = False
    return model