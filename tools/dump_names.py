import torch, sys

pt_path = sys.argv[1]
out_path = sys.argv[2]

ckpt = torch.load(pt_path, map_location="cpu")
names = None

# YOLOv5 thường nằm ở đây
if isinstance(ckpt, dict):
    if "model" in ckpt and hasattr(ckpt["model"], "names"):
        names = ckpt["model"].names
    elif "names" in ckpt:
        names = ckpt["names"]

if names is None:
    # fallback: đôi khi ckpt['model'] là object có .names
    m = ckpt.get("model", None) if isinstance(ckpt, dict) else None
    if m is not None and hasattr(m, "names"):
        names = m.names

if names is None:
    raise RuntimeError("Không đọc được names từ pt")

# names có thể là dict {id: name} hoặc list
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

with open(out_path, "w", encoding="utf-8") as f:
    for n in names:
        f.write(str(n).strip() + "\n")

print("Saved:", out_path, "classes=", len(names))

