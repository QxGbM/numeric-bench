import torch
import torchvision.transforms as transforms
import torchvision.models as models
import tarfile
import io
from PIL import Image
import numpy as np
from tqdm import tqdm
import pdb
import os

TAR_PATH = "ILSVRC2012_img_train.tar"
OUTPUT_NPY = "imagenet1k_train_resnet50_features_2048d.npy"
BATCH_SIZE = 128

# Standard ImageNet preprocessor (exactly what ResNet50 expects)
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

torch.set_num_threads(8)
device = torch.device("cpu") #device=torch.device("cuda")
print(f"Using device: {device}")

total_images = 1281167
features = np.empty((total_images, 2048), dtype=np.float32)
print(f"Memory allocated: {features.nbytes / 1024**3:.2f} GB")

# --------------------------------------------------------------------
# Load ResNet50 (pretrained on ImageNet) and extract features
# --------------------------------------------------------------------
print("Loading ResNet50 and extracting 2048-d features...")
# Official pretrained weights (same as the standard ImageNet preprocessor)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Remove the final classification layer → 2048-d avgpool features
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

idx = 0
check_flag = True
batch_tensors = []
pbar = tqdm(total=total_images, desc="Extracting features", unit="img", position=0, leave=True)

with tarfile.open(TAR_PATH, mode="r") as outer_tar:
  for outer_member in outer_tar:
    if not (outer_member.isfile() and outer_member.name.endswith(".tar")):
      continue

    inner_file = outer_tar.extractfile(outer_member)
    if inner_file is None:
      continue

    # Streaming inner tar
    with tarfile.open(fileobj=inner_file, mode="r|") as inner_tar:
      for inner_member in inner_tar:
        if not (inner_member.isfile() and inner_member.name.lower().endswith((".jpg", ".jpeg"))):
          continue

        # ---------- Image loading (1-by-1) ----------
        img_bytes = inner_tar.extractfile(inner_member).read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess (still on CPU)
        tensor = preprocess(img)                 # shape (3, 224, 224)
        batch_tensors.append(tensor)

        # ---------- Batch inference when ready ----------
        if len(batch_tensors) == BATCH_SIZE:
          batch_tensor = torch.stack(batch_tensors).to(device)

          with torch.no_grad():
            batch_feat = model(batch_tensor)          # (B, 2048, 1, 1)

          # Flatten to (B, 2048) and store directly in the big array
          batch_feat_np = batch_feat.view(batch_feat.size(0), -1).cpu().numpy()
          features[idx:idx + BATCH_SIZE] = batch_feat_np

          idx += BATCH_SIZE
          batch_tensors = []
          pbar.update(BATCH_SIZE)

# Don't forget the last partial batch
if batch_tensors:
  batch_tensor = torch.stack(batch_tensors).to(device)
  with torch.no_grad():
    batch_feat = model(batch_tensor)
  batch_feat_np = batch_feat.view(batch_feat.size(0), -1).cpu().numpy()
  features[idx:idx + len(batch_tensors)] = batch_feat_np
  idx += len(batch_tensors)
  pbar.update(len(batch_tensors))

pbar.close()

# --------------------------------------------------------------------
# Final checks and save
# --------------------------------------------------------------------
assert idx == total_images, f"Index mismatch: {idx} != {total_images}"
print("\nFeature extraction completed successfully!")
print(f"Final matrix shape: {features.shape} (FP32)")
print(f"Memory used by features: {features.nbytes / 1024**3:.2f} GB")

np.save(OUTPUT_NPY, features)
print(f"Saved to: {OUTPUT_NPY}")
