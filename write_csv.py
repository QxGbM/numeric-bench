import sys
import numpy as np

def main() -> int:
  M = 1281167
  N = 2048
  A = np.load("/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.npy")

  print(A.shape)
  means = A.mean(axis=0, keepdims=True)
  A = A - means
  print(means)

  csv_file = "/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv"
  np.savetxt(csv_file, A, delimiter=',', newline='\n', fmt="%.8e")

  row_bytes = [0]
  length = 0
  idx_cache = (csv_file).removesuffix(".csv") + ".cache"

  with open(csv_file, 'rb') as f:
    for line in f:
      length += len(line)
      row_bytes.append(length)

  with open(idx_cache, 'w') as out:
    for offset in row_bytes:
      out.write(f"{offset}\n")

  return 0

if __name__ == "__main__":
  sys.exit(main())
