import sys
import numpy as np
from numpy.linalg import svd

def main() -> int:
  M = 1281167
  N = 2048
  A = np.load("/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.npy")

  print(A.shape)
  means = A.mean(axis=0, keepdims=True)
  A = A - means
  print(means)

  _, S, _ = svd(A, full_matrices=False)
  print(S)
  np.savetxt("singular_values.csv", S)

  return 0

if __name__ == "__main__":
  sys.exit(main())
