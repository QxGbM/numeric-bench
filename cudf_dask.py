import os
import multiprocessing as mp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import dask.array as da
import dask.dataframe as dd
import numpy as np
import cupy as cp
from cuml.dask.decomposition import TruncatedSVD
import time
import logging
import argparse

if __name__ == '__main__':
  mp.freeze_support()

  parser = argparse.ArgumentParser("cuML")
  parser.add_argument("components", help="Number components from PCA", type=int)
  args = parser.parse_args()
  print(args.components)

  cluster = LocalCUDACluster(silence_logs=logging.CRITICAL)
  client = Client(cluster)

  num_gpus = len(client.scheduler_info()['workers'])
  print(f"Dask LocalCUDACluster is using {num_gpus} devices (GPUs).")

  df = dd.read_csv("/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv", header=None, dtype='float32', blocksize="128MB")
  X = df.to_dask_array(lengths=True)
  X_cudf = X.map_blocks(cp.asarray, dtype=cp.float32)
  X_cudf = X_cudf.persist()
  wait(X_cudf)
  print(X_cudf.shape)

  cumlModel = TruncatedSVD(n_components=args.components)

  t0 = time.perf_counter_ns()
  XT = cumlModel.fit_transform(X_cudf)
  hXT = XT.persist()
  wait(hXT)
  t1 = time.perf_counter_ns()

  print(hXT)
  duration = 1.e-6 * (t1 - t0)
  print("Host wall-time: " + str(duration) + " ms.")

  evr = cumlModel.explained_variance_ratio_
  evr_sum = evr.sum()

  print("sum evr:")
  print(float(evr_sum))

  s_cp = cumlModel.singular_values_
  s_np = cp.asnumpy(s_cp)
  np.savetxt("cudf_sv.csv", s_np)

  client.close()
  cluster.close()
