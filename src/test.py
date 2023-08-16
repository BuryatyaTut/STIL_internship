from algo.codec.pca_to_kmeans_1d_c import PCAToKMeansCompression
from algo.codec.kmeans_1d_c import KMeansLinearCCompression
from algo.process.normalize import NormalizeProcessing
from scaffold import Scaffold
import numpy as np
np.seterr(all='warn')
benchmarks = []
#np.seterr(all='raise') # something breaks on the large dataset
#datasets = [f'../data/sample_dataset_{cutoff}.csv' for cutoff in cutoff_list]
Algos = [KMeansLinearCCompression(max_rmse=0.1, threads=8)]
Processes = [NormalizeProcessing()]
for algo in Algos:
    for proc in Processes:
        a = Scaffold(algo, proc, "../data/sample_dataset_big.csv")
        a.start("test", logs=True)
        print(a.benchmark)
        benchmarks.append(a.benchmark)
        print("---------------------")