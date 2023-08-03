import pickle
from functools import lru_cache
import numpy as np
import warnings

import pandas as pd
import zstandard as zstd
from algo.codec import NonLearningCompressionAlgorithm, LossyCompressionAlgorithm


class KMeansLinear:

    def __init__(self, x: np.ndarray, max_k: int, max_rmse: np.longdouble):

        super().__init__()
        self.x = x
        if np.isnan(x).any():
            raise ValueError('NaNs in arrays are not supported')
        if x.size == 0:
            raise ValueError('Empty array')
        if not x.ndim == 1:
            raise ValueError('Only 1-dim arrays are supported')
        if not self.is_sorted():
            warnings.warn('source array is unsorted, consider sorting it or we will do it ourselves')
            index = x.argsort()
            self.reverse_index = index.argsort()
            self.x = x[index]
        else:
            self.reverse_index = np.arange(x.size)

        self.max_k = max_k
        self.max_rmse = max_rmse
        self.nUnique = self.n_of_unique()
        k = min(self.nUnique, self.max_k)
        if self.nUnique > 1:
            self.S = np.zeros(shape=(k, x.size), dtype=np.longdouble)
            self.J = np.zeros(shape=(k, x.size), dtype=int)

        self.median = self.x[self.x.size // 2]

        self.sum_x = np.cumsum(self.x - self.median)
        self.sum_x_sq = np.cumsum((self.x - self.median) ** 2)

    def n_of_unique(self):
        return self.x.shape[0] - (self.x[:-1] == self.x[1:]).sum()

    def is_sorted(self):
        return np.all(self.x[:-1] <= self.x[1:])

    def ssq(self, j: int, i: int) -> float:
        # print("i, j", i, j)
        if j >= i:
            return 0
        elif j > 0:
            sji = self.sum_x_sq[i] - self.sum_x_sq[j - 1] - (self.sum_x[i] - self.sum_x[j - 1]) / (i - j + 1) * (
                    self.sum_x[i] - self.sum_x[j - 1])  # some magnitude shenanigans to not overflow?
            # print("sji", sji)
            return max(0.0, sji)
        else:
            sji = self.sum_x_sq[i] - self.sum_x[i] * self.sum_x[i] / (i + 1)
            # print("sji", sji)
            return max(0.0, sji)

    def find_min_from_candidates(self, imin: int, imax: int, istep: int, q: int, js: np.ndarray):
        rmin_prev = 0
        for i in range(imin, imax + 1, istep):
            rmin = rmin_prev
            self.S[q, i] = self.S[q - 1, js[rmin] - 1] + self.ssq(js[rmin], i)
            self.J[q, i] = js[rmin]
            for r in range(rmin + 1, js.size):
                j_abs = js[r]
                if j_abs < self.J[q - 1, i]: continue
                if j_abs > i: break

                Sj = self.S[q - 1, j_abs - 1] + self.ssq(j_abs, i)
                if Sj <= self.S[q, i]:
                    self.S[q, i] = Sj
                    self.J[q, i] = js[r]
                    rmin_prev = r

    def reduce_in_place(self, imin: int, imax: int, istep: int, q: int, js: np.ndarray):
        N = ((imax - imin) // istep) + 1
        js_red = js.copy()
        if N >= js.size: return js_red

        # OG: two positions to move candidate j's back and forth
        left = -1  # OG: points to last favorable position / column
        right = 0  # OG: points to current position / column

        m = js_red.size

        while m > N:  # OG: js_reduced has more than N positions / columns

            p = left + 1

            i = imin + p * istep
            j = js_red[right]
            Sl = self.S[q - 1, j - 1] + self.ssq(j, i)

            jplus1 = js_red[right + 1]
            Slplus1 = self.S[q - 1, jplus1 - 1] + self.ssq(jplus1, i)

            if Sl < Slplus1 and p < N - 1:
                left += 1
                js_red[left] = j
                right += 1  # OG: move on to next position / column p + 1
            elif Sl < Slplus1 and p == N - 1:
                right += 1
                js_red[right] = j  # OG: delete position / column p + 1
                m -= 1
            else:  # OG: Sl >= Slplus1
                if p > 0:  # OG: i > imin
                    # OG: delete position / column p and
                    # OG:   move back to previous position / column p-1:
                    js_red[right] = js_red[left]
                    left -= 1
                else:
                    right += 1
                m -= 1
        for r in range(left + 1, m):
            js_red[r] = js_red[right]
            right += 1
        # print("js_red")
        # print(js_red)
        js_red = np.resize(js_red, m)
        return js_red

    def fill_even_positions(self, imin: int, imax: int, istep: int, q: int, js: np.ndarray):
        # print("fill_even_positions")
        # OG: Derive j for even rows (0-based)
        n = js.size
        istepx2 = istep << 1  # feelin' fancy? bitshift by one to the left like in OG
        jl = js[0]  # aka jmin?
        r = 0
        for i in range(imin, imax + 1, istepx2):
            # r = np.argmax(js >= jl)
            while js[r] < jl:  # TODO: this is probably not ok, but numpy.argwhere might actually be worse in some cases
                # OG: Increase r until it points to a value of at least jmin
                r += 1
            # OG: Initialize S[q][i] and J[q][i]
            self.S[q, i] = self.S[q - 1, js[r] - 1] + self.ssq(js[r], i)
            self.J[q, i] = js[r]  # OG: rmin

            # OG: Look for minimum S upto jmax within js
            if i + istep <= imax:
                jh = self.J[q, i + istep]
            else:
                jh = js[n - 1]

            jmax = min(jh, i)

            sjimin = self.ssq(jmax, i)  # weird init in og

            r += 1
            while r < n and js[r] <= jmax:
                jabs = js[r]

                if jabs > i:
                    r += 1
                    break

                if jabs < self.J[q - 1, i]:
                    r += 1
                    continue

                s = self.ssq(jabs, i)
                Sj = self.S[q - 1, jabs - 1] + s

                if Sj <= self.S[q, i]:
                    self.S[q, i] = Sj
                    self.J[q, i] = js[r]
                elif self.S[q - 1, jabs - 1] + sjimin > self.S[q, i]:
                    r += 1
                    break
                r += 1
            r -= 1
            jl = jh

    def SMAWK(self, imin: int, imax: int, istep: int, q: int, js: np.ndarray):
        if imax - imin <= 0:
            self.find_min_from_candidates(imin, imax, istep, q, js)
        else:  # OG: REDUCE
            js_odd = self.reduce_in_place(imin, imax, istep, q, js)
            istepx2 = istep * 2
            imin_odd = imin + istep
            imax_odd = imin_odd + ((imax - imin_odd) // istepx2) * istepx2
            self.SMAWK(imin_odd, imax_odd, istepx2, q, js_odd)
            self.fill_even_positions(imin, imax, istep, q, js)

    def fill_row_q(self, imin: int, imax: int, q: int):
        js = np.arange(q, imax + 1)
        self.SMAWK(imin, imax, 1, q, js)  # SMAWK() is recursive and js is modified inside

    def fill_dp_matrix(self) -> int:

        """
        (OG docs)
        :param x: One dimension vector to be clustered, must be sorted (in any order)
        :param S: K x N matrix. S[q][i] is the sum of squares of the distance from each x[i] to its cluster mean when there are exactly x[i] is the last point in cluster q
        :param J: K x N backtrack matrix
        NOTE: All vector indices in this program start at position 0
        """

        K = self.S.shape[0]
        N = self.S.shape[1]

        max_se = (self.max_rmse ** 2) * N

        for i in range(self.S.shape[1]):
            self.S[0, i] = self.ssq(0, i)
        self.J[0] = np.zeros_like(self.J[0])

        for q in range(1, K):
            if q < K - 1:
                imin = max(1, q)
            else:
                imin = N - 1  # OG: No need to compute S[K-1][0] ... S[K-1][N-2]

            self.fill_row_q(imin, N - 1, q)
            print("filled", q)
            if max_se >= self.S[q - 1, -1]:  # early exit for when we found out q clusters is enough
                print(q, "is good enough, se =", self.S[q - 1, -1])
                self.S = self.S[:q]
                self.J = self.J[:q]
                return q
            # print("S: ",S)
            # print("J: ",J)
        return K

    def backtrack(self):
        K = self.J.shape[0]
        N = self.J.shape[1]
        # print("backtrack")
        cluster_right = N - 1
        cluster = np.empty(N, dtype=int)
        centers = np.empty(K)
        count = np.empty(K)
        for q in range(K - 1, -1, -1):
            cluster_left = self.J[q, cluster_right]
            for i in range(cluster_left, cluster_right + 1):
                # print(i)
                cluster[i] = q
            summ = self.x[cluster_left:cluster_right + 1].sum()
            centers[q] = summ / (cluster_right - cluster_left + 1)
            count[q] = cluster_right - cluster_left + 1
            if q > 0:
                cluster_right = cluster_left - 1
        return {"cluster": cluster[self.reverse_index], "centers": centers, "k_res": K}

    def run(self) -> dict:

        if self.nUnique > 1:
            self.fill_dp_matrix()
            result = self.backtrack()
            # result["centers"] += self.median
        else:
            cluster = np.zeros(self.x.size)
            centers = self.x[0]
            result = {"cluster": cluster, "centers": centers, "k_res": 1}
        return result


class KMeansLinearDTO:
    pass


class KMeansLinearCompression(NonLearningCompressionAlgorithm, LossyCompressionAlgorithm):
    name = "CKMeans.1d.dp"

    def __init__(self, max_rmse, raw_compression_ratio=1):
        super().__init__()
        self.raw_compression_ratio = raw_compression_ratio
        self.max_rmse = max_rmse

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            df = pickle.load(table_file)
            numeric = df.select_dtypes(['number']).dropna()
            non_numeric = df.select_dtypes(exclude=['number'])
            order = df.columns
            order_numeric = numeric.columns
            index = df.index
            quantized_columns = []
            print(numeric)
            for column in numeric.to_numpy().T:
                kmeans = KMeansLinear(column, np.ceil(column.size / self.raw_compression_ratio), self.max_rmse)

                result = kmeans.run()
                del kmeans.S
                del kmeans.J
                del kmeans
                quantized_columns.append(result)
                del result
            output = KMeansLinearDTO()
            output.non_numeric = non_numeric
            output.order = order
            output.order_numeric = order_numeric
            output.index = index
            output.quantized_columns = quantized_columns
            file = pickle.dumps(output)
            compressor = zstd.ZstdCompressor(level=22)
            compressed_data = compressor.compress(file)
            compressed_file.write(compressed_data)
            compressed_file.flush()

    def decompress(self, compressed_file_path, decompressed_file_path):
        with open(compressed_file_path, 'rb') as compressed_file, open(decompressed_file_path,
                                                                       'wb') as decompressed_file:
            decompressor = zstd.ZstdDecompressor()
            decompressed = decompressor.decompress(compressed_file.read())
            from_file = pickle.loads(decompressed)
            quantized_columns = from_file.quantized_columns

            restored_numeric_df = pd.DataFrame(data=restored_numeric, index=from_file.index,
                                               columns=from_file.order_numeric)
            restored = pd.merge(from_file.non_numeric, restored_numeric_df, left_index=True, right_index=True)[
                from_file.order]
            pickle.dump(restored, decompressed_file)
