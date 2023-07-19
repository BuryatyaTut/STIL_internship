import pickle
import zstandard as zstd
import pandas as pd
import sklearn
import sklearn.decomposition
import numpy as np
from algo.codec import LossyCompressionAlgorithm, NonLearningCompressionAlgorithm


def vecT(vec):
    return np.array(vec)[np.newaxis].T


class BPCA:
    def __init__(self, y, components):
        self.rows = y.shape[0]
        self.cols = y.shape[1]
        self.comps = components
        self.y_orig = y.copy()
        self.yest = y.copy()

        self.nans = np.isnan(y)  # boolean matrix
        self.row_nomiss = np.argwhere(~np.isnan(y).any(axis=1))[:, 0]
        self.row_miss = np.argwhere(np.isnan(y).any(axis=1))[:, 0]
        self.yest[self.row_miss, :] = 0

        self.scores = None
        self.covy = np.cov(self.yest, rowvar=False)
        # print("covy: ", self.covy)
        (U, S_arr, V) = np.linalg.svd(self.covy)  # TODO: replace with sklearn : truncatedSVD()
        S = np.diag(S_arr[:components])
        print("covy: ", self.covy)
        U = U[:, :components]

        self.mean = np.nanmean(y, axis=0)

        # print("S: ", S)
        self.PA = U @ np.sqrt(S)  # aka W
        self.tau = 1 / (np.trace(self.covy) - np.sum(S_arr))

        taumax = 1e10
        taumin = 1e-10

        self.tau = np.clip(self.tau, taumin, taumax)

        self.galpha0 = 1e-10
        self.balpha0 = 1
        self.alpha = (2 * self.galpha0 + self.cols) / (
                    self.tau * np.diag(self.PA.T @ self.PA) + 2 * self.galpha0 / self.balpha0)
        print("alpha thing: ", np.diag(self.PA.T @ self.PA))
        self.gmu0 = 0.001

        self.btau0 = 1
        self.gtau0 = 1e-10
        self.SigW = np.eye(components)
        self.x = 0

    def doStep(self):
        # fun fact: the data are not normalized besides being "zero mean"-centered, as it seems
        self.scores = np.full((self.rows, self.comps), fill_value=np.nan)
        Rx = np.eye(self.comps) + self.tau * self.PA.T @ self.PA + self.SigW
        # print("PA: ",self.PA)
        Rxinv = np.linalg.inv(Rx)  # aka Σ_x
        idx = self.row_nomiss
        if idx.size == 0:
            trS = 0
            T = 0
        else:
            dy = self.y_orig[idx, :] - np.tile(self.mean,
                                               (idx.size, 1))  # (t_n - μ)^T in CM Bishop? aka mean-shifted data

            x = self.tau * Rxinv @ self.PA.T @ dy.T  # m_x in CM Bishop ? probably the scores
            # yes that's what they are
            # print(self.tau)
            T = dy.T @ x.T  # what is T?..
            trS = np.sum(dy * dy)  # is (t_n - μ) * (t_n - μ) called S here?
            # yes it is, S is the covariance matrix
            # why the scalar multiplication? sum of squares? why call it trace then

            xTranspose = vecT(x)
            # print("xTranspose: ",xTranspose)
            # print("idx: ",idx)
            for _id, row_observed in enumerate(idx):
                # print("scores i: ",self.scores[i, :])
                # print("xTranspose _id: ",xTranspose[_id, :])
                self.scores[row_observed, :] = xTranspose[_id,
                                               :].T  # set scores of complete data to the analytical pca results
            # print("scores: ", self.scores)

        if self.row_miss.size > 0:
            for n, row_missing in enumerate(self.row_miss):
                dyo = self.y_orig[row_missing, ~self.nans[row_missing]] - self.mean[
                    ~self.nans[row_missing]]  # mean-shifted observed data
                # print("y thing: ", self.y_orig[i, ~self.nans[i]])
                # print("dyo: ", dyo)
                Wm = self.PA[self.nans[row_missing], :]  # why PA/W inconsistency? Wm is missing loadings (W^miss),
                Wo = self.PA[~self.nans[row_missing], :]  # Wo is observed loadings (W^obs)
                # print("Wo: ", Wo)
                Rxinv = np.linalg.inv(Rx - self.tau * Wm.T @ Wm)  # recalc Σ_x with the loadings for missing data?
                # print("Rxinv: ",Rxinv)
                ex = self.tau * Wo.T @ vecT(dyo)  # expectation of x?
                # print("ex: ", ex)
                x = Rxinv @ ex  # what's even going on anymore
                dym = Wm @ x  # estimate missing values?
                dy = self.y_orig[row_missing, :].copy()
                # print("nans:" ,np.transpose(self.nans[i].nonzero()))
                dy[np.transpose((~self.nans[row_missing]).nonzero())] = vecT(dyo)  # update both observed ...
                dy[np.transpose(self.nans[row_missing].nonzero())] = vecT(dym)  # ... and missing values?
                # updating observed values seems redundant, but you're the boss
                # probably for the mysterious T to fit to all the data?

                # print("dym: ",dym)
                # print("dyo: ",dyo)
                # print("dy: ", dy)
                self.yest[row_missing, :] = dy + self.mean  # update full table estimations for the missing data
                # print("mean: ", self.mean)
                T = T + vecT(dy) @ x.T
                T[self.nans[row_missing], :] = T[self.nans[row_missing],] + Wm @ Rxinv
                trS = trS + dy @ vecT(dy) + np.sum(self.nans[row_missing]) / self.tau + np.trace(
                    Wm @ Rxinv @ Wm.T)  # some funky way to not recalculate trace from scratch every time? seems weird
                # np.sum() call just counts the nans
                self.scores[self.row_miss[n], :] = x.T
                # print("scores: ",self.scores)
        T /= self.rows
        trS /= self.rows
        Rxinv = np.linalg.inv(Rx)
        Dw = Rxinv + self.tau * T.T @ self.PA @ Rxinv + np.diag(
            self.alpha) / self.rows  # Σ_w^(-1) in CM Bishop, although np.diag(self.alpha) is not normalized there
        # print("Dw:", Dw)
        Dwinv = np.linalg.inv(Dw)  # Σ_w
        self.PA = T @ Dwinv  # OG comment: The new estimate of the principal axes (loadings)
        print("PA: ", self.PA)
        self.tau = (self.cols + 2 * self.gtau0 / self.rows) / (trS - np.trace(T.T @ self.PA) + (
                    self.mean @ vecT(self.mean) * self.gmu0 + 2 * self.gtau0 / self.btau0) / self.rows)
        # gamma distribution mean? a / b from wikipedia, or ã_τ / b̃_τ in CM bishop
        # except multiplied by two in both nominator and denominator
        # though b̃_τ seems to be really weird

        # print("tau: ",self.tau)
        self.tau = self.tau[0]
        self.SigW = Dwinv * (self.cols / self.rows)
        self.alpha = (2 * self.galpha0 + self.cols) / (
                    self.tau * np.diag(self.PA.T @ self.PA) + np.diag(self.SigW) + 2 * self.galpha0 / self.balpha0)
        # product of gamma distributions mean according to CM Bishop
        # ã_α / b̃_α (?)
        # print("alpha: ",self.alpha)

class BPCADTO:
    pass
class BPCACompression(NonLearningCompressionAlgorithm, LossyCompressionAlgorithm):
    name = "Shigeyuki Oba + Wolfram Stacklies BPCA"

    def __init__(self, n_components, threshold=1e-4, max_steps=2000):
        super().__init__()
        self.n_components = n_components
        self.max_steps = max_steps
        self.threshold = threshold

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            df = pickle.load(table_file)
            numeric = df.select_dtypes(['number'])
            non_numeric = df.select_dtypes(exclude=['number'])
            order = df.columns
            order_numeric = numeric.columns
            index = df.index
            bpca = BPCA(numeric.to_numpy(), self.n_components)
            tauold = np.inf
            for step in range(self.max_steps):
                bpca.doStep()
                if step % 10 == 0:
                    tau = bpca.tau
                    dtau = np.abs(np.log10(tau) - np.log10(tauold))
                    if dtau < self.threshold:
                        break
                    tauold = tau
            nonzero_loadings = np.nonzero(np.nanmax(np.abs(bpca.PA), axis=0) > 1e-10)[0]
            output = BPCADTO()
            output.scores = bpca.scores[:, nonzero_loadings]
            output.PA = bpca.PA[:, nonzero_loadings]
            output.non_numeric = non_numeric
            output.order = order
            output.order_numeric = order_numeric
            output.index = index

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
            restored_numeric = from_file.scores @ from_file.PA.T
            restored_numeric_df = pd.DataFrame(data=restored_numeric, index=from_file.index, columns=from_file.order_numeric)
            restored = pd.merge(from_file.non_numeric, restored_numeric_df, left_index=True, right_index=True)[from_file.order]
            pickle.dump(restored, decompressed_file)
