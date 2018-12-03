import math
import numpy as np
import pandas as pd
from quadprog import solve_qp
from scipy.stats import norm
from scipy.sparse.csr import csr_matrix
import random
import time

#########################################################################################################
# pyiSA is a Python package which provides access to iSA technology developed by
# VOICES from the Blogs. It is released for academic use only and licensed
# under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License
# see http://creativecommons.org/licenses/by-nc-nd/4.0/
# Warning: Commercial use of iSA is protected under the U.S. provisional patent application No. 62/215264
#########################################################################################################


class PyiSA(object):
    def __init__(self, boot_count=1000, predict=False, sequence_length=5, sparse=False,
                 verbose=False, tolerance=0):
        self.boot_count = boot_count
        self.predict = predict
        self.sequence_length = sequence_length
        self.sparse = sparse
        self.verbose = verbose
        self.tolerance = tolerance
        self.estimate = None
        self.estimate_table = None
        self.best = None
        self.best_table = None
        self.boot = None
        self.predict_cats = None
        self.elapsed_time = None
        return

    def _letters(self, num):
        if num > 25:
            return None
        return 'abcdefghijklmnopqrstuvwxyz'[num]

    def fit(self, X_train, X_test, y_train):
        # Expecting X: lists of hex strings, y: list of codes
        # This function will carry out the iSA algorithm and store its results in the class variables.

        if self.verbose:
            print('\nCommencing iSA run (verbose mode enabled)...')

        start_time = time.time()

        if not isinstance(y_train, list):
            try:
                y_train = list(y_train)
            except:
                raise Exception('Expected a list-like object for y_train; got {}.'.format(type(y_train)))

        X = X_train + X_test
        y = y_train + ([None] * len(X_test))

        feature_space = len(X[0])
        if self.verbose:
            print('Feature Space: {} features X {} documents ({} training, {} test)'.format(feature_space,
                                                                                            len(X),
                                                                                            len(X_train),
                                                                                            len(X_test)))

        seqlen = self.sequence_length
        if seqlen > 0:
            if feature_space <= 3:
                seqlen = 1
            if feature_space > seqlen * 24:
                seqlen = int(feature_space/24)
            if feature_space > 10 and seqlen < 3:
                seqlen = 3
            if seqlen >= feature_space:
                seqlen = feature_space

            split_count = int(math.floor(feature_space / seqlen))
            splits = list(np.cumsum([seqlen] * split_count))

            if splits[-1] > feature_space - 2:
                splits[-1] = feature_space
            else:
                splits.append(feature_space)
                split_count += 1

            if self.verbose:
                print('Augmenting dataset using {} splits...'.format(split_count))

            doc_count = len(X)
            new_X = [' '] * (split_count * doc_count)

            if split_count > 2:
                new_X[0:doc_count] = [self._letters(0) + X[i][0:splits[0]] for i in range(0, doc_count)]
                for this_split in range(1, split_count):
                    new_X[(doc_count * this_split):(doc_count * (this_split + 1))] = [self._letters(this_split) + X[i][splits[this_split - 1]:splits[this_split]] for i in range(0, doc_count)]

                new_y = y * split_count

                X = new_X
                y = new_y

            X_train = [X[i] for i in range(0, len(X)) if y[i] is not None]
            y_train = [y[i] for i in range(0, len(y)) if y[i] is not None]

        X_dist = pd.Series(X).value_counts(normalize=True)
        X_stems = X_dist.index

        if self.sparse:
            prXy = pd.crosstab(np.array(X_train), np.array(y_train), normalize='columns').to_sparse(fill_value=0.0)
        else:
            prXy = pd.crosstab(np.array(X_train), np.array(y_train), normalize='columns')

        cats = prXy.columns
        stems = prXy.index

        P = pd.DataFrame(index=X_stems, columns=cats).fillna(0)
        for a_stem in stems:
            P.iloc[X_stems.get_loc(a_stem)] = prXy.loc[a_stem]

        cat_count = len(P.columns)
        if self.verbose:
            print('Training iSA for {} categories.'.format(cat_count))

        A_matrix = pd.DataFrame(index=np.arange(cat_count), columns=np.arange(cat_count * 2 + 1)).fillna(0)
        A_matrix.iloc[:, 0] = 1
        A_matrix.iloc[:, 1:cat_count + 1] = np.diag([1] * cat_count)
        A_matrix.iloc[:, (cat_count + 1):(2 * cat_count + 1)] = np.diag([-1] * cat_count)

        b_vector = np.array([1] + ([0] * cat_count) + ([-1] * cat_count)).astype(np.float)

        abs_det_P = abs(np.linalg.det(P.transpose().dot(P)))
        if self.verbose:
            print(f"Absolute determinant of (P'*P): {abs_det_P:}")
        if abs_det_P < self.tolerance:
            raise Exception("Matrix P'*P is not invertible")

        try:
            b = solve_qp((P.transpose().dot(P)).values.astype(np.float),
                         X_dist.transpose().dot(P).values.astype(np.float),
                         A_matrix.values.astype(np.float), b_vector, meq=1)[0]
        except:
            b = [np.nan] * cat_count

        sigma2 = ((X_dist.values - (P.dot(b)).values)**2).sum() / (len(X_dist) - len(P.columns) - 1)
        quadratic_result = pd.DataFrame(index=cats, columns=['iSA'])
        quadratic_result.iloc[:, 0] = b

        result_table = pd.DataFrame(index=cats, columns=['Estimate', 'Std. Error', 'z value', 'Pr(>|z|)'])
        result_table.index.names = ['categories']
        result_table.iloc[:, 0] = quadratic_result.iloc[:, 0]

        try:
            std_err = np.sqrt(sigma2 * np.diag(np.linalg.inv(P.transpose().dot(P))))
        except:
            std_err = np.array([None] * cat_count)

        result_table['Std. Error'] = list(std_err)
        result_table['z value'] = result_table['Estimate'] / result_table['Std. Error']
        result_table['Pr(>|z|)'] = norm.sf(result_table['z value'])*2

        boot = []

        if self.boot_count > 0:

            if self.verbose:
                print('Bootstrapping... ({} passes)'.format(self.boot_count))

            for boot_pass in range(0, self.boot_count):
                indexer = random.choices(range(0, len(X_dist)), k=len(X_dist))
                this_P = pd.DataFrame(P.iloc[indexer].reset_index(drop=True))
                this_X_dist = X_dist.iloc[indexer].reset_index(drop=True)

                try:
                    b = solve_qp((this_P.transpose().dot(this_P)).values.astype(np.float),
                                 this_X_dist.transpose().dot(this_P).values.astype(np.float),
                                 A_matrix.values.astype(np.float), b_vector, meq=1)[0]
                except:
                    b = [np.nan] * cat_count

                boot.append(b)

            boot_df = pd.DataFrame(boot)

            best_table = pd.DataFrame(index=cats, columns=['Estimate', 'Std. Error', 'z value', 'Pr(>|z|)'])
            best_table.index.names=['categories']

            best_cf = pd.DataFrame(list(boot_df.mean(axis=0, skipna=True)), index=cats, columns=['iSAb'])
            best_table['Estimate'] = best_cf
            best_table['Std. Error'] = list(boot_df.std(axis=0, skipna=True))
            if np.isnan(best_table['Std. Error'].sum()):
                best_table['z value'] = [[np.nan] for i in range(0, len(cats))]
            else:
                best_table['z value'] = best_table['Estimate'] / best_table['Std. Error']
            best_table['Pr(>|z|)'] = norm.sf(best_table['z value']) * 2

        else:

            best_table = result_table
            best_cf = quadratic_result

        if self.predict:
            indexer = [X_stems.get_loc(a_stem) for a_stem in X]
            predicted = []
            for this_idx in indexer:
                predicted.append(np.argmax([P.iloc[this_idx, this_cat] * best_cf.iloc[this_cat] / X_dist[this_idx] for this_cat in range(0, cat_count)]))
            predict_names = [cats[i] for i in predicted]
        else:
            predict_names = None

        self.estimate = quadratic_result
        self.estimate_table = result_table
        self.best = best_cf
        self.best_table = best_table
        self.boot = boot
        self.predict_cats = predict_names

        elapsed_time = time.time() - start_time
        if self.verbose:
            print('[DONE]   Execution time: {:.3f} seconds'.format(elapsed_time))
        self.elapsed_time = elapsed_time

        return

    @staticmethod
    def prep_data(X):
        # This method takes a TDM (docs in rows, vocab in columns) and returns hex strings.
        # Can accept either a numpy Matrix or a Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()
        if not (isinstance(X, np.ndarray) or isinstance(X, csr_matrix)):
            raise Exception('prep_data() takes a Numpy array or a Scipy CSR sparse matrix; got {}'.format(type(X)))
        is_sparse = isinstance(X, csr_matrix)
        X_hex = []
        for a_row in X:
            if is_sparse:
                this_row = ''.join(['0' if np.isnan(i) or i == 0 else '1' for i in a_row.toarray()[0]])
            else:
                this_row = ''.join(['0' if np.isnan(i) or i == 0 else '1' for i in a_row])
            hex_string = ''.join((hex(int(this_row[i:i + 4], 2))[2:] for i in range(0, len(this_row), 4)))
            X_hex.append(hex_string)
        return X_hex


if __name__ == '__main__':
    print('pyiSA module; to use, import the "iSA" class into your program.')