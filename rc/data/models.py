#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2024 Robert A. Milton. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#  following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#  following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
#  promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Models for data storage. """

from __future__ import annotations

import numpy as np
import pandas as pd

from rc.base import *
from copy import deepcopy
import itertools
import random
import shutil
import scipy.stats
from enum import IntEnum


"""
class DataTable:

    @classmethod
    @property
    def CSV_OPTIONS(cls) -> Dict[str, Any]:
        return {'sep': ',', 'header': [0, 1], 'index_col': 0, }

    @property
    def csv(self) -> Path:
        return self._csv

    @property
    def is_empty(self) -> bool:
        return 0 == len(self._csv.parts)

    def write(self):
        assert not self.is_empty, 'Cannot write when DataTable.is_empty.'
        self.df.to_csv(path_or_buf=self._csv, sep=DataTable.CSV_OPTIONS['sep'], index=True)

    def __repr__(self) -> str:
        return str(self._csv)

    def __str__(self) -> str:
        return self._csv.name

    def __init__(self, csv: Path | str = Path(), df: pd.DataFrame = pd.DataFrame(), **kwargs):
        Args:
            csv: The csv file.
            df: The initial data. If this is empty, it is read from csv, otherwise it overwrites (or creates) csv.
        Keyword Args:
            kwargs: Updates DataTable.CSV_OPTIONS for csv reading as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
                This is not relevant to writing, which just uses DataTable.CSV_OPTIONS.
        self._csv = Path(csv)
        if self.is_empty:
            assert df.empty, 'csv is an empty path, but df is not an empty pd.DataFrame.'
            self.df = df
        elif df.empty:
            self.df = pd.read_csv(self._csv, **{**DataTable.CSV_OPTIONS, **kwargs})
        else:
            self.df = df
            self.write()
"""
class Normalisation:
    """ Encapsulates the normalization of data.
        X data is assumed to follow a Uniform distribution, which is normalized to U[0,1] , then inverse probability transformed to N[0,1].
        Y data is normalized to zero mean and unit variance.
    """

    @classmethod
    @property
    def UNIFORM_MARGIN(cls) -> float:
        return 1.0E-12

    @property
    def csv(self) -> Path:
        return self._fold.folder / 'normalization.csv'

    @property
    def DataTable(self) -> Table:
        self._frame = Table(self.csv) if self._frame is None else self._frame
        return self._frame

    @property
    def _relevant_stats(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        return (self.DataTable.pd.iloc[self.DataTable.pd.index.get_loc('min'), :self._fold.M], self.DataTable.pd.iloc[self.DataTable.pd.index.get_loc('rng'), :self._fold.M],
                self.DataTable.pd.iloc[self.DataTable.pd.index.get_loc('mean'), self._fold.M:], self.DataTable.pd.iloc[self.DataTable.pd.index.get_loc('std'), self._fold.M:])

    @property
    def is_applicable(self) -> bool:
        return self._is_applicable

    def apply_to(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply this normalization.

        Args:
            df: The pd.DataFrame to Normalize.
        Returns: df, Normalized.
        """
        if self._is_applicable:
            X_min, X_rng, Y_mean, Y_std = self._relevant_stats
            X = df.iloc[:, :self._fold.M].copy(deep=True)
            Y = df.iloc[:, self._fold.M:].copy(deep=True)
            X = X.sub(X_min, axis=1)[X_min.axes[0]].div(X_rng, axis=1)[X_rng.axes[0]].clip(lower=self.UNIFORM_MARGIN, upper=1 - self.UNIFORM_MARGIN)
            X.iloc[:, :] = scipy.stats.norm.ppf(X, loc=0, scale=1)
            Y = Y.sub(Y_mean, axis=1).div(Y_std, axis=1)
            return pd.concat((X, Y), axis=1)
        else:
            return df

    def undo_from(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Undo this normalization.

        Args:
            df: The (Normalized) pd.DataFrame to UnNormalize.
        Returns: df, UnNormalized.
        """
        if self._is_applicable:
            X_min, X_rng, Y_mean, Y_std = self._relevant_stats
            X = df.iloc[:, :self._fold.M].copy(deep=True)
            Y = df.iloc[:, self._fold.M:].copy(deep=True)
            X.iloc[:, :] = scipy.stats.norm.cdf(X, loc=0, scale=1)
            X = X.mul(X_rng, axis=1)[X_rng.axes[0]].add(X_min, axis=1)[X_min.axes[0]]
            Y = Y.mul(Y_std, axis=1)[Y_std.axes[0]].add(Y_mean, axis=1)[Y_mean.axes[0]]
            return pd.concat((X, Y), axis=1)
        else:
            return df

    def unscale_Y(self, dfY: pd.DataFrame) -> pd.DataFrame:
        """ Undo the Y-scaling of this normalization, without adding the Y-Mean. Suitable treatment for unNormalizing SD, for example.

        Args:
            dfY: The (Normalized) pd.DataFrame to UnNormalize.
        Returns: dfY, UnNormalized.
        """
        X_min, X_rng, Y_mean, Y_std = self._relevant_stats
        return dfY.copy(deep=True).mul(Y_std, axis=1)[Y_std.axes[0]] if self._is_applicable else dfY

    def X_gradient(self, X: NP.Matrix, m: int | List[int]):
        """ Computes the gradient of the unormalized inputs ``X[m]`` with respect to the normalized inputs ``Z[m]``.

        Args:
            X: An (N,M) matrix of unormalized inputs ``X[M]``
            m: A list of input axes to differentiate.
        Returns: An (N,len(m)) matrix of derivatives
        """
        X_rng = self._relevant_stats[1].values[m]
        return X_rng * scipy.stats.norm.pdf(X[..., m], loc=0, scale=1) if self._is_applicable else m / m

    def __repr__(self) -> str:
        return str(self.csv)

    def __str__(self) -> str:
        return self.csv.name

    def __init__(self, fold: Repo, data: Optional[pd.DataFrame] = None, is_applicable: bool = True):
        """ Initialize this Normalization. If the fold has already been Normalized, that Normalization is returned.

        Args:
            fold: The fold to Normalize.
            data: The data from which to calculate Normalization.
            is_applicable: Whether this Normalization should be applied to the data or not.
        """
        self._fold = fold
        self._is_applicable = is_applicable
        if self.csv.exists():
            self._frame = Table(self.csv)
        elif data is None:
            self._frame = None
        else:
            mean = data.mean()
            mean.name = 'mean'
            std = data.std()
            std.name = 'std'
            semi_range = std * np.sqrt(3)
            semi_range.name = 'rng'
            m_min = mean - semi_range
            m_min.name = 'min'
            m_max = mean + semi_range
            m_max.name = 'max'
            df = pd.concat((mean, std, 2 * semi_range, m_min, m_max), axis=1)
            self._frame = Table(self.csv, df.T)

#: Slice for ``n`` (row) in a Repo Table.
n: Tuple[slice, slice] = (slice(None, None, None), slice(None, 1, None))

#: Slice for ``l`` (categorical state) in a Repo Table.
l: Tuple[slice, slice] = (slice(None, None, None), slice(1, 2, None))

#: Slice for ``x`` (inputs) in a Repo Table.
x: Tuple[slice, slice] = (slice(None, None, None), slice(2, -1, None))

#: Slice for ``y`` (output) in a Repo Table.
y: Tuple[slice, slice] = (slice(None, None, None), slice(-1, None, None))

#: Delimiter for concatenating categorical variable values into a categorical state ``l``.
split: str = '; '

class Normalization(DataBase):
    class Tables(Tables):
        class NT(NamedTuple):
            names[i]: Table | Matrix = pd.DataFrame(defaults[names[i]].pd)
            ...
        readMetaData: dict[str, MetaData] = {names[i]: data[i].readMetaData, ...}
        writeMetaData: dict[str, MetaData] = {names[i]: data[i].writeMetaData, ...}
    defaultMetaData: MetaData = {'Override': 'Should not be empty'}


    def __call__(self, **metadata: Any) -> Self:
        return self

    def __init__(self, path: Store.Path, **data: Table | PD.DataFrame):
        super().__init__(path, **data)

    @classmethod
    def create(cls, path: Store.Path, train: PD.DataFrame, **meta: MetaData) -> Self:
        """ Create a Normalization from a ``PD.DataFrame``.

        Args:
            path: The Path to this Repo.
            train: The training data to normalize, as an indexed DataFrame with
                ``columns = [l, n, ...x.[m]..., y]``.
            meta: The meta to update ``cls.defaultMetaData`` and record in ``meta.json``.

        Returns: The Normalization created.
        """
        if not 'n' in train.columns:
            # Insert column 'n'.
            cols = train.columns.tolist()
            train['n'] = train.index
            train = train['n', *cols]
        # Sorted, unique l strings.
        train = train.sort_values(['l', 'n'])
        ll = pd.Series(train.loc[:, 'l'].unique())
        # Table of split l strings.
        data = {'l_columns': np.atleast_2d(ll.apply(lambda l: str(l).split(split), result_type = 'expand'))}
        columns = meta.pop('headers', {}).pop('l', None)
        columns = [f'l.{l}' for l in range(data['l_columns'].shape[1] - 1)] if columns is None else columns
        columns += ['output']
        data['l_columns'] = pd.DataFrame(data['l_columns'], columns)
        # Table for mapping l.
        ll = pd.Series(data = ll.index.values, index = ll)
        l_map = train.loc[:, 'l']
        l_map = l_map[l_map.ne(l_map.shift())]
        data['l'] = pd.concat([ll, pd.Series(data = l_map.index.values, index = l_map)], axis = 1)
        data['l'].columns = ['l', 'n']
        # Stats for x and y.
        xx = train[x]
        Meta.create(cls._meta_in(path), **meta)
        return cls(path, **data)

class Repo(DataBase):
    """ A Repo is a model consisting only of (training) data and metadata.
        This must be further split into Fold(Repositories) contained within the Repo before it can be used.
    """

    class Tables(Tables):

        class Tables(NamedTuple):
            """ The DataTables of a Repo.

            Attributes:
                train: Training data.
            """
            train = pd.DataFrame([[None, None, None]], columns=('l','x','y'))

    defaultMetaData: MetaData = {'headers': {'l': {'category'}, 'x': {'input'}, 'y': {'output'}},
                                 'shuffle before folding': False, 'K': 0}

    @property
    def folds(self) -> range:
        """ The indices of the folds contained in this Repo."""
        return range(abs(self._meta['K']) if self._meta['K'] <= 0 else self._meta['K'] + 1)

    def fold_path(self, k: int) -> Path:
        return self._path / f'fold.{k:d}'

    def rotate(self, rotation: NP.Matrix | None) -> Repo:
        """ Uniformly rotate the Folds in a Repo. The rotation (like normalization) applies to each fold, not the repo itself.

        Args:
            rotation: The (M,M) rotation matrix to apply to the inputs. If None, the identity matrix is used.
            If the matrix supplied has the wrong dimensions or is not orthogonal, a random rotation is generated and used instead.
        Returns: ``self``, for chaining calls.
        """
        M = self._meta['M']
        if rotation is None:
            rotation = np.eye(M)
        elif rotation.shape != (M, M) or not np.allclose(np.dot(rotation, rotation.T), np.eye(M)):
            rotation = scipy.stats.special_ortho_group.rvs(M)
        for k in self.folds:
            Fold(self, k).X_rotation = rotation
        return self

    def __call__(self, K: int, **metadata: Any) -> Self:
        """ Fold this repo into K Folds, indexed by range(K).

        Args:
            K: The number of Folds, of absolute value between 1 and N inclusive.
                An improper Fold, indexed by K and including all data for both training and testing is included by default.
                To suppress this give K as a negative integer.
            shuffle_before_folding: Whether to shuffle the data before sampling.
            normalization: An optional normalization.csv file to use.
            is_normalization_applicable: Whether normalization is applicable. ``False`` means that normalization whatsoever will be applied.
        Returns: ``self``, for chaining calls.
        Raises:
            IndexError: Unless 1 &lt= K &lt= N.
        """
        data = self.data.df
        N = data.shape[0]
        if not (1 <= abs(K) <= N):
            raise IndexError(f'K={K:d} does not lie between 1 and N={N:d} inclusive.')
        for k in range(max(abs(K), self.K) + 1):
            shutil.rmtree(self.fold_path(k), ignore_errors=True)
        index = list(range(N))
        if shuffle_before_folding:
            random.shuffle(index)
        self._meta.update({'K': abs(K), 'has_improper_fold': K > 0, 'shuffle before folding': shuffle_before_folding})
        self.write_meta()
        normalization = Normalization(self, self._data.pd).csv if normalization is None else normalization
        if K > 0:
            Fold.from_dfs(parent=self, k=K, data=data.iloc[index], test_data=data.iloc[index], normalization=normalization,
                          is_normalization_applicable=is_normalization_applicable)
        K = abs(K)
        K_blocks = [list(range(K)) for dummy in range(int(N / K))]
        K_blocks.append(list(range(N % K)))
        for K_range in K_blocks:
            random.shuffle(K_range)
        indicator = list(itertools.chain(*K_blocks))
        for k in range(K):
            indicated = tuple(zip(index, indicator))
            data_index = [index for index, indicator in indicated if k != indicator]
            test_index = [index for index, indicator in indicated if k == indicator]
            data_index = test_index if data_index == [] else data_index
            Fold.from_dfs(parent=self, k=k, data=data.iloc[data_index], test_data=data.iloc[test_index], normalization=normalization,
                          is_normalization_applicable=is_normalization_applicable)
        return self

    def __init__(self, path: Store.Path, train: Table | PD.DataFrame = None):
        """ Read the Repo in ``path``.

        Args:
            path: The Path to this Repo.
            train: The training data to populate this Repo. If ``None``, the Repo in ``path``
                will be read, otherwise it will be created.
        """
        super().__init__(path, train = train)
        self._normalization = Normalization(self._normalization_in(path))

    @classmethod
    def create(cls, path: Store.Path, train: PD.DataFrame, **meta: Any) -> Self:
        """ Create a Repo from a ``PD.DataFrame``.

        Args:
            path: The Path to this Repo.
            train: The data to record in ``train.csv``.
            meta: The meta to update ``cls.defaultMetaData`` and record in ``meta.json``.

        Returns: The Repo created.
        """
        Meta.create(cls._meta_in(path), **(cls.defaultMetaData | meta))
        return Repo(path, train = train)

    @classmethod
    def from_pd(cls, path: Store.Path, train: PD.DataFrame, **meta: Any) -> Self:
        """ Create a Repo from a ``PD.DataFrame``.

        Args:
            path: The Path to this Repo.
            train: The data to record in ``train.csv``.
            meta: The meta to update ``cls.defaultMetaData`` and record in ``meta.json``.

        Returns: The Repo created.
        """
        meta = cls.defaultMetaData | meta
        train = train.rename(str.lower, axis = 'columns', level = 0)
        dd = {'n': pd.DataFrame(train.index, columns=['n'])}
        for header, headers in meta['headers'].items():
            dd[header] = {'headers': {header}.union(headers).intersection(train.columns.levels[0])}
            dd[header] |= {'pd': train[list(dd[header]['headers'])]}
            dd[header] |= {'pd': pd.DataFrame(dd[header]['pd'].to_numpy(),
                                              columns = dd[header]['pd'].columns.droplevel(0))}
        meta['normalization'] |= {'Li': dd['l']['pd'].shape[1], 'Lo': dd['y']['pd'].shape[1]}
        result = dd['n'].join([dd[header]['pd'] for header in meta['headers'].keys()])
        id_vars = dd['n'].columns.union(dd['l']['pd'].columns).union(dd['x']['pd'].columns)
        result = result.melt(id_vars = id_vars, var_name = 'col', value_name = 'y').dropna()
        meta['normalization'] |= {'headers': {header: dd[header]['pd'].columns.to_list()
                                             for header in meta['headers']}}
        result['l'] = result[dd['l']['pd'].columns.to_list()+['col']].astype(str).agg(split.join, axis = 1)
        train = result['n', 'l', *(dd['x']['pd'].columns.to_list()), 'y']
        meta |=  { 'M': dd['x']['pd'].shape[1]}
        normalization = Normalization.create(cls._normalization_in(path), train, **meta['normalization'])
        return cls.create(path, train, **meta)

    @classmethod
    def from_csv(cls, path: Store.Path, train: Store.Path, **meta: Any) -> Repo:
        """ Create a Repo from a csv file.

        Args:
            path: The location (folder) of the target Repo.
            train: The file containing the data to record in [Return].csv.
            meta: The meta to record in meta.json, defaulting to
                ``{'src': {'path': [path], 'read_options': {'header': [0, 1]}}}``.
                ``'src' : 'read_options'``, which may be amended, is ``MetaData`` passed directly to
                `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_.

        Raises:
            FileNotFoundError: If ``train`` is not a file.

        Returns: The Repo created.
        """
        meta = {'src': {'path': str(train.absolute()), 'read_options': {'header': [0, 1], 'index_col': 0}}
                } | meta
        train = Path(train)
        if not train.is_file():
            raise FileNotFoundError(f'Training data {train} does not exist.')
        return cls.from_pd(path = path, train = pd.read_csv(train, **meta['src']['read_options']), **meta)

    @staticmethod
    def _normalization_in(path: Store.Path) -> Path:
        return Path(path) / 'normalization'

# def from_csv(cls, path: Store.Path, train: Store.Path, PCA: bool = False, meta: Dict = None,
#              **kwargs) -> Repo:
#     """ Create a Repo from a csv file.
#
#     Args:
#         path: The location (folder) of the target Repo.
#         train: The file containing the data to record in [Return].csv.
#         PSA: Whether to create a single fold in which Principal Component Analysis (PCA) has been performed on the inputs.
#         meta: The metadata to record in [Return].meta.json.
#         kwargs: Updates Repo.CSV_OPTIONS for reading the csv file, as detailed in
#             https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pd.read_csv.html.
#     Returns: A new Repo located in folder.
#     """
#     train = Path(train)
#     origin_csv_kwargs = cls.CSV_OPTIONS | kwargs
#     data = DataTable(train, **origin_csv_kwargs)
#     meta = cls.META if meta is None else cls.META | meta
#     meta['origin'] = {'csv': str(train.absolute()), 'origin_csv_kwargs': origin_csv_kwargs}
#     repo = cls.from_df(path, data.pd, meta)
#     if PCA:
#         repo = repo.into_K_folds(-1)
#         fold = Repo(repo.fold_folder(0))
#         X = fold.X.values
#         print(f'pre mean = {np.mean(fold.X.values, axis = 0)}')  # DEBUG:
#         cov = np.cov(X, rowvar = False)
#         eigenvalues, eigenvectors = np.linalg.eigh(cov)
#         idx = eigenvalues.argsort()[::-1]
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]
#         cov = np.einsum('ij,ij->j', eigenvectors, eigenvectors)
#
#         repo = repo.rotate_folds(eigenvectors.T)
#         # Test Code
#         fold = Fold(repo, 0)
#         fold.data.df.iloc[:, :fold.M] /= np.sqrt(eigenvalues)
#         fold.test_data.pd.iloc[:, :fold.M] /= np.sqrt(eigenvalues)
#         print(f'post mean = {np.mean(fold.X.values, axis = 0)}')  # DEBUG:
#         print(f'post cov = {np.cov(fold.X.values, rowvar = False)}')  # DEBUG:
#         # end of
#         path = repo.fold_folder(0)
#         path.rename(path.parent / 'PCA')
#     return repo


class Fold(Repo):
    """ A Fold is defined as a folder containing a ``data.csv``, a ``meta.json`` file and a ``test.csv`` file.
    A Fold is a Repo equipped with a test_data pd.DataFrame backed by ``test.csv``.

    Additionally, a fold can reduce the dimensionality ``M`` of the input ``X``.
    """

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def test_csv(self) -> Path:
        return self._test_csv

    @property
    def test_data(self) -> Table:
        return self._test_data

    @property
    def test_x(self) -> pd.DataFrame:
        """ The test_data input x, as an (n,M) design Matrix with column headings."""
        return self._test_data.pd[self._meta['data']['X_heading']]

    @property
    def test_y(self) -> pd.DataFrame:
        """ The test_data output y as an (n,L) Matrix with column headings."""
        return self._test_data.pd[self._meta['data']['Y_heading']]

    def _X_rotate(self, DataTable: Table, rotation: NP.Matrix):
        """ Rotate the input variables in a DataTable.

        Args:
            DataTable: The DataTable to rotate. Will be written after rotation.
            rotation: The rotation Matrix.
        """
        DataTable.pd.iloc[:, :self.M] = np.einsum('Nm,Mm->NM', DataTable.pd.iloc[:, :self.M], rotation)
        DataTable.write()

    @property
    def X_rotation(self) -> NP.Matrix:
        """ The rotation matrix applied to the input variables self.X, stored in X_rotation.csv. Rotations are applied and stored cumulatively."""
        return Table(self._X_rotation, header=[0]).pd.values if self._X_rotation.exists() else np.eye(self.M)

    @X_rotation.setter
    def X_rotation(self, value: NP.Matrix):
        """ The rotation matrix applied to the input variables self.X, stored in X_rotation.csv. Rotations are applied and stored cumulatively."""
        self._X_rotate(self._data, value)
        self._X_rotate(self._test_data, value)
        old_value = self.X_rotation
        Table(self._X_rotation, pd.DataFrame(np.matmul(old_value, value)))

    def __init__(self, parent: Repo, k: int, **kwargs):
        """ Initialize Fold by reading existing files. Creation is handled by the classmethod Fold.from_dfs.

        Args:
            parent: The parent Repo.
            k: The index of the Fold within parent.
            M: The number of input columns used. If not 0 &lt M &lt self.M, all columns are used.
        """
        init_mode = kwargs.get('init_mode', Repo._InitMode.READ)
        super().__init__(parent.fold_path(k), init_mode=init_mode)
        self._X_rotation = self.folder / 'X_rotation.csv'
        self._test_csv = self.folder / 'test.csv'
        if init_mode == Repo._InitMode.READ:
            self._test_data = Table(self._test_csv)
            self._normalization = Normalization(self)

    @classmethod
    def from_dfs(cls, parent: Repo, k: int, data: pd.DataFrame, test_data: pd.DataFrame,
                 normalization: Optional[Path | str] = None, is_normalization_applicable: bool = True) -> Fold:
        """ Create a Fold from a pd.DataFrame.

        Args:
            parent: The parent Repo.
            k: The index of the fold to be created.
            data: Training data.
            test_data: Test data.
            normalization: An optional normalization.csv file to use.
            is_normalization_applicable: Whether normalization is applicable. ``False`` means that normalization whatsoever will be applied.
        Returns: The Fold created.
        """

        fold = cls(parent, k, init_mode=Repo._InitMode.CREATE)
        fold._meta = cls.META | parent.meta | {'k': k}
        fold._normalization = Normalization(fold, data, is_normalization_applicable)
        if normalization is not None:
            shutil.copy(Path(normalization), fold._normalization.csv)
        fold._data = Table(fold._csv, fold.normalization.apply_to(data))
        fold._test_data = Table(fold._test_csv, fold.normalization.apply_to(test_data))
        fold._update_meta()
        return fold

