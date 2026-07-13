#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2027 Robert A. Milton
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

""" Experiment Designs and Measures. """
import pylab as pl

from rc.base import *


class Abstract(Table):
    """ Abstract scaffolding for Design and Measure, providing shared tabulation facilities for user data."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a DataFrame or Table. """

    axisTypes: list[str] = []
    """ The axisTypes in any Design, ordered from left to right. """

    class AxisValidator(NamedTuple):
        """ A NamedTuple for validating an axis in a Design. Used exclusively by the ``__call__()`` method."""
        axis: str  #: The axis to validate.
        Type: type = Category | Float   #: The Type of data accepted by this axis.

    class Slice(NamedTuple):
        """ A pair of ``slices``, to slice TableData."""
        rows: slice = slice(None, None) #: Rows are the first tensor rank
        axes: slice = slice(None, 1)   #: Axes are the second tensor rank

    extPivot: str = '.yy'
    """ The suffix appended to Design.yPivot() Tables. """

    @cached_property
    def isFat(self) -> bool:
        """ Whether this design is Fat."""
        return self._isFat

    @property
    def M(self) -> int:
        """ Counts the x-axes. """
        return self._M

    @property
    def J(self) -> int:
        """ Counts the i-axes. """
        return self._J

    @property
    def L(self) -> int:
        """ Counts the y-axes occurring in the l-axis. """
        return self._L

    @property
    def ls(self) -> list[str]:
        """ Lists the y-axes occurring in the l-axis. """
        return self._ls

    @property
    def js(self) -> list[str]:
        """ Lists the i-axes. """
        return self._js

    def ijs(self, j: str = '') -> list[str]:
        """ Lists the values ``i(j)`` appearing in the i-axis named ``j``.

        Args:
            j: The i-axis to list values of. If ``j`` is empty, all i-values are listed.
        Returns: A list of ``i(j)`` values.
        Raises: ValueError if non-empty ``j`` is not in ``self.js``.
        """
        if self.isFat:
            result =  self._df.filter(pl.col('j') == j)['i'].unique(maintain_order=True).to_list()
        elif j == '' or j in self.js:
            result = self._df['i'].unique(maintain_order=True).to_list() if self.J else []
            if j:
                idx = self.js.index(j) + 1
                result = list({cat.split('│')[idx] for cat in result})
        else:
            raise ValueError(f'There is no i-axis called {j} in {self}.js = {self.js}.')
        return result
    
    @property
    @abstractmethod
    def y(self) -> Abstract.Slice:
        raise NotImplementedError

    def yPivot(self, path: PathLike | None = None) -> Table:
        """ Create a Table at ``path`` consisting of ``self`` with 'y' values pivoted on the 'l' axis.

        Args:
            path: The Path to store this Table, overwritten if existing.

        Returns: A fat Table with no 'l' axis but several output axes in place of the 'y' axes.
        """
        if not path: path = self.path.with_suffix(self.path.suffix + self.extPivot)
        yAxisPrefix = 'y│'
        axes = self.head[self.y.axes]
        rename = ({l: yAxisPrefix + l for l in self.ls} if len(axes) == 1 else
                  {'_'.join((axis,l)): yAxisPrefix + l + axis[1:] for l in self.ls for axis in axes})
        columns = self.head[:-len(axes) - 1] + list(rename.values())
        df = self._df.pivot('l', values=axes, maintain_order=True)
        df = df.rename(rename)
        return Table.create(path, df.select(columns))

    @abstractmethod
    def __call__(self, update: TableData | Self | None = None, write_csv: bool = True) -> Self:
        """ Overrides must set ``self._M, self._J, self._L, self._isFat, self._ls, self._js``. """
        super().__call__(update, write_csv=False)
        return self

    @classmethod
    @abstractmethod
    def create(cls, path: PathLike, df: DataFrame | Table , isFat: bool = False) -> Self:
        raise NotImplementedError


class Design(Abstract):
    """ A Design of user data, tabulating continuous inputs, categorical inputs, and unpivoted outputs."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a Table. """

    axisLexicon: dict[str,str] = {'index': 'n',
                                  'input': 'x', 'in': 'x', 'continuous': 'x', 'float': 'x',
                                  'category': 'i', 'cat': 'i', 'discrete': 'i', 'int': 'i', 'str': 'i',
                                  'output-axis': 'l', 'y-axis': 'l', 'output axis': 'l', 'y axis': 'l',
                                  'output': 'y', 'out':'y', 'map': 'y', 'func':'y', }
    """ The lexicon of axisTypes. """

    axisTypes: list[str] = list(dict.fromkeys(axisLexicon.values()))
    """ The axisTypes in any Design, ordered from left to right. """

    @property
    def n(self) -> Abstract.Slice:
        return Abstract.Slice()

    @property
    def x(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(1, self._M + 1))

    @property
    def i(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(self._M + 1, -2))

    @property
    def l(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(-2, -1))

    @property
    def y(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(-1, None))

    def __call__(self, update: TableData | Self | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        self._M = getattr(self, '_M', None)
        if __debug__ or self._M is None:
            schema = tuple(self.AxisValidator(*schema) for schema in self.schema.items())
            # Verify singleton axisTypes
            validate = {'n': schema[0],
                     'l': schema[len(schema)-2],
                     'y': schema[len(schema)-1],}
            for axisType, actual in validate.items():
                assert axisType == actual.axis, f'Axis {actual.axis} should be {axisType} in {self}.'
                if axisType == 'y':
                    self._df = self._df.with_columns(pl.col(axisType).cast(Float))
                elif actual.Type != Category:
                    self._df = self._df.with_columns(pl.col(axisType).cast(String).cast(Category))
            # Verify non-singleton axisTypes
            correct = (self.AxisValidator(axis='i', Type=Category), self.AxisValidator(axis='x', Type=Float))
            validate = {}
            for valid in correct:
                validate[valid.axis] = [check for check in schema if check.axis.startswith(valid.axis)]
                invalid = [_invalid.axis for _invalid in validate[valid.axis] if _invalid.Type != valid.Type]
                if invalid: self._df = self._df.with_columns(pl.col(invalid).cast(String).cast(valid.Type))
            iAxes = len(validate['i'])
            isFat = iAxes > 1
            ls = self._df['l'].unique(maintain_order=True).to_list()
            js = ([axisValidator.axis[2:] for axisValidator in validate['i']] if isFat else
                  validate['i'][0].axis.split('│')[1:] if iAxes else [])
            valid = {'_M': len(validate['x']), '_L': len(ls), '_J': len(js),
                     '_isFat': isFat, '_ls': ls, '_js': js,}
            if self._M is None:
                for attr, actual in valid.items():
                    setattr(self, attr, actual)
            else:
                for attr, actual in valid.items():
                    assert getattr(self, attr, None) == correct, \
                        f'{actual[-1]} changed from {getattr(self, attr, None)} to {actual} in {self}.'
            #
            head = self.head
            assert head[1] == validate['x'][0].axis, f'Axis[1] should be an x-axis in {self}.'
            if validate['i']:
                assert self.isFat ^ (iAxes == 1), \
                    f'Only one i-axis allowed in {self} unless self.isFat.'
                assert head.index(validate['i'][0].axis) == head.index(validate['x'][-1].axis) + 1, \
                    f'i-axes should immediately follow x-axes in {self}.'
                assert head[-3] == validate['i'][-1].axis, f'Axis[-3] should be an i-axis in {self}.'
            else:
                assert head[-3] == validate['x'][-1].axis, f'Axis[-3] should be an x-axis in {self}.'
        if write_csv: self._df.write_csv(self._path, **self.writeOptions)
        return self

    @classmethod
    def create(cls, path: PathLike, df: DataFrame | Table , isFat: bool = False) -> Self:
        """ Create a Table at ``path``, overwriting.

        Args:
            path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            df: The DataFrame to store.
            isFat: Whether to  create a fat Design.

        Returns: The Table created.
        Raises: AssertionError if ``df`` is unacceptable.
        """
        df = getattr(df, 'df', df)
        head = ['n'] + [cls.axisLexicon.get(left.lower(), left.lower()) + con + right
                         for _head in df.columns[1: ] for left, con, right in [_head.partition('│')]]
        df = df.rename(dict(zip(df.columns, head)))
        head = {axisType: [_head for _head in head if _head.partition('│')[0] == axisType]
                 for axisType in cls.axisTypes}
        y = head.pop('y', [])
        assert y, f'Design at {path} has no y-axis.'
        if len(head['l']) > 0:
            # Only accept first output axis
            df = df.with_columns(*[_head for axisType in head.keys() for _head in head[axisType]]
                                 , pl.col(y[0]).alias('y'))
        else:
            df = df.unpivot(y, index=[_head for axisType in head.keys() for _head in head[axisType]],
                            variable_name='l', value_name='y').with_columns(pl.col('l').str.slice(2))
        if isFat:
            assert len(head['i']), f'A fat Design requires at least one i-axis in {df}.'
            if set([_head.count('│') for _head in head['i']]) != {1}:
                # split head['i']
                rename = {_head: '_' + _head for _head in head['i']}
                df = df.rename(rename)
                cats = {'_' + _head : _head.split('│') for _head in head['i']}
                head = head['n'] + head['x']
                lnx = len(head)
                for old, new in cats.items():
                    newHeads = {f'field_{j}': 'i│' + cat for j, cat in enumerate(new[1:])}
                    df = (df.with_columns(pl.col(old).cast(String).str.split_exact('│', len(new))
                                                  .alias('│││')).unnest('│││').rename(newHeads))
                    head += newHeads.values()
                df = df.select(head + ['l', 'y'])
                head = {'i': head[lnx:]}
        elif len(head['i']) > 1:
            # join head['i']
            _head = '│'.join(head['i'][0:1] + [cat[2:] for cat in head['i'][1:]])
            df = df.with_columns(pl.concat_str([pl.col(cat) for cat in head['i']],
                                               separator='│').alias(_head))
            df = df.select(head['n'] + head['x'] + [_head, 'l', 'y'])
            head = {'i': [_head]}
        return cls(path, df.sort(['l'] + head['i'], maintain_order=True))


class Measure(Abstract):
    """ Measures a Design of user data."""

    axisTypes: list[str] = ['x', 'j', 'i', 'l', 'y']
    """ The axisTypes in any Design, ordered from left to right. """

    agg: str = '││'
    """ The value which indicates aggregation over a column. """

    _by: dict[str, [Callable[[], DataFrame]]] = {'mean': pl.mean, 'min': pl.min, 'max': pl.max, 'std': pl.std, }
    by: list[str] = list(_by.keys())
    """ Ways to measure a Float axis. """

    _y = -len(by) - 1

    @property
    def x(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(None, self._x))

    @property
    def j(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(self._x, self._x + (1 if self.isFat else 0)))

    @property
    def i(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(self._y - 2, self._y - (1 if self._J  else 2)))

    @property
    def l(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(self._y - 1, self._y))

    @property
    def y(self) -> Abstract.Slice:
        return Abstract.Slice(axes=slice(self._y, None))

    def xAxes(self, path: PathLike, statistic: str = '50%') -> Table:
        """ Describe_ the x-axes of this Measure.

        .. _Describe: https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.describe.html

        Args:
            path: The Path to store the description.
            statistic: The statistic recorded as the aggregate ││.

        Returns: A Table of statistics for each x-axis.

        """
        head = self.head[self.x.axes]
        if iAxis := self.head[self.i.axes]:
            df = self._df.filter(pl.col(iAxis) != '││').select(head)
        else:
            df = self._df.select(head)
        for m in range(self._M):
            idx = m * len(self.by)
            xAxis = head[idx].rpartition('│')[0][2:]
            desc = (df.select(head[idx: idx + len(self.by)])
                    .describe().with_columns(pl.col('statistic').cast(Category)))
            stat = (desc.filter(pl.col('statistic') == statistic)
                         .with_columns(statistic=pl.lit('││', Category)))
            desc = desc.vstack(stat)
            desc = desc.rename({_head: _head[len(xAxis) + 3 : ] for _head in desc.columns[1:]})
            desc = desc.insert_column(0, pl.lit(xAxis, Category).alias('x-axis'))
            result = desc if m == 0 else result.vstack(desc)
        return Table(path, result)

    def __call__(self, update: TableData | Self | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        if __debug__ or self._M is None:
            schema = tuple(self.AxisValidator(_head.partition('│')[0], _type)
                           for _head, _type in self.schema.items())
            count = {axisType: len([axisValidator for axisValidator in schema if axisValidator.axis == axisType])
                     for axisType in self.axisTypes}
            isFat = count['j'] > 0
            ls = self._df['l'].unique(maintain_order=True).to_list()
            js = (self._df['j'].unique(maintain_order=True).to_list() if isFat else
                  self.head[count['x']].split('│')[1:] if count['i'] > 0 else [])
                  # If there is an i-axis, its index is count['x'].
            valid = {'_M': count['x'] // len(self.by), '_L': len(ls), '_J': len(js),
                     '_isFat': isFat, '_ls': ls, '_js': js, }
            if getattr(self, '_M', None) is None:
                for attr, correct in valid.items():
                    setattr(self, attr, correct)
                self._x = self._M * len(self.by)
            else:
                for attr, correct in valid.items():
                    assert getattr(self, attr, None) == correct, \
                        f'{correct[-1]} changed from {getattr(self, attr, None)} to {correct} in {self}.'
            validate = [self.AxisValidator('x', Float), self.AxisValidator('j', Category),
                        self.AxisValidator('i', Category), self.AxisValidator('l', Category),
                        self.AxisValidator('y', Float),]
            for check in validate:
                assert count[check.axis] == 0 \
                       or list(set(schema[getattr(self, check.axis).axes])) == [check], \
                    f'{self}.{check.axis} schema violation.'
        if write_csv: self._df.write_csv(self._path, **self.writeOptions)
        return self

    def thin(self) -> Self:
        """ Thin this Measure.

        Returns: The thin version of this measure.
        """
        if not self.isFat: return self
        df = self._df.select()
        return self.create(self.path.with_suffix(''), df)
    @classmethod

    def create(cls, path: PathLike, design: Design) -> Self:
        """ Measure a Design.

        Args:
            path: The Path to store this Measure.
            design: The Design to measure.

        Returns: The Measure created.
        """
        by = {axis + '│' + by: f(axis)
              for axis in design.head[design.x.axes] + design.head[design.y.axes]
              for by, f in cls._by.items()}
        columns = list(by.keys())
        by |= {'y│q' : pl.count('y').cast(Float), }
        cats = ['j', 'i'] if design.isFat else design.head[design.i.axes] if design.J > 0 else []
        columns[cls._y + 1 : cls._y + 1] = cats + ['l', 'y│q']
        result = design.df.group_by('l').agg(**by)
        if design.J == 0:
            return cls(path, result.select(columns).sort('l'))
        count = result.select(['l', 'y│q']).rename({'y│q': '│││'})
        result = result.with_columns(*tuple((pl.lit(cls.agg, Category).alias(_cat)
                                             for _cat in cats))).select(columns)
        for cat in reversed(design.head[design.i.axes]):
            df = design.df.select(design.head[design.x.axes] + [cat] + design.head[design.l.axes]
                                  + design.head[design.y.axes])
            df = df.group_by((cat, 'l'), maintain_order=True).agg(**by)
            if design.isFat: df = df.with_columns(pl.lit(cat[2:], Category).alias('j')).rename({cat: 'i'})
            df = df.join(count, on='l').with_columns(**{'y│q': pl.col('y│q')/pl.col('│││')})
            result = df.select(columns).vstack(result)
        return cls(path, result.sort('l'))
