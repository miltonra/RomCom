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

""" Design Tables (Matrices). """

from rc.base import *


class Design(Table):
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

    class AxisVerifier(NamedTuple):
        """ A NamedTuple for verifying an axis in a Design. Used exclusively by the ``__call__()`` method."""
        axis: str  #: The axis to verify.
        Type: type = Category | Float   #: The Type of data accepted by this axis.

    class Slice(NamedTuple):
        """ A pair of ``slices``, to slice TableData."""
        rows: slice = slice(None, None) #: Rows are the first tensor rank
        axes: slice = slice(None, 1)   #: Axes are the second tensor rank

    extFat: str = '.f'
    """ The suffix appended to fat Designs. """

    extPivot: str = '.y'
    """ The suffix appended to Design.yPivot() Tables. """

    @property
    def isFat(self) -> bool:
        """ Whether this design is Fat."""
        return str(self).endswith(self.extFat)

    @property
    def M(self) -> int:
        """ Counts the number of continuous inputs. """
        return self._M

    @property
    def J(self) -> int:
        """ Counts the number of Category axes. """
        return self._J

    @property
    def L(self) -> int:
        """ Counts the number of output axes. """
        return self._L

    @property
    def n(self) -> Slice:
        return self.Slice()

    @property
    def x(self) -> Slice:
        return self.Slice(axes=slice(1, self._M + 1))

    @property
    def i(self) -> Slice:
        return self.Slice(axes=slice(self._M + 1, -2))

    @property
    def l(self) -> Slice:
        return self.Slice(axes=slice(-2, -1))

    @property
    def y(self) -> Slice:
        return self.Slice(axes=slice(-1, None))

    def yPivot(self, path: Path | None = None, yAxisPrefix: str = f'y{Table.con}', **kwargs) -> Table:
        """ Create a Table at ``path`` consisting of ``self`` with 'y' values pivoted on the 'l' axis.

        Args:
            path: The Path to store this Table, overwritten if existing.
            yAxisPrefix: The prefix of the new 'y' axis, to appear before the 'l' axis values.
                The default is 'y│', pass to ``''`` for no prefix.
            **kwargs: KeywordArguments passed directly to `DataFrame()`_.

        Returns: A Table with no 'l' axis but several output axis in place of the 'y' axis.

        .. DataFrame(): https://docs.pola.rs/api/python/dev/reference/dataframe/index.html
        """
        return Table.create(path if path else self.path.with_suffix(self.path.suffix + self.extPivot),
                            self._df.with_columns((pl.lit(yAxisPrefix) + pl.col('l')).alias('l'))
                            .pivot('l', values='y'), **kwargs)

    def __call__(self, update: Self | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        self._M = getattr(self, '_M', 0)
        if __debug__ or self._M == 0:
            schema = tuple(self.AxisVerifier(*schema) for schema in self.schema.items())
            # Verify singleton axisTypes
            verify = {'n': schema[0],
                     'l': schema[len(schema)-2],
                     'y': schema[len(schema)-1],}
            for axisType, actual in verify.items():
                assert axisType == actual.axis, f'Axis {actual.axis} should be {axisType} in {self}.'
                if axisType == 'y':
                    self._df = self._df.with_columns(pl.col(axisType).cast(Float))
                elif actual.Type != Category:
                    self._df = self._df.with_columns(pl.col(axisType).cast(String).cast(Category))
            # Verify non-singleton axisTypes
            verify = (self.AxisVerifier(axis='i', Type=Category), self.AxisVerifier(axis='x', Type=Float))
            check = {}
            for correct in verify:
                check[correct.axis] = [check for check in schema if check.axis.startswith(correct.axis)]
                incorrect = [incorrect.axis for incorrect in check[correct.axis] if incorrect.Type != correct.Type]
                if incorrect:
                    self._df = self._df.with_columns(pl.col(incorrect).cast(String).cast(correct.Type))
            verify = {'_M': len(check['x']),
                      '_L': self._df.select(pl.col('l').n_unique()).item(),
                      '_J': len(check['i']) if len(check['i']) != 1 else check['i'][0].count(self.con),}
            if self._M == 0:
                for attr, correct in verify.items():
                    setattr(self, attr, correct)
            else:
                for attr, correct in verify.items():
                    assert getattr(self, attr, None) == correct, \
                        f'{correct[-1]} changed from {getattr(self, attr, None)} to {correct} in {self}.'
            #
            heads = self.heads
            assert heads[1] == check['x'][0].axis, f'Axis[1] should be an x-axis in {self}.'
            if check['i']:
                assert self.isFat ^ (len(check['i']) == 1), f'Only one i-axis allowed in {self} unless self.isFat.'
                assert heads.index(check['i'][0].axis) == heads.index(check['x'][-1].axis) + 1, \
                    f'i-axes should immediately follow x-axes in {self}.'
                assert heads[-3] == check['i'][-1].axis, f'Axis[-3] should be an i-axis in {self}.'
            else:
                assert heads[-3] == check['x'][-1].axis, f'Axis[-3] should be an x-axis in {self}.'
        if write_csv: self._df.write_csv(self._path, **self.writeOptions)
        return self

    @classmethod
    def create(cls, path: PathLike, df: Table | DataFrame, isFat: bool = False) -> Self:
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
        heads = ['n'] + [cls.axisLexicon.get(left.lower(), left.lower()) + con + right
                         for head in df.columns[1: ] for left, con, right in [head.partition(cls.con)]]
        df = df.rename(dict(zip(df.columns, heads)))
        heads = {axisType: [head for head in heads if head.partition(cls.con)[0] == axisType]
                 for axisType in cls.axisTypes}
        y = heads.pop('y', [])
        assert y, f'Design at {path} has no y-axis.'
        if len(heads['l']) > 0:
            # Only accept first output axis
            df = df.with_columns(*[head for axisType in heads.keys() for head in heads[axisType]]
                                 , pl.col(y[0]).alias('y'))
        else:
            df = df.unpivot(y, index=[head for axisType in heads.keys() for head in heads[axisType]],
                            variable_name='l', value_name='y').with_columns(pl.col('l').str.slice(2))
        if isFat:
            assert len(heads['i']), f'A fat Design requires at least one i-axis in {df}.'
            path = Path(str(path) + cls.extFat)
            if set([head.count(cls.con) for head in heads['i']]) != {1}:
                # split heads['i']
                rename = {head: '_' + head for head in heads['i']}
                df = df.rename(rename)
                cats = {'_' + head : head.split(cls.con) for head in heads['i']}
                heads = heads['n'] + heads['x']
                for old, new in cats.items():
                    newHeads = {f'field_{j}': 'i' + cls.con + cat for j, cat in enumerate(new[1:])}
                    df = (df.with_columns(pl.col(old).cast(String).str.split_exact(cls.con, len(new))
                                                  .alias(cls.con*3)).unnest(cls.con*3).rename(newHeads))
                    heads += newHeads.values()
                df = df.select(heads + ['l', 'y'])
        elif len(heads['i']) > 1:
            # join heads['i']
            head = cls.con.join(heads['i'][0:1] + [cat[2:] for cat in heads['i'][1:]])
            df = df.with_columns(pl.concat_str([pl.col(cat) for cat in heads['i']],
                                               separator=cls.con).alias(head))
            df = df.select(heads['n'] + heads['x'] + [head, 'l', 'y'])
        return cls(path, df)


class Stats(Table):
    """ The Statistics of a Design."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a Design. """

    axisTypes: list[str] = ['x', 'j', 'i', 'l', 'y']
    """ The axisTypes in any Design, ordered from left to right. """

    stats: list[Callable[[], DataFrame]] = [pl.min, pl.max, pl.mean, pl.std, pl.count]
    """ The stats collected for any Design. """

    @property
    def M(self) -> int:
        """ Counts the number of continuous inputs. """
        return self._M

    @property
    def L(self) -> int:
        """ Counts the number of output axes. """
        return self._L

    @property
    def x(self) -> Design.Slice:
        return Design.Slice(axes=slice(None, self._M * len(self.stats)))

    @property
    def j(self) -> Design.Slice:
        xAxes = self._M * len(self.stats)
        return Design.Slice(axes=slice(xAxes, xAxes + 1))

    @property
    def i(self) -> Design.Slice:
        xAxes = self._M * len(self.stats)
        return Design.Slice(axes=slice(xAxes + 1, xAxes + 2))
        return Design.Slice(axes=slice(self._M * len(self.stats) + 1, self._M * len(self.stats)) + 2)

    @property
    def l(self) -> Design.Slice:
        yAxes = len(self.stats)
        return Design.Slice(axes=slice(-yAxes - 1, -yAxes))

    @property
    def y(self) -> Design.Slice:
        return Design.Slice(slice(-len(self.stats), None))

    @property
    def means(self) -> Design.Slice:
        return Design.Slice(rows=slice(-self._L, None), axes=slice(None, None))

    def yPivot(self, path: Path, yAxisPrefix: str = f'y{Table.con}', **kwargs) -> Table:
        """ Create a Table at ``path`` consisting of ``self`` with 'y' values pivoted on the 'y-axis' axis.

        Args:
            path: The Path to store this Table, overwritten if existing.
            yAxisPrefix: The prefix of the new 'y' axes, to appear before the 'y-axis' axis values.
                The default is 'y│', pass to ``''`` for no prefix.
            **kwargs: KeywordArguments passed directly to `DataFrame()`_.

        Returns: A Table with no 'y-axis' axis but several output axes in place of the 'y' axis.

        .. DataFrame(): https://docs.pola.rs/api/python/dev/reference/dataframe/index.html
        """
        return Table.create(path, (self._df.with_columns((pl.lit(yAxisPrefix) + pl.col('y-axis')).alias('y-axis'))
                                   .pivot('y-axis', values='y')), **kwargs)

    def __call__(self, update: Self | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        schema = tuple(((head.partition(self.con)[0], _type) for head, _type in self.schema.items()))
        # Verify x, y
        assert set(schema[self.x.axes]) == ('x', Float), f'{self}.x schema violation.'
        assert set(schema[self.y.axes]) == ('y', Float), f'{self}.y schema violation.'

    @classmethod
    def create(cls, design: Design) -> Self:
        """ Create a Distribution from a Design.

        Args:
            design: The Design described by this Distribution.

        Returns: The Distribution created.
        """
        path = design.path.with_name(f'{design.path.name}.stats')

        aggregate = {head + cls.con + stat.__name__: stat(head)
                     for head in design.heads[design.x.axes] + design.heads[design.y.axes]
                     for stat in cls.stats}
        df = design.df.group_by('o').agg(**aggregate)
        return cls(path, df)
