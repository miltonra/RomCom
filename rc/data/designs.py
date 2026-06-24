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

    @property
    def M(self) -> int:
        """ Counts the number of continuous inputs. """
        return self._M

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

    def yPivot(self, path: Path, yAxisPrefix: str = f'y{Table.con}', **kwargs) -> Table:
        """ Create a Table at ``path`` consisting of ``self`` with 'y' values pivoted on the 'l' axis.

        Args:
            path: The Path to store this Table, overwritten if existing.
            yAxisPrefix: The prefix of the new 'y' axis, to appear before the 'l' axis values.
                The default is 'y│', pass to ``''`` for no prefix.
            **kwargs: KeywordArguments passed directly to `DataFrame()`_.

        Returns: A Table with no 'l' axis but several output axis in place of the 'y' axis.

        .. DataFrame(): https://docs.pola.rs/api/python/dev/reference/dataframe/index.html
        """
        return Table.create(path, (self._df.with_columns((pl.lit(yAxisPrefix) + pl.col('l')).alias('l'))
                                   .pivot('l', values='y')), **kwargs)

    def __call__(self, update: Self | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        schema = tuple(self.AxisVerifier(*schema) for schema in self.schema.items())
        # Verify singleton axisTypes
        verify = {'n': schema[0],
                 'l': schema[len(schema)-2],
                 'y': schema[len(schema)-1],}
        for axisType, actual in verify.items():
            assert axisType == actual.axis, f'Axis {actual.axis} should be {axisType}'
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
        self._M = len(check['x'])
        self._L = self._df.select(pl.col('l').n_unique()).item()
        #
        heads = self.heads
        assert heads[1] == check['x'][0].axis, f'Axis[1] should be an x-axis'
        if check['i']:
            assert heads.index(check['i'][0].axis) == heads.index(check['x'][-1].axis) + 1, \
                f'i-axes should immediately follow x-axes'
            assert heads[-3] == check['i'][-1].axis, f'Axis[-3] should be an i-axis'
        else:
            assert heads[-3] == check['x'][-1].axis, f'Axis[-3] should be an x-axis'
        #
        if write_csv:
            self._df.write_csv(self._path, **self.writeOptions)
        return self

    @classmethod
    def create(cls, path: PathLike, tableData: Table | TableData, **kwargs: Any) -> Self:
        table = Table.create(path, tableData, **kwargs)
        heads = ['n'] + [cls.axisLexicon.get(left.lower(), left.lower()) + con + right
                         for head in table.heads[1: ] for left, con, right in [head.partition(cls.con)]]
        df = table.df.rename(dict(zip(table.heads, heads)))
        heads = {axisType: [head for head in heads if head.partition(cls.con)[0] == axisType]
                 for axisType in cls.axisTypes}
        if y := heads.pop('y', []):
            if len(heads['l']) > 0:
                # Only accept first output axis
                df = df.with_columns(*[head for axisType in heads.keys() for head in heads[axisType]]
                                     , pl.col(y[0]).alias('y'))
            else:
                df = df.unpivot(y, index=[head for axisType in heads.keys() for head in heads[axisType]],
                                variable_name='l', value_name='y').with_columns(pl.col('l').str.slice(2))
        else:
            raise ValueError('Design must have at least one output axis.')
        return cls(table.path, df)


class Design0(Design):
    """ A Design of user data with 0 categorical inputs."""

    def __call__(self, update: Self | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv)
        assert len(self) == 1 + self._M + 0 + 2, (f'Too many input axes: '
                                                  f'{len(self)-3} provided when self.M={self._M}')
        return self


class Design1(Design):
    """ A conjoined Design of user data with 1 categorical input."""

    def __call__(self, update: Self | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv)
        assert len(self) == 1 + self._M + 1 + 2, (f'Too many input axes: '
                                                  f'{len(self)-4} provided when self.M={self._M}')
        return self

    @classmethod
    def create(cls, path: PathLike, tableData: Table | TableData, **kwargs: Any) -> Self:
        d = Design.create(path, tableData, **kwargs)
        cats = d.heads[d.i.axes]
        assert cats, f'Cannot create Design1 from TableData with no i-axes. Use Design0.create() instead.'
        # join d
        head = cls.con.join(cats[0:1] + [cat[2:] for cat in cats[1:]])
        update = d.df.with_columns(pl.concat_str([pl.col(cat) for cat in cats], separator=cls.con)
                                   .alias(head))
        heads = d.heads[0 : d.M + 1] + [head, 'l', 'y']
        update = update.select(heads)
        return cls(d.path, update)


class DesignS(Design):
    """ An unjoined Design of user data with every possible categorical input."""

    def __call__(self, update: Table | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv)
        assert set([head.count(self.con)
                    for head in self.heads[self.i.axes]]) == {1}, f'Category heads contain{self.con}'
        return self

    @classmethod
    def create(cls, path: PathLike, tableData: Table | TableData, **kwargs: Any) -> Self:
        d = Design.create(path, tableData, **kwargs)
        cats = list(d.heads[d.i.axes])
        if set([head.count(d.con) for head in cats]) == {1}:
            update = None
        else:
            # split d
            rename = {cat: '_' + cat for cat in cats}
            update = d.df.rename(rename)
            cats = {'_' + cat : cat.split(cls.con) for cat in cats}
            heads = d.heads[0:d.M+1]
            for old, new in cats.items():
                newHeads = {f'field_{j}': 'i' + cls.con + cat for j, cat in enumerate(new[1:])}
                update = (update.with_columns(pl.col(old).cast(String).str.split_exact(cls.con, len(new))
                                              .alias(cls.con*3)).unnest(cls.con*3).rename(newHeads))
                heads += newHeads.values()
            update = update.select(heads + ['l', 'y'])
        return cls(d.path, update)


class Statistics(Table):
    """ The Statistics of a *Design*."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from a Design0, Design1 or DesignS. """

    axisTypes: list[str] = ['x', 'i-axis', 'i', 'y-axis', 'y']
    """ The axisTypes in any *Design*, ordered from left to right. """

    stats: list[Callable[[], DataFrame]] = [pl.min, pl.max, pl.mean, pl.std, pl.count]
    """ The stats collected for any *Design*. """

    @property
    def M(self) -> int:
        """ Counts the number of continuous inputs. """
        return self._M

    @property
    def L(self) -> int:
        """ Counts the number of output axes. """
        return self._L

    @property
    def J(self) -> int:
        """ Counts the number of output axes. """
        return self._J

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
        assert isinstance(design, (Design0, Design1, DesignS)), (f'{type(design)} not in '
                                                                 f'(Design0, Design1, DesignS).')
        path = design.path.with_name(f'{design.path.name}.stats')

        aggregate = {head + cls.con + stat.__name__: stat(head)
                     for head in design.heads[design.x.axes] + design.heads[design.y.axes]
                     for stat in cls.stats}
        df = design.df.group_by('o').agg(**aggregate)
        return cls(path, df)
