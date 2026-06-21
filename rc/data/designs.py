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
                                  'outputaxis': 'o', 'outputindex': 'o', 'outputcategory': 'o',
                                  'output': 'y', 'out':'y', 'map': 'y', 'func':'y', }
    """ The lexicon of axisTypes. """

    axisTypes: list[str] = list(dict.fromkeys(axisLexicon.values()))
    """ The axisTypes in any Design, ordered from left to right. """

    class AxisVerifier(NamedTuple):
        """ A NamedTuple for verifying axes in a Design. Used exclusively by the ``__call__()`` method."""
        axes: int | list[str] | dict[str, type]     #: The axes to verify.
        Type: type = Category | Float   #: The Type of data accepted by this axisType.

    class Slice(NamedTuple):
        """ A pair of ``slices``, to slice TableData."""
        rows: slice = slice(None, None) #: Rows are the first tensor rank
        axes: slice = slice(None, 1)   #: Axes are the second tensor rank

    @property
    def M(self) -> int:
        """ Counts the number of continuous inputs. """
        return self._M

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
    def o(self) -> Slice:
        return self.Slice(axes=slice(-2, -1))

    @property
    def y(self) -> Slice:
        return self.Slice(axes=slice(-1, None))

    def yPivot(self, path: Path, yAxisPrefix: str = f'y{Table.con}', **kwargs) -> Table:
        """ Create a Table at ``path`` consisting of ``self`` with 'y' values pivoted on the 'o' axis.

        Args:
            path: The Path to store this Table, overwritten if existing.
            yAxisPrefix: The prefix of the new 'y' axes, to appear before the 'o' axis values.
                The default is 'y│', pass to ``''`` for no prefix.
            **kwargs: KeywordArguments passed directly to `DataFrame()`_.

        Returns: A Table with no 'o' axis but several output axes in place of the 'y' axis.

        .. DataFrame(): https://docs.pola.rs/api/python/dev/reference/dataframe/index.html
        """
        return Table.create(path, (self._df.with_columns((pl.lit(yAxisPrefix) + pl.col('o')).alias('o'))
                                   .pivot('o', values='y')), **kwargs)

    def __call__(self, update: Self | TableData | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        schema = self.schema
        # Verify singleton axisTypes
        verify = {'n': self.AxisVerifier(axes=0, Type=Category),
                 'o': self.AxisVerifier(axes=len(schema) - 2, Type=Category),
                 'y': self.AxisVerifier(axes=len(schema) - 1, Type=Float),}
        actual = {axisType: self.AxisVerifier(axes=index, Type=_type)
                      for index, (axisType, _type) in enumerate(schema.items()) if axisType in verify}
        for axisType, correct in verify.items():
            assert actual[axisType].axes == correct.axes, f'Axis {actual[axisType].axes} should be {axisType}'
            if actual[axisType].Type != correct.Type:
                self._df = self._df.with_columns(pl.col(axisType).cast(String).cast(correct.Type))
        # Verify non-singleton axisTypes
        verify = {'x': self.AxisVerifier(axes={}, Type=Float),
                  'i': self.AxisVerifier(axes={}, Type=Category),}
        for axisType, correct in verify.items():
            correct.axes.update({head: _type for head, _type in schema.items() if head.startswith(axisType)})
            incorrect = [head for head, _type in correct.axes.items() if _type != correct.Type]
            if incorrect:
                self._df = self._df.with_columns(pl.col(incorrect).cast(String).cast(correct.Type))
            verify[axisType] = list(correct.axes.keys())
        self._M = len(verify['x'])
        #
        heads = self.heads
        assert heads[1] == verify['x'][0], f'Axis[1] should be an x-axis'
        if verify['i']:
            assert heads.index(verify['i'][0]) == heads.index(verify['x'][-1]) + 1, \
                f'i-axes should immediately follow x-axes'
            assert heads[-3] == verify['i'][-1], f'Axis[-3] should be an i-axis'
        else:
            assert heads[-3] == verify['x'][-1], f'Axis[-3] should be an x-axis'
        #
        if write_csv:
            self._df.write_csv(self._path, **self.writeOptions)
        return self

    @classmethod
    def create(cls, path: PathLike, tableData: Table | TableData, **kwargs: Any) -> Self:
        table = Table.create(path, tableData, **kwargs)
        heads = table.heads[1:]
        heads = [cls.axisLexicon.get(left.lower(), left.lower()) + con + right
                 for head in heads for left, con, right in [head.partition(cls.con)]]
        heads.insert(0, 'n')
        df = table.df.rename(dict(zip(table.heads, heads)))
        heads = {axisType: [head for head in heads if head[0] == axisType] for axisType in cls.axisTypes}
        if y := heads.pop('y', []):
            if len(heads['o']) > 0:
                # Only accept first output axis
                df = df.with_columns(*[head for axisType in heads.keys() for head in heads[axisType]]
                                     , pl.col(y[0]).alias('y'))
            else:
                df = df.unpivot(y, index=[head for axisType in heads.keys() for head in heads[axisType]],
                                variable_name='o', value_name='y').with_columns(pl.col('o').str.slice(2))
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
        heads = d.heads[0 : d.M + 1] + [head, 'o', 'y']
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
            update = update.select(heads + ['o', 'y'])
        return cls(d.path, update)

