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

""" Design Matrix formats. """

from rc.base import *


#: Slice for ``n`` (row index) in a Point Design
n: int = 0

#: Slice for ``x`` (continuous inputs) in a PointDesign.
x: tuple[slice, slice] = (slice(None, None), slice(1, -2))

#: Slice for ``y`` (outputs) in a PointDesign.
y: tuple[slice, slice] = (slice(None, None), slice(-1, None))


class Design(Table):
    """ A Design of user data, tabulating continuous inputs, categorical inputs, and outputs."""

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

    def __call__(self, update: Self | Matrix | None = None) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            update: The data updates.

        Returns: ``self``.
        """


    def create(cls, path: PathLike, table: Table) -> Design:
        """ Create a ``Design`` at ``path``.

        Args:
            path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            table: The ``Table`` to reformat and store in ``path``.
        Returns: The ``Table`` created.
        """
        heads = table.heads
        heads[0] = 'n' + cls.con
        heads = [head.split(cls.con,1) for head in heads]
        heads = [cls.axisLexicon.get(head[0].lower(), head[0]) + cls.con + head[1] for head in heads]
        df = table.df.rename(dict(zip(table.heads, heads)))
        heads = {axisType: [head for head in heads if head[0] == axisType] for axisType in cls.axisTypes}
        if y := heads.pop('y', []):
            if len(heads['o']) > 0:
                # Only accept first output column
                df = df.with_columns(*[head for axisType in heads.keys() for head in heads[axisType]]
                                     , pl.col(y[0]).alias('y'))
            else:
                df = df.unpivot(y, index=[head for axisType in heads.keys() for head in heads[axisType]],
                                variable_name='o', value_name='y')
        else:
            raise ValueError('Design must have at least one output axis.')

        return (Table(Table.mkdir(path), df),
                {axisType: len(heads.get(axisType,[])) for axisType in cls.axisTypes})


class Design0(Table):
    """ A Design of user data with 0 categorical inputs."""

    axisLexicon: dict[str,str] = {'index': 'n',
                                  'input': 'x', 'in': 'x', 'continuous': 'x', 'float': 'x',
                                  'category': 'i', 'cat': 'i', 'discrete': 'i', 'int': 'i', 'str': 'i',
                                  'outputaxis': 'o', 'outputindex': 'o', 'outputcategory': 'o',
                                  'output': 'y', 'out':'y', 'map': 'y', 'func':'y', }
    """ The lexicon of axisTypes. """

    axisTypes: list[str] = list(dict.fromkeys(axisLexicon.values()))
    """ The axisTypes in any Design, ordered from left to right. """

    def __call__(self, update: Self | Matrix | None = None) -> Self:
        """ Update and store ``self``, overwriting.

        Args:
            update: The data updates.

        Returns: ``self``.
        """


    def create(cls, path: PathLike, table: Table) -> Design:
        """ Create a ``Design`` at ``path``.

        Args:
            path: The Path to store this Table, overwritten if existing.
                A ``.csv`` extension is implicitly appended.
            table: The ``Table`` to reformat and store in ``path``.
        Returns: The ``Table`` created.
        """
        heads = table.heads
        heads[0] = 'n' + cls.con
        heads = [head.split(cls.con,1) for head in heads]
        heads = [cls.axisLexicon.get(head[0].lower(), head[0]) + cls.con + head[1] for head in heads]
        df = table.df.rename(dict(zip(table.heads, heads)))
        heads = {axisType: [head for head in heads if head[0] == axisType] for axisType in cls.axisTypes}
        if y := heads.pop('y', []):
            if len(heads['o']) > 0:
                # Only accept first output column
                df = df.with_columns(*[head for axisType in heads.keys() for head in heads[axisType]]
                                     , pl.col(y[0]).alias('y'))
            else:
                df = df.unpivot(y, index=[head for axisType in heads.keys() for head in heads[axisType]],
                                variable_name='o', value_name='y')
        else:
            raise ValueError('Design must have at least one output axis.')

        return (Table(Table.mkdir(path), df),
                {axisType: len(heads.get(axisType,[])) for axisType in cls.axisTypes})


class PointDesign(Design):
    """ The internal format of ``Design``, which is narrow.
    Categorical axes are concatenated into a single column of categorical points."""

    @classmethod
    def create(cls, path: PathLike, design: Design) -> Self:
        coordDesign = super().create(path, design)
        match design:
            case PointDesign():
                return cls(path)
            case CoordDesign():
                # Reformat the CoordDesign to a PointDesign.
                # Strip out the ``inputAxes`` from ``design``.
                inputAxes = {'x│': None, 'i│': None}
                for key in inputAxes.keys():
                    inputAxes[key] = coordDesign.get(key)
                    if inputAxes[key] is not None: coordDesign.drop(columns=key, level=0, inplace=True)
                coordDesign.columns = coordDesign.columns.droplevel(0)
                # Return the categorical inputs to ``design``.
                if inputAxes['i│'] is not None:
                    pointCoord = cls.coordSeparator.join(['ο'] + inputAxes['i│'].columns.to_list())
                    coordDesign[pointCoord] = inputAxes['i│'].astype(str).agg(cls.coordSeparator.join, axis=1)
                    pointDesign = coordDesign.reset_index(names='n│').melt(id_vars=['n│', pointCoord],
                                                                          var_name='ο',
                                                                          value_name='y│',
                                                                          ignore_index=True).dropna()
                    pointDesign[pointCoord] = (pointDesign
                                              ['ο'].astype(str) + cls.coordSeparator +
                                               pointDesign[pointCoord])
                    pointDesign.drop(columns=['ο'], inplace=True)
                else:
                    pointCoord = 'ο'
                    pointDesign = coordDesign.reset_index(names='n│').melt(id_vars=['n│'], var_name='ο',
                                                                          value_name='y│',
                                                                          ignore_index=True).dropna()
                # Return the continuous inputs to ``design``.
                if inputAxes['x│'] is not None:
                    columns = ['n│'] + inputAxes['x│'].columns.to_list() + [pointCoord, 'y│']
                    pointDesign = pointDesign.join(inputAxes['x│'], on='n│', how='left').reindex(columns=columns)
            case _:
                raise NotImplementedError(f'I do not know how to create a PointDesign from {type(design)}')
        return cls(cls.mkdir(path), pointDesign)


class CoordDesign(Design):
    """ The familiar user format of a Design which has many axes (columns), and two header rows.
    The first header row contains the axisType, the second header row contains the axis."""

    readOptions: MetaData = Table.readOptions | {'header': [0, 1]}
    """ File read options passed directly to
    `pd.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`__."""

    @classmethod
    def create(cls, path: PathLike, design: Design) -> Self:
        match design:
            case CoordDesign():
                # Reformat colum labels and categorical coords to strings.
                coordDesign = design.pd.copy(deep=True).rename(cls.axisType, axis='columns', level=0)
                if 'i│' in coordDesign.columns.get_level_values(0):
                    coordDesign['i│'] = coordDesign['i│'].astype(str)
            case PointDesign():
                # Reformat the PointDesign to an CoordDesign.
                pointDesign = design.pd.copy(deep=True).set_index('n│')
                pointDesign[pointDesign.columns[-2]] = pointDesign[pointDesign.columns[-2]].astype(str)
                # convert Points to coords.
                iDesign = pointDesign.iloc[:, -2].apply(lambda point: point.split(cls.coordSeparator))
                iDesign = pd.DataFrame(iDesign.tolist(), index=pointDesign.index,
                                       columns=iDesign.name.split(cls.coordSeparator))
                # convert first categorical variable to output axes.
                yAxes = iDesign['ο'].drop_duplicates().tolist()
                yDesign = pd.concat([iDesign['ο'], pointDesign.iloc[:, -1]], axis=1)
                for yCoord in yAxes:
                    yDesign[yCoord] = (yDesign['ο'] == yCoord).astype(int) * yDesign.iloc[:, 1]
                # Collect x, i, and y column groups.
                coordDesign = pd.concat([pointDesign.iloc[:, :-2].groupby(level=0, sort=False).mean(),
                                         iDesign.iloc[:, 1:].groupby(level=0, sort=False).first(),
                                         yDesign.iloc[:, 2:].groupby(level=0, sort=False).sum(), ],
                                        axis=1)
                # Label axes correctly.
                coordDesign.columns = pd.MultiIndex.from_tuples(
                    [('x│', col) for col in pointDesign.columns[:-2]] +
                    [('i│', col) for col in iDesign.columns[1:]] +
                    [('y│', col) for col in yAxes])
                # Detect NaN.
                coordDesign['y│'] = coordDesign['y│'].replace(0.0, np.nan)
            case _:
                raise NotImplementedError(f'I do not know how to create an CoordDesign from {type(design)}')
        return cls(cls.mkdir(path), coordDesign)
