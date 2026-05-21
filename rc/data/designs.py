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


#: Slice for ``x`` (continuous inputs) in a PointDesign.
x: tuple[slice, slice] = (slice(None, None), slice(1, -2))

#: Slice for ``y`` (output) in a PointDesign.
y: tuple[slice, slice] = (slice(None, None), slice(-1, None))


class Design(Table):
    """ A Design of user data, tabulating continuous inputs, categorical inputs, and outputs."""

    class CreateP(CreateP):
        """ Creates a new instance of ``cls`` at ``path`` from ``design: PointDesign | CoordDesign``. """

    coordSeparator: str = '│'
    """ The separator used to delimit coords in a column of categorical points. """

    outputAxis: str = 'ο'
    """ The label used to denote the output axis in a PointDesign. """

    @classmethod
    def axisType(cls, axis: str) -> str:
        """ The axisType of the given ``axis``.

        Args:
            axis: A ``str`` axisType, such as ``'x', 'i', 'y', 'continuous', 'discrete', 'output', etc.

        Returns: ``axisType in 'x', 'i', 'y', '?'``. ``'?'`` is returned if ``axis`` is not recognized.

        """
        match axis.lower():
            case 'x' | 'x│' | 'input' | 'in' | 'continuous' | 'float' :
                return 'x│'
            case 'i' | 'i│' | 'category' | 'cat' | 'discrete' | 'int' | 'str' :
                return 'i│'
            case 'y' | 'y│' | 'output' | 'out'| 'map' | 'func' :
                return 'y│'
        return '?│'

    @classmethod
    @abstractmethod
    def create(cls, path: PathLike, design: Design) -> Self:
        """ Create a ``Design`` at ``path``.

        Args:
            path: The ``Path`` to store this Table, overwritten if existing.
                A ``.csv`` extension is automatically appended.
            design: The ``CoordDesign | PointDesign`` to reformat if necessary and store in ``path``.

        Returns: The ``Design`` created.
        """


class PointDesign(Design):
    """ The internal format of ``Design``, which is thin (has few columns), and only one header row.
    Categorical axes are concatenated into a single column of categorical points."""

    @classmethod
    def create(cls, path: PathLike, design: Design) -> Self:
        match design:
            case PointDesign():
                # If already a PointDesign, just copy it.
                pointDesign = design.pd.copy(deep=True)
            case CoordDesign():
                # Reformat the CoordDesign to a PointDesign.
                coordDesign = design.pd.copy(deep=True).rename(cls.axisType, axis='columns', level=0)
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
