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

""" Normalizers for Designs. """

from rc.data.designs import *


class IAxes(Table):
    """A Table of i-axis probabilities calculated as ``y│q`` and used as ``q``."""

    @property
    def isFat(self) -> bool:
        """ Whether the Table has a j-axis."""
        return self._isFat

    @property
    def ls(self) -> list[str]:
        """ Lists the y-axes occurring in the l-axis. """
        return self._ls

    @property
    def js(self) -> list[str]:
        """ Lists the i-axes. """
        return self._js

    def __call__(self, update: TableData | Self | None = None, write_csv: bool = True) -> Self:
        super().__call__(update, write_csv=False)
        head = self.head
        self._isFat = head[0] == 'j'
        self._ls = self._df['l'].unique(maintain_order=True).to_list() if head[-3] == 'l' else []
        self._js = (self._df['j'].unique(maintain_order=True).to_list() if self._isFat else
                    self.head[0].split('│')[1:])
        head = ['j'] if self._isFat else [] + (['l'] if self._ls else [])
        qs = ('y│q', 'q')
        check = self._df.group_by(head).agg(pl.col(qs).sum())
        assert check.select(pl.all_horizontal(pl.col(qs).is_close(1.0, abs_tol=zero).all())).item(), \
            (f'IAxes {check.filter(~(pl.all_horizontal(pl.col(qs).is_close(1.0, abs_tol=zero))))}'
             f'do not add up in {self}')
        # for axis in ('y│q', 'q'):
        #     check = self._df.group_by(head).agg(pl.sum(axis))
        #     assert check.select(pl.col(axis).is_close(1.0, abs_tol=zero)).all().item(), \
        #         (f'IAxes {check.filter(~(pl.col(axis).is_close(1.0, abs_tol=zero)))} '
        #          f'do not add up in {self}')
        if write_csv: self._df.write_csv(self._path, **self.writeOptions)
        return self

    @classmethod
    def create(cls, path: PathLike, measure: Measure, lAgg: pl.Expr | None = pl.median) -> Self:
            """ Calculate the i-axis probabilities ``q`` from ``measure``.

            Args:
                path: The Path to store the description.
                measure: The Measure to use.
                lAxis: How to aggregate over the l-axis.
                    If ``None`` no aggregation is performed and the l-axis appears in the result.
            Returns: A Table of probabilities for the i-axis values.
            """
            assert measure._J, f'No i-axis in {measure}.iAxes().'
            head = (['j'] if measure.isFat else []) + measure.head[measure.i.axes] + ['l', 'y│q']
            result = measure._df.select(head)
            if lAgg:
                result = (result.drop('l').filter(pl.col(head[0]) != '││')
                          .group_by(head[:-2], maintain_order=True).agg(lAgg('y│q')))
            return cls(path, result.with_columns(pl.col('y│q').alias('q')))
