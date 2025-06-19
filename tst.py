#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2025 Robert A. Milton
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

""" Unit Tests for the RomCom Library. """

from __future__ import annotations

from rc.data.models import *

import unittest as ut
import inspect

ROOT = Path('../tst')


class TestCase(ut.TestCase):

    @classmethod
    def root(cls) -> Path:
        return ROOT / cls.__name__.lower() / inspect.stack()[0][3][5:]


class Data(TestCase):

    def setUp(self):
        self.src = DesignMatrix(self.root() / 'src')

    def test_DesignMatrix(self):
        print(self.src.pd)


# class Toy(DataBase):
#     class Tables(Tables):
#         class NT(NamedTuple):
#             data: Table | Matrix | MetaData = pd.DataFrame(data=[[0, 0, 0]],
#                                                 columns=pd.MultiIndex.from_tuples((('Input', 'float'), ('Category', 'int'), ('Output', 'float'))))
#             def __call__(self, field: str) -> Table | Matrix | MetaData:
#                 return getattr(self, field)
#         options: NT[MetaData] = NT(data =  {'header': [0, 1]})
#     defaultMetaData: MetaData = {'Tables': Tables.options._asdict()}
#

if __name__ == '__main__':
    unittest.main()
