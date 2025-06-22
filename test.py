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
        caller = inspect.stack()[1][3]
        if caller.startswith("test_"): caller = caller[5:]
        return ROOT / cls.__name__.lower() / caller


class Base(TestCase):

    def test_Meta(self):
        meta = Meta.create(self.root() / 'meta', **Table.Options.defaults())
        meta(update=1)

    def test_Table(self):
        Store.delete(self.root())
        table = Table.create(self.root() / 'table', pd.DataFrame([0]))
        table(pd.DataFrame([1]))

    def test_DataBase(self):
        class MyDataBase(DataBase):
            class NamedTables(NamedTuple):
                zero: Table | Matrix | MetaData = pd.DataFrame([0])
                one: Table | Matrix | MetaData = pd.DataFrame([1])

                def __call__(self, name: str) -> Table | Matrix | MetaData:
                    """ Returns the Table named ``name``."""
                    return getattr(self, name)

            options: NamedTables[MetaData] = NamedTables(zero=Table.Options.defaults(),
                                                         one=Table.Options.defaults())
            """ Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
            Override as necessary for bespoke ``Table.options``.
            Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
            the remainder populate ``self[i].options.read``."""

            defaultMetaData: MetaData = {'options': options._asdict()}

        object = MyDataBase.create(self.root() / 'object.path.name')
        ambition = MyDataBase.create(self.root() / 'object.path.name')

@ut.skip('Not yet')
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
    ut.main()
