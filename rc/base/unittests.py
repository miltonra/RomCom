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

from __future__ import annotations

from rc.data.models import *

class TestCase(ut.TestCase):

    def test_Meta(self):
        meta = Meta.create(Test.folder() / 'meta', **Table.Options.defaults())
        meta(update=1)

    def test_Table(self):
        table = Table.create(Test.folder() / 'table', pd.DataFrame([0]))
        table(pd.DataFrame([1]))

    # @ut.skip('Not yet')
    def test_DataBase(self):
        class MyDataBase(DataBase):
            class NamedTables(NamedTuple):
                zero: Table | Matrix | MetaData = pd.DataFrame([0])
                one: Table | Matrix | MetaData = pd.DataFrame([1])

                def __call__(self, name: str) -> Table | Matrix | MetaData:
                    return getattr(self, name)

            options: NamedTables[MetaData] = NamedTables(zero=Table.Options.defaults(),
                                                         one=Table.Options.defaults())

            defaultMetaData: MetaData = {'options': options._asdict()}

        object = MyDataBase.create(Test.folder() / 'object.path.name')
        ambition = MyDataBase.create(Test.folder() / 'object.path.name')

if __name__ == '__main__':
    ut.main()
