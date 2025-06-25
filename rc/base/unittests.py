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

import pandas as pd

from rc.data.models import *

class TestCase(ut.TestCase):

    def test_Meta(self):
        try:
            empty = Meta.create(Test.folder() / 'empty')
            raise Exception('Meta.create should not create an empty MetaData object')
        except Exception:
            pass
        created = Meta.create(Test.folder() / 'created', first=0, second=1)
        read = Meta(Test.folder() / 'created')
        self.assertEqual(created, read)
        copied = Meta.copy(src=read, dst=Test.folder() / 'copied')
        self.assertEqual(copied, read)
        self.assertNotEqual(copied.path, read.path)
        mangled = Meta.copy(src=copied, dst=Test.folder() / 'mangled')
        mangled.delete(mangled.path)
        created(update=1)
        self.assertNotEqual(created, read)
        created['update'] = 2
        read = Meta(Test.folder() / 'created')
        self.assertEqual(read['update'], 2)
        self.assertEqual(len(read), 3)

    def test_Table(self):
        try:
            empty = Meta.create(Test.folder() / 'empty')
            raise Exception('Table.create should not create an empty MetaData object')
        except Exception:
            pass
        created = Table.create(Test.folder() / 'created', np.zeros((1, 1)))
        for i in range(2):
            value = np.ones((1, 1)) * i
            shouldBe = self.assertNotEqual if i else self.assertEqual
            shouldBe(created, pd.DataFrame(value, columns=['0']))
            shouldBe(created, value)
            shouldBe(created, tc.tensor(value))
        read = Table(Test.folder() / 'created')
        self.assertEqual(created, read)
        copied = Table.copy(src=read, dst=Test.folder() / 'copied')
        mangled = Table.copy(src=copied, dst=Test.folder() / 'mangled')
        mangled.delete(mangled.path)
        created(tc.tensor([1]))
        self.assertNotEqual(created, read)
        read = Table(Test.folder() / 'created')
        self.assertEqual(created, read)

    # @self.skip('Not yet')
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

        created = MyDataBase.create(Test.folder() / 'created')
        read = MyDataBase(Test.folder() / 'created')
        copied = MyDataBase.copy(src=read, dst=Test.folder() / 'copied')
        mangled = MyDataBase.copy(src=copied, dst=Test.folder() / 'mangled'/ 'mangled')
        mangled.meta.delete(mangled.meta.path)
        deleted = MyDataBase.copy(src=copied, dst=Test.folder() / 'mangled')
        deleted.delete(deleted.path)


if __name__ == '__main__':
    self.main()
