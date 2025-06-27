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
            empty = Table.create(Test.folder() / 'empty')
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

    def test_DataBase(self):
        class MyDataBase(DataBase):
            class NamedTables(NamedTuple):
                zero: Table | Matrix | MetaData = pd.DataFrame(np.atleast_2d(0.0))
                one: Table | Matrix | MetaData = pd.DataFrame(np.ones((1, 1)))

                def __call__(self, name: str) -> Table | Matrix | MetaData:
                    return getattr(self, name)

            options: NamedTables = NamedTables(zero=Table.Options.defaults(),
                                                         one=Table.Options.defaults())

            defaultMetaData: MetaData = {'options': options._asdict()}

        empty = MyDataBase.create(Test.folder() / 'default')
        created = MyDataBase.create(Test.folder() / 'created', zero = np.zeros((1, 1)), one = np.ones((1, 1)))
        for i in range(2):
            value = np.ones((1, 1)) * i
            self.assertEqual(created[i], pd.DataFrame(value, columns=['0']))
            self.assertNotEqual(created[abs(i-1)], pd.DataFrame(value, columns=['0']))
            self.assertEqual(created[i], value)
            self.assertNotEqual(created[abs(i-1)], value)
            self.assertEqual(created[i], tc.tensor(value))
            self.assertNotEqual(created[abs(i-1)], tc.tensor(value))
        read = MyDataBase(Test.folder() / 'created')
        self.assertEqual(created, read)
        copied = MyDataBase.copy(src=read, dst=Test.folder() / 'copied')
        mangled = MyDataBase.copy(src=copied, dst=Test.folder() / 'mangled')
        Meta.copy(src=mangled.meta, dst=mangled.meta.path / 'copied')
        mangled.delete(mangled.path)
        created(zero=tc.tensor(1.0), one=tc.tensor(2.0))
        self.assertNotEqual(created, read)
        read = MyDataBase(Test.folder() / 'created')
        self.assertEqual(created, read)
        created['zero'] = tc.tensor(2.0)
        created['one'] = tc.tensor(1.0)
        self.assertNotEqual(created, read)
        read = MyDataBase(Test.folder() / 'created')
        self.assertEqual(created, read)
        created[:] = copied[:]
        self.assertEqual(created, copied)
        self.assertNotEqual(created, read)
        created(**read.namedTables._asdict())
        self.assertEqual(created, read)
        created[:] = copied['zero'], copied['one']
        self.assertEqual(created, copied)
        self.assertNotEqual(created, read)
        deleted = MyDataBase.copy(created, Test.folder() / 'deleted')
        deleted.delete(deleted.path)

if __name__ == '__main__':
    ut.main()
