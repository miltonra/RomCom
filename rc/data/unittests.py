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

    def setUp(self):
        self.src = DesignMatrix.copy(DesignMatrix(Test.folder() / 'src'), Test.folder() / 'src')
        self.srcmo = DesignMatrix.copy(DesignMatrix(Test.folder() / 'srcmo'), Test.folder() / 'srcmo')
        self.srcco = DesignMatrix.copy(DesignMatrix(Test.folder() / 'srcco'), Test.folder() / 'srcco')
        print(self.src.pd)

    def test_StateMatrix(self):
        stateMatrix = StateMatrix.create(Test.folder() / 'src', self.src)
        stateMatrix = StateMatrix.create(Test.folder() / 'srcmo', self.srcmo)
        stateMatrix = StateMatrix.create(Test.folder() / 'srcco', self.srcco)

    def test_DesignMatrix(self):
        src = StateMatrix(Test.folder().parent / 'StateMatrix' / 'src')
        designMatrix = DesignMatrix.create(Test.folder() / 'src', src)
        self.assertEqual(designMatrix, self.src)
        srcmo = StateMatrix(Test.folder().parent / 'StateMatrix' / 'srcmo')
        designMatrix = DesignMatrix.create(Test.folder() / 'srcmo', srcmo)
        self.assertEqual(designMatrix, self.srcmo)
        srcco = StateMatrix(Test.folder().parent / 'StateMatrix' / 'srcco')
        designMatrix = DesignMatrix.create(Test.folder() / 'srcco', srcco)
        self.assertEqual(designMatrix, self.srcco)


if __name__ == '__main__':
    ut.main()
