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
        self.src = CoordDesign.copy(CoordDesign(Test.folder() / 'src'), Test.folder() / 'src')
        self.srcmo = CoordDesign.copy(CoordDesign(Test.folder() / 'srcmo'), Test.folder() / 'srcmo')
        self.srcco = CoordDesign.copy(CoordDesign(Test.folder() / 'srcco'), Test.folder() / 'srcco')
        print(self.src.pd)

    def test_PointDesign(self):
        pointDesign = PointDesign.create(Test.folder() / 'src', self.src)
        pointDesign = PointDesign.create(Test.folder() / 'srcmo', self.srcmo)
        pointDesign = PointDesign.create(Test.folder() / 'srcco', self.srcco)

    # @ut.skip('CoordDesign is not valid')
    def test_CoordDesign(self):
        src = PointDesign(Test.folder().parent / 'PointDesign' / 'src')
        coordDesign = CoordDesign.create(Test.folder() / 'src', src)
        self.assertEqual(coordDesign, self.src)
        srcmo = PointDesign(Test.folder().parent / 'PointDesign' / 'srcmo')
        coordDesign = CoordDesign.create(Test.folder() / 'srcmo', srcmo)
        self.assertEqual(coordDesign, self.srcmo)
        srcco = PointDesign(Test.folder().parent / 'PointDesign' / 'srcco')
        coordDesign = CoordDesign.create(Test.folder() / 'srcco', srcco)
        self.assertEqual(coordDesign, self.srcco)


if __name__ == '__main__':
    ut.main()
