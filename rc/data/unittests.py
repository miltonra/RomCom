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

from rc.data.normalizers import *


class TestCase(ut.TestCase):

    def setUp(self):
        self.src = CoordDesign(Test.folder() / 'src')
        self.srcmo = CoordDesign(Test.folder() / 'srcmo')
        self.srcco = CoordDesign(Test.folder() / 'srcco')
        self.basic = CoordDesign(Test.folder() / 'basic')
        self.srcNorm = CoordDesign(Test.folder() / 'srcNorm')
        self.srcsoNorm = CoordDesign(Test.folder() / 'srcsoNorm')
        self.srcmoNorm = CoordDesign(Test.folder() / 'srcmoNorm')
        self.srccoNorm = CoordDesign(Test.folder() / 'srccoNorm')
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

    def test_Normalizer(self):
        basic = Normalizer.create(Test.folder() / 'basic', self.basic)
        srcNorm = Normalizer.create(Test.folder() / 'srcNorm', self.srcNorm)
        srcsoNorm = Normalizer.create(Test.folder() / 'srcsoNorm', self.srcsoNorm)
        srcmoNorm = Normalizer.create(Test.folder() / 'srcmoNorm', self.srcmoNorm)
        srccoNorm = Normalizer.create(Test.folder() / 'srccoNorm', self.srccoNorm)

if __name__ == '__main__':
    ut.main()
