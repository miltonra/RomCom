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

from rc.data.designs import *


class TestCase(ut.TestCase):

    def setUp(self):
        self.table0 = Table.conjoinHeads(Test.folder() / 'experiment.0', Test.folder() / 'table.0')
        self.table = Table.conjoinHeads(Test.folder() / 'experiment', Test.folder() / 'table')

    # @ut.skip('CoordDesign is not valid')
    def test_Design(self):
        design0 = Design.create(Test.folder() / 'design.0', self.table0)
        yPivot0 = design0.yPivot()
        design = Design.create(Test.folder() / 'design', self.table)
        yPivot = design.yPivot()
        self.assertRaises(AssertionError, Design.create, Test.folder() / 'design.0', self.table0, True)
        designF = Design.create(Test.folder() / 'design.fat', self.table, isFat=True)
        yPivotF = designF.yPivot()
        self.assertEqual(design0, Design(Test.folder() / 'design.0'))
        self.assertEqual(design0, Design.create(Test.folder() / 'design.0', yPivot0))
        self.assertEqual(design, Design(Test.folder() / 'design'))
        self.assertEqual(design, Design.create(Test.folder() / 'design', yPivot))
        self.assertEqual(design, Design.create(Test.folder() / 'design', designF))
        self.assertEqual(designF, Design(Test.folder() / f'design.fat'))
        self.assertEqual(designF, Design.create(Test.folder() / 'design.fat.y', yPivotF, isFat=True))
        self.assertEqual(designF, Design.create(Test.folder() / 'design.fat', design, isFat=True))

    def test_Measure(self):
        design = Design.create(Test.folder() / 'design.0', self.table0)
        measure = Measure.create(Test.folder() / 'measure.0', design)
        yPivot = measure.yPivot()
        xAxes = measure.xAxes(Test.folder() / 'xAxes.0')
        for isFat in (False, True):
            fat = ".fat" if isFat else ""
            design = Design.create(Test.folder() / f'design{fat}', self.table, isFat=isFat)
            measure = Measure.create(Test.folder() / f'measure{fat}', design)
            yPivot = measure.yPivot()
            xAxes = measure.xAxes(Test.folder() / f'xAxes{fat}')


if __name__ == '__main__':
    ut.main()
