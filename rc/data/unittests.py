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
        self.table = Table.conjoinHeads(Test.folder() / 'experiment', Test.folder() / 'table')
        self.table0 = Table.conjoinHeads(Test.folder() / 'experiment.0', Test.folder() / 'table.0')
        self.design = Design.create(Test.folder() / 'design', self.table)
        self.design0 = Design.create(Test.folder() / 'design.0', self.table0)

    # @ut.skip('CoordDesign is not valid')
    def test_Design(self):
        self.yPivot = Design.yPivot(self.design, Test.folder() / 'yPivot')
        self.assertEqual(self.design, Design.create(Test.folder() / 'design', self.yPivot))
        self.assertEqual(self.design, Design(Test.folder() / 'design'))
        self.yPivot0 = Design.yPivot(self.design0, Test.folder() / 'yPivot.0')
        self.assertEqual(self.design0, Design.create(Test.folder() / 'design.0', self.yPivot0))
        self.assertEqual(self.design0, Design(Test.folder() / 'design.0'))

    def test_Design0(self):
        design0 = Design0.create(Test.folder() / 'design.0', self.design0)
        self.assertEqual(design0, Design0.create(Test.folder() / 'design.0.t', self.table0))
        self.assertRaises(AssertionError, Design0.create, Test.folder() / 'design.0.d', self.design)
        self.assertRaises(AssertionError, Design0.create, Test.folder() / 'design.0.d', self.table)

    def test_Design1(self):
        design1 = Design1.create(Test.folder() / 'design.1', self.design)
        self.assertEqual(design1, Design1.create(Test.folder() / 'design.1.t', self.table))
        self.assertRaises(AssertionError, Design1.create, Test.folder() / 'design.1.d', self.design0)
        self.assertRaises(AssertionError, Design1.create, Test.folder() / 'design.1.d', self.table0)

    def test_DesignS(self):
        designS = DesignS.create(Test.folder() / 'design.S', self.design)
        design1 = Design1.create(Test.folder() / 'design.1', self.design)
        self.assertEqual(designS, DesignS.create(Test.folder() / 'design.S.t', self.table))
        self.assertEqual(designS, DesignS.create(Test.folder() / 'design.1.S', design1))
        design1 = Design1.create(Test.folder() / 'design.1', designS)
        self.assertEqual(designS, DesignS.create(Test.folder() / 'design.1.S', design1))
        self.assertRaises(AssertionError, DesignS.create, Test.folder() / 'design.S.d', self.design0)
        self.assertRaises(AssertionError, DesignS.create, Test.folder() / 'design.S.d', self.table0)


if __name__ == '__main__':
    ut.main()
