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
from rc.base.unittests import TestCase as base
from rc.data.unittests import TestCase as data


# def suite():
#     suite = ut.TestSuite()
#     suite.addTest(base())
#     suite.addTest(data())
#     return suite

if __name__ == '__main__':
    ut.main()
    # runner = ut.TextTestRunner()
    # runner.run(suite())