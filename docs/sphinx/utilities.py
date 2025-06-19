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

""" RomCom's documentation utilities. """

import os, argparse, time
import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

from rc.base import *

replacements = {'rc.base.definitions.': '', }

_here = Path(os.path.abspath(__file__)).parent  # docs/sphinx/
docs = _here.parent


def _empty(path: Path, preserve: List[str], confirm: bool = True):
    """ Remove all files and directories contained in ``path`` except those listed in ``preserve``.

    Args:
        path: The folder to empty
        preserve: The files to preserve in `path`.
        confirm: ``True`` to require confirmation of every deletion.
    """
    for filename in path.iterdir():
        if filename.name not in preserve:
            if filename in _here.parents:
                raise FileExistsError(f'Cannot delete {filename} as it contains the clean.py script.')
            if confirm and input(f'Press return to delete {filename}'):
                raise InterruptedError('User aborted cleanup.')
            else:
                Store.delete(filename)


def _clean():
    """ Clean all Sphinx generated files from ``docs/``. """
    _empty(docs / 'sphinx' / 'source' / 'pages' / 'api', preserve=[], confirm=False)
    _empty(docs, preserve=['sphinx', '.nojekyll'], confirm=False)


def _tidyfile(filename: Path):
    """ Tidy a ``.html`` file.

    Args:
        filename: The file to tidy
    """
    with open(filename, "r+") as f:
        content = f.read()
        for old, new in replacements.items():
            content = content.replace(old, new)
        f.seek(0)
        f.write(content)
        f.truncate()


def _tidy():
    """ Tidy the Sphinx generated ``html`` in ``docs/``. """
    for filename in (docs / 'pages').iterdir():
        _tidyfile(filename)


if __name__ == "__main__":
    """ Command line interface to RomCom's documentation utilities. 
    The ``parser`` help is misleading -- it is actually user help for the ``make`` command, 
    which this module is designed to serve.

    When using these utils outside ``make``:

    * ``clean`` removes all Sphinx output (usually before ``sphinx-build``). 
    * ``tidy`` amends all ``.html`` (only makes sense after ``sphinx-build``).
    """
    print()
    parser = argparse.ArgumentParser(prog='make',
        description='Make RomCom documentation using Sphinx <https://www.sphinx-doc.org/en/master/index.html#>.')
    subparsers = parser.add_subparsers(dest='cmd', help='Use one of these targets.')
    subparsers.add_parser('tidy', help='Completely re-build all RomCom documentation. Slowest. Recommended.')
    subparsers.add_parser('clean',
                          help='Flawed re-build of all RomCom documentation. Quicker, docs may be ugly. Not recommended.')
    subparsers.add_parser('dirty',
                          help='Flawed re-build of missing or corrupted RomCom documentation. Quickest, docs may be incomplete. Definitely not recommended.')
    start_time = time.time()
    match parser.parse_args().cmd:
        case 'clean':
            _clean()
            print(f'{parser.parse_args().cmd} took {time.time() - start_time}')
        case 'tidy':
            _tidy()
            print(f'{parser.parse_args().cmd} took {time.time() - start_time}')
        case _:
            parser.parse_args(['--help'])
