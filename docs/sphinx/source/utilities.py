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

from rc.base import *

import os, argparse, time


replacements = {'rc.base.definitions.': '', }


_here = Path(os.path.abspath(__file__)).parent  # docs/sphinx/source/
docs = _here.parent.parent


def _empty(path: Path, preserve: List[str], confirm: bool = True):
    """ Remove all files and directories contained in ``path`` except those listed in ``preserve``.

    Args:
        path: The folder to empty
        preserve: The files to preserve in `path`.
        confirm: ``True`` to require confirmation of every deletion.
    """
    for filename in path.iterdir():
        if filename.name not in preserve:
            if filename in here.parents:
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
    # Get the command line arguments.
    parser = argparse.ArgumentParser(description='A program to clean and tidy RomCoom documentation.')
    subparsers = parser.add_subparsers(dest ='cmd')
    subparsers.add_parser('clean', help='Clean all Sphinx documentation files from ``docs/``.')
    subparsers.add_parser('tidy', help='Tidy Sphinx generated html in ``docs/``.')
    start_time = time.time()
    match parser.parse_args().cmd:
        case 'clean':
            _clean()
        case 'tidy':
            _tidy()
        case _:
            parser.parse_args(['--help'])
            exit(0)
    print(f'{parser.parse_args().cmd} took {time.time() - start_time}')
