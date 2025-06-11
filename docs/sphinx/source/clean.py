# Cleanup script for the Sphinx documentation builder.


import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from rc.base import *

here = Path(os.path.abspath(__file__)).parent  # docs/sphinx/source/
scope = Path('C:/Users/rober/Documents/Research/RomCom/bin/docs')


def empty(path: Path, preserve: List[str]):
    """ Remove all files and directories contained in ``path`` except those listed in ``preserve``.

    Args:
        path: The folder to empty
        preserve: The files to preserve in `path`.

    Returns: ``path`` which now contains only ``preserve``.
    """
    for filename in path.iterdir():
        if filename.name not in preserve:
            if filename in here.parents:
                raise UserError(f'Cannot delete {filename} as it contains the clean.py script.')
            if not input(f'Press return to delete {filename}'):
                Store.delete(filename)
            else:
                raise UserError('User aborted cleanup.')

if __name__ == "__main__":
    empty(scope / 'sphinx' / 'source' / 'pages' / 'api', preserve=['api.rst'])
    empty(scope / 'sphinx', preserve=['source'])
    empty(scope, preserve=['sphinx', '.nojekyll'])
