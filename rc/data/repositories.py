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

""" Repositories for data storage. """

from rc.data.normalizers import *

# class Repo(DataBase):
#     """ A Repository of data and models. Informally a dataset and all the things we'd like to do to it. """
#
#     class NamedTables(NamedTuple):
#
#         data: Table | Matrix | MetaData = pd.DataFrame(columns=('x', 'l', 'y'))
#
#         def __call__(self, name: str) -> Table | Matrix | MetaData:
#             """ Returns the Table named ``name``."""
#             return getattr(self, name)
#
#     options: NamedTables[MetaData] = NamedTables(data = Table.Options.defaults())
#
#     defaultMetaData: MetaData = {'K': 0}
#
#     @property
#     def fold(self):
#         """ The current fold. """
#         return self._fold
#
#     @fold.setter
#     def fold(self, value: int):
#         """ The current fold. A negative value refers test data in the fold numbered ``abs(value)``.
#             In case ``abs(value)`` is 0 or greater ``len(self)-1`` the current fold is ``self``. """
#         self._fold = value
#
#     def __len__(self) -> int:
#         """ 1 + K proper folds in ``self``. """
#         return self._meta['K'] + 1
#
#     def __getitem__(self, fold: int | slice) -> Path | tuple[Path, ...]:
#         """ Indexer returns the Path (s) to the Folds indexed or sliced by ``fold``. """
#         if isinstance(fold, int):
#             return self.path  if fold == 0 else self.path / f'{abs(fold)}'
#         else:
#             return tuple((self[i] for i in range(len(self))))[fold]
#
#     def __setitem__(self, fold: int | slice , tables: Table | Matrix | tuple[Table | Matrix, ...]):
#         """ Indexer creates the ``Fold`` (s) named or sliced by ``name``."""
#         self[fold] = tables
#
#     def __call__(self, **meta: Any) -> Self:
#         """ Optimize and update ``self``.
#
#         Args:
#             **meta: Optimization ``MetaData``.
#
#         Returns: ``self``
#         """
#         self(**meta)
#         return self
#

