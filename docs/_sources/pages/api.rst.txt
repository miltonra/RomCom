
API
==================

When ordered alphabetically, the module hierarchy is understood top-down, and used bottom-up.
For example, the ``base`` module is fundamental to everything, but the ``user`` module is the best interface
to achieve most tasks.
Naturally, code dependency is bottom-up alphabetically.

Alphabetical ordering applies to submodules too.
For example, the ``data`` module depends only on ``base`` and consists of three submodules: ``data.samples`` depends on ``data.models`` which depends on ``data.functions``.

.. include:: api/api.rst


Conventions
------------------

.. include:: ../conventions.rst

