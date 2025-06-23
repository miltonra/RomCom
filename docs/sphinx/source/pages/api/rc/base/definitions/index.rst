rc.base.definitions
===================

.. py:module:: rc.base.definitions

.. autoapi-nested-parse::

   Basic Protocols, Types and constants.

   All modules of RomCom ``import *`` from ``rc.definitions``, so all types and constants in this module are referenced
   without adornment throughout RomCom. The ``rc.definitions`` namespace includes::

   from typing import *
   from abc import ABC, abstractmethod
   from pathlib import Path
   from copy import copy, deepcopy
   import unittest as ut



Protocols
----------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/Protocol
   /pages/api/rc/base/definitions/Indexed
   /pages/api/rc/base/definitions/Len
   /pages/api/rc/base/definitions/Create
   /pages/api/rc/base/definitions/Read
   /pages/api/rc/base/definitions/Update
   /pages/api/rc/base/definitions/Delete
   /pages/api/rc/base/definitions/Copy
   /pages/api/rc/base/definitions/StrRepr

.. autoapisummary::

   rc.base.definitions.Protocol
   rc.base.definitions.Indexed
   rc.base.definitions.Len
   rc.base.definitions.Create
   rc.base.definitions.Read
   rc.base.definitions.Update
   rc.base.definitions.Delete
   rc.base.definitions.Copy
   rc.base.definitions.StrRepr


Attributes
----------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/zero

.. autoapisummary::

   rc.base.definitions.zero


Classes
-------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/Pd
   /pages/api/rc/base/definitions/Np
   /pages/api/rc/base/definitions/Tc

.. autoapisummary::

   rc.base.definitions.Pd
   rc.base.definitions.Np
   rc.base.definitions.Tc


