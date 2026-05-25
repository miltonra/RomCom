rc.base.definitions
===================

.. py:module:: rc.base.definitions

.. autoapi-nested-parse::

   Basic Protocols, Types and constants.

   All modules of RomCom ``import *`` from ``rc.definitions``, so all types and constants in this module are referenced
   without adornment throughout RomCom. The ``rc.definitions`` namespace includes::

       from typing import *
       from collections.abc import *
       from abc import ABC, abstractmethod
       from copy import copy, deepcopy
       from inspect import stack
       from pathlib import Path
       import polars as pl
       import numpy as np
       import torch as tc
       import unittest as ut



Protocols
----------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/Protocol
   /pages/api/rc/base/definitions/IndexP
   /pages/api/rc/base/definitions/EqualsP
   /pages/api/rc/base/definitions/CreateP
   /pages/api/rc/base/definitions/ReadP
   /pages/api/rc/base/definitions/UpdateP
   /pages/api/rc/base/definitions/DeleteP
   /pages/api/rc/base/definitions/CopyP
   /pages/api/rc/base/definitions/NameP

.. autoapisummary::

   rc.base.definitions.Protocol
   rc.base.definitions.IndexP
   rc.base.definitions.EqualsP
   rc.base.definitions.CreateP
   rc.base.definitions.ReadP
   rc.base.definitions.UpdateP
   rc.base.definitions.DeleteP
   rc.base.definitions.CopyP
   rc.base.definitions.NameP


Attributes
----------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/zero
   /pages/api/rc/base/definitions/PathLike

.. autoapisummary::

   rc.base.definitions.zero
   rc.base.definitions.PathLike


Classes
-------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/Pl
   /pages/api/rc/base/definitions/Np
   /pages/api/rc/base/definitions/Tc

.. autoapisummary::

   rc.base.definitions.Pl
   rc.base.definitions.Np
   rc.base.definitions.Tc


