:html_theme.sidebar_secondary.remove:

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
       import polars as pl
       import numpy as np
       import torch as tc
       import unittest as ut
       from inspect import stack
       from pathlib import Path




Protocols
----------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/Protocol
   /pages/api/rc/base/definitions/NameP
   /pages/api/rc/base/definitions/IndexP
   /pages/api/rc/base/definitions/EqualsP
   /pages/api/rc/base/definitions/CreateP
   /pages/api/rc/base/definitions/ReadP
   /pages/api/rc/base/definitions/UpdateP
   /pages/api/rc/base/definitions/DeleteP
   /pages/api/rc/base/definitions/CopyP

.. autoapisummary::

   rc.base.definitions.Protocol
   rc.base.definitions.NameP
   rc.base.definitions.IndexP
   rc.base.definitions.EqualsP
   rc.base.definitions.CreateP
   rc.base.definitions.ReadP
   rc.base.definitions.UpdateP
   rc.base.definitions.DeleteP
   rc.base.definitions.CopyP



Attributes
----------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/zero
   /pages/api/rc/base/definitions/Int
   /pages/api/rc/base/definitions/Ints
   /pages/api/rc/base/definitions/Float
   /pages/api/rc/base/definitions/Floats
   /pages/api/rc/base/definitions/String
   /pages/api/rc/base/definitions/PathLike
   /pages/api/rc/base/definitions/IndexLike
   /pages/api/rc/base/definitions/DataFrame
   /pages/api/rc/base/definitions/Category
   /pages/api/rc/base/definitions/MetaData
   /pages/api/rc/base/definitions/TableData

.. autoapisummary::

   rc.base.definitions.zero
   rc.base.definitions.Int
   rc.base.definitions.Ints
   rc.base.definitions.Float
   rc.base.definitions.Floats
   rc.base.definitions.String
   rc.base.definitions.PathLike
   rc.base.definitions.IndexLike
   rc.base.definitions.DataFrame
   rc.base.definitions.Category
   rc.base.definitions.MetaData
   rc.base.definitions.TableData



Classes
-------

.. toctree::
   :hidden:

   /pages/api/rc/base/definitions/Np
   /pages/api/rc/base/definitions/Tc

.. autoapisummary::

   rc.base.definitions.Np
   rc.base.definitions.Tc


