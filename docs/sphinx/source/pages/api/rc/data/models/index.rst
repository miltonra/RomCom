rc.data.models
==============

.. py:module:: rc.data.models

.. autoapi-nested-parse::

   Models for data storage.



Classes
-------

.. autoapisummary::

   rc.data.models.DesignMatrix
   rc.data.models.NormalDesignMatrix
   rc.data.models.Normalization
   rc.data.models.Repo


Module Contents
---------------

.. py:class:: DesignMatrix(path, data = None, **options)

   Bases: :py:obj:`rc.base.Table`

   .. autoapi-inheritance-diagram:: rc.data.models.DesignMatrix
      :parts: 1


   The familiar user format of ``DesignMatrix`` which is fat (has many columns).


   .. py:attribute:: Label

      Class attribute aliasing acceptable Types for column (or index) labels.


   .. py:attribute:: skeleton
      :type:  rc.base.Pd.DataFrame

      DataFrame of the minimal, skeleton ``DesignMatrix``.


   .. py:attribute:: defaultOptions
      :type:  rc.base.MetaData

      Default file handling ``DesignMatrix.Options()``.


   .. py:method:: create(path, src, columns_in_l = '')
      :classmethod:


      Reformat the ``NormalDesignMatrix`` in ``src`` as a ``Self(DesignMatrix)``.

      :param path: The ``Path`` to store the ``DesignMatrix`` created, overwritten if existing.
      :param src: The ``NormalDesignMatrix`` to reformat.

      Returns: The ``NormalDesignMatrix`` created at ``dst``.



   .. py:method:: copy(src, dst = '')
      :classmethod:


      Reformat this ``DesignMatrix`` to a ``NormalDesignMatrix``.

      :param src: The ``DesignMatrix`` to reformat.
      :param dst: Optional ``Path`` to the ``NormalDesignMatrix``.
                  Defaults to ``''``, which overwrites ``src``.

      Returns: The ``NormalDesignMatrix`` created at ``dst``.



.. py:class:: NormalDesignMatrix(path, data = None, **options)

   Bases: :py:obj:`DesignMatrix`

   .. autoapi-inheritance-diagram:: rc.data.models.NormalDesignMatrix
      :parts: 1


   The internal format of ``DesignMatrix``, which is thin (has few columns).


   .. py:method:: create(path, src)

      Reformat the ``NormalDesignMatrix`` in ``src`` as a ``Self(DesignMatrix)``.

      :param path: The ``Path`` to store the ``DesignMatrix`` created, overwritten if existing.
      :param src: The ``NormalDesignMatrix`` to reformat.

      Returns: The ``NormalDesignMatrix`` created at ``dst``.



   .. py:method:: copy(src, dst = '')

      Reformat this ``DesignMatrix`` to a ``NormalDesignMatrix``.

      :param src: The ``DesignMatrix`` to reformat.
      :param dst: Optional ``Path`` to the ``NormalDesignMatrix``.
                  Defaults to ``''``, which overwrites ``src``.

      Returns: The ``NormalDesignMatrix`` created at ``dst``.



.. py:class:: Normalization(path, **tables)

   Bases: :py:obj:`rc.base.DataBase`

   .. autoapi-inheritance-diagram:: rc.data.models.Normalization
      :parts: 1


   Normalization of a Repo.


   .. py:class:: NamedTables

      Bases: :py:obj:`rc.base.NamedTuple`

      .. autoapi-inheritance-diagram:: rc.data.models.Normalization.NamedTables
         :parts: 1


      Must be overridden.


      .. py:method:: __call__(name)

         Returns the Table named ``name``.




   .. py:attribute:: options
      :type:  Normalization.NamedTables[rc.base.MetaData]

      Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
      Override as necessary for bespoke ``Table.options``.
      Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
      the remainder populate ``self[i].options.read``.


   .. py:method:: __call__(**meta)

      Optimize and update ``self``.

      :param \*\*meta: Optimization ``MetaData``.

      Returns: ``self``



   .. py:method:: create(path, data, **meta)
      :classmethod:


      Create a ``Normalization`` in ``path``.

      :param path: The folder to store the ``Normalization`` in. Need not exist,
                   any existing ``Tables`` will be overwritten if it does.
      :param \*\*meta: Optimization ``MetaData``.

      Returns: The ``Normalization`` created.



.. py:class:: Repo(path, **tables)

   Bases: :py:obj:`rc.base.DataBase`

   .. autoapi-inheritance-diagram:: rc.data.models.Repo
      :parts: 1


   A Repository of data and models. Informally a dataset and all the things we'd like to do to it.


   .. py:class:: NamedTables

      Bases: :py:obj:`rc.base.NamedTuple`

      .. autoapi-inheritance-diagram:: rc.data.models.Repo.NamedTables
         :parts: 1


      Must be overridden.


      .. py:method:: __call__(name)

         Returns the Table named ``name``.




   .. py:attribute:: options
      :type:  Repo.NamedTables[rc.base.MetaData]

      Class attribute of the form ``NamedTables(**{names[i]: options[i], ...})``.
      Override as necessary for bespoke ``Table.options``.
      Elements of ``options[i]`` found in ``Table.writeOptions`` populate ``self[i].options.write``,
      the remainder populate ``self[i].options.read``.


   .. py:property:: fold

      The current fold.


   .. py:method:: __len__()

      1 + K proper folds in ``self``.



   .. py:method:: __getitem__(fold)

      Indexer returns the ``Path`` (s) to the Folds indexed or sliced by ``fold``.



   .. py:method:: __setitem__(fold, tables)

      Indexer creates the ``Fold`` (s) named or sliced by ``name``.



   .. py:method:: __call__(**meta)

      Optimize and update ``self``.

      :param \*\*meta: Optimization ``MetaData``.

      Returns: ``self``



