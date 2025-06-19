rc.task.contexts
================

.. py:module:: rc.task.contexts

.. autoapi-nested-parse::

   **Context managers**



Methods
---------

.. autoapisummary::

   rc.task.contexts.Timer
   rc.task.contexts.Environment


Module Contents
---------------

.. py:function:: Timer(name = '', is_inline = True)

   Context Manager for timing operations.

   :param name: The name of this context, ``print``ed as what is being timed. The (default) empty string will not be timed.
   :param is_inline: Whether to report timing inline (the default), or with linebreaks to top and tail a paragraph.


.. py:function:: Environment(name = '', device = '', **kwargs)

   Context Manager setting up the environment to task operations.

   :param name: The name of this context, ``print``ed as what is being task. The (default) empty string will not be timed.
   :param device: The device to task on. If this ends in the regex ``[C,G]PU*`` then the logical device ``/[C,G]PU*`` is used,
                  otherwise device allocation is automatic.
   :param \*\*kwargs: Is passed straight to the implementation GPFlow manager. Note, however, that ``float=float32`` is inoperative due to SciPy.
                      ``eager=bool`` is passed to `tf.config.run_functions_eagerly <https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly>`_.


