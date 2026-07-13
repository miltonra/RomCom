:html_theme.sidebar_secondary.remove:

rc.data.designs.Design.yPivot
=============================

.. py:method:: rc.data.designs.Design.yPivot(path: rc.base.PathLike | None = None) -> rc.base.Table

   Create a Table at ``path`` consisting of ``self`` with 'y' values pivoted on the 'l' axis.

   :param path: The Path to store this Table, overwritten if existing.

   Returns: A fat Table with no 'l' axis but several output axes in place of the 'y' axes.

