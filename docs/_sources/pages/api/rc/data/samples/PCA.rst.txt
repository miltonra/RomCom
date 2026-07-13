:html_theme.sidebar_secondary.remove:

rc.data.samples.PCA
===================

.. py:function:: rc.data.samples.PCA(root: str | rc.base.Path, csv: str | rc.base.Path) -> rc.base.Path

   Perform Principal Component Analysis on a Repo.

   :param root: The root folder.
   :param csv: The csv to read.
   :param normalization: An optional csv file to use for normalization.

   Returns: The folder written to, namely root``/PCA``

