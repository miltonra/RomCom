:html_theme.sidebar_secondary.remove:

rc.data.samples.Function.collection
===================================

.. py:method:: rc.data.samples.Function.collection(sub_folder: rc.base.Union[rc.base.Path, str]) -> dict[str, rc.base.Any]

   Construct a dict for task.results.Collect, with appropriate ``extra_columns``.

   :param folder: The folder under ``self.repo.folder`` housing the csvs to collect.

   Returns: The dict for ``self.repo``.

