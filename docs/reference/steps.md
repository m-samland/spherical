# Pipeline steps

A reduction runs these steps in order. Each step has a switch in `config.steps`;
switching one off skips it, and by default a step whose outputs already exist is
skipped too (see `config.steps.force` to recompute).

::::{tab-set}
:sync-group: instrument

:::{tab-item} IFS
:sync: ifs

```{eval-rst}
.. step-table:: ifs
```
:::

:::{tab-item} IRDIS
:sync: irdis

```{eval-rst}
.. step-table:: irdis
```
:::
::::
