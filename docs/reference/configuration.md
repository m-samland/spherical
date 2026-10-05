# Configuration

A reduction is configured with one Python object, `IFSReductionConfig` or
`IRDISReductionConfig`, passed to `execute_targets`. Each groups several
sub-configs; you change a value with `config.<sub-config>.<field> = value` or
`config.<sub-config> = config.<sub-config>.merge(field=value)`. TRAP has its own
configuration, described in [TRAP settings](trap-settings.md).

::::{tab-set}
:sync-group: instrument

:::{tab-item} IFS
:sync: ifs

```{eval-rst}
.. config-table:: spherical.pipeline.pipeline_config.IFSReductionConfig
   :common: directories.base_path, directories.raw_directory, directories.reduction_directory, preprocessing.eso_username, preprocessing.frame_types_to_extract, steps.force, use_gaia_stellar_parameters
```
:::

:::{tab-item} IRDIS
:sync: irdis

```{eval-rst}
.. config-table:: spherical.pipeline.pipeline_config.IRDISReductionConfig
   :common: directories.base_path, directories.raw_directory, directories.reduction_directory, preprocessing.eso_username, preprocessing.frame_types_to_extract, irdis_preprocessing.crop, irdis_preprocessing.crop_size, steps.force
```
:::
::::

To set the CPU count, call `config.set_ncpu(n)` once instead of setting the
individual CPU fields. For TRAP, also call `config.apply_trap_resources(trap_config)`.
