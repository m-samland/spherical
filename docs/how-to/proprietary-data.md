# Proprietary data and ESO credentials

After this guide you can download your own proprietary SPHERE data with your ESO account.

## Log in for the download

Without an account, `download_data` fetches public data anonymously. For data still in
its proprietary period, give your ESO user name in the reduction configuration.

```python
from spherical.pipeline.pipeline_config import IFSReductionConfig

config = IFSReductionConfig()
config.preprocessing = config.preprocessing.merge(
    eso_username="your_eso_username",
    store_password=True,
    delete_password_after_reduction=False,
)
```

The download step logs in through astroquery (`Eso().login`). The first time, it asks
for your password in the terminal.

`store_password`
: Keep the password in your system keyring, so you are asked only once. The default
  is `True`.

`delete_password_after_reduction`
: Remove the password from the keyring after `execute_targets` has reduced all
  observations. The default is `True`.

The reduction templates set both to `False`, which means a prompt for every observation
that needs a download, and nothing stored. Change them in your copy of the template to suit how you work.

## Run without a terminal

A batch job or a run under `nohup` cannot answer a password prompt. Store the password
in the keyring once, from an interactive shell on the same machine, under the service
name astroquery uses (astroquery 0.4.10, `astroquery/eso/core.py`).

```bash
python -m keyring set astroquery:www.eso.org your_eso_username
```

Then set `store_password=True` and `delete_password_after_reduction=False`, so the
password is still there for the next run. This needs a keyring backend. macOS has one
built in. On a Linux server without a desktop session there is often none, and then
the password cannot be stored. Check with `python -m keyring diagnose`.

## Use files you already have

`download_data` skips every raw file whose `SPHER.*` name it finds anywhere below
`config.directories.raw_directory`, and moves it into its place in the raw-data layout
([Output products](../concepts/products.md)). Files you downloaded yourself, for
example through the ESO archive web interface, can therefore be put into that
directory, and only the missing ones are fetched.

## Select only public data

To make sure a selection needs no login at all, filter with `public=True`. It keeps
sequences taken more than a year ago, past the usual proprietary period.

```python
from astropy.table import Table

from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase

database_dir = resolve_database_dir(default="~/data/sphere/database")
db = SphereDatabase(
    Table.read(database_dir / "table_of_observations_ifs.fits"),
    Table.read(database_dir / "table_of_files_ifs.csv"),
    instrument="ifs",
)
selected = db.filter(target_list=["51 Eri"], usable_only=True, public=True)
print(len(selected))
```

```text
10
```

All ten usable 51 Eri sequences in the v3.0.0 tables are public.
