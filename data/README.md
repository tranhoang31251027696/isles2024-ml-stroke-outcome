# Data are not redistributed in this repository

Download the public ISLES 2024 Zenodo version 7 release from [10.5281/zenodo.17652035](https://doi.org/10.5281/zenodo.17652035), subject to its terms of use.

For the V3 numerical workflow, place the source clinical files below `data/source_v7/` while preserving their source filenames and directory structure. The ingestion script expects clinical baseline and outcome spreadsheets in `data/source_v7/clinical_raw/` and writes derived files to `outputs/` during a local run.

Do not commit downloaded patient-level files, derived patient-level predictions, or trained models to a public fork.
