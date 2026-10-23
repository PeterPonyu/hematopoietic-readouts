# Hematopoietic readouts: code and summary data

Scientific code and full-precision aggregate results for **Cell composition, endpoint geometry and policy learning in single-cell fate analysis**, by Zeyu Fu
([ORCID 0009-0001-8329-0108](https://orcid.org/0009-0001-8329-0108)).
Correspondence: fuzeyu99@tmmu.edu.cn; fuzeyu99@126.com.

This release supports the current four-cohort analysis: irradiation, TPO
culture, Dapp1 and sleep. The normal-marrow analysis is a separate technical
control. It contains the small source tables used by the seven main and nine
supplementary figure plates, the aggregate comparisons underlying the printed
supplementary tables, and the analysis implementations.

## Use the release

Run `python run.py verify` to check every released file against its recorded
SHA-256 digest. This standard-library operation does not fit or rerun a model.

Run `python run.py display --output-root /path/to/new-output` with an existing
compatible environment to regenerate all 16 figure plates from the included
aggregate inputs. The output folder must be new. The scientific estimates are
read from the saved tables; the display command does not recalculate them from
cells. Exact publication typography requires installed Arial. Font files are
not redistributed. Observed display dependencies are listed in
`requirements-display.txt`; they are not a transitive environment lock.

`source_data/` and `supplement_tables/` contain aggregate source tables.
`revision_data/` contains the external-method, endpoint, partition, training
and bounded diagnostic results and their analysis code. `analysis_replay/`
and `analysis_support/code/` contain retained analysis implementations.
Internal path names are retained so scientific input references stay traceable.

## Data access and limits

Source GEO accessions and the irradiation provenance gap are described in
[DATA_SOURCES.md](DATA_SOURCES.md). This is a deliberately small code-and-summary
release. Count matrices, cell identities, per-cell model outputs, subset masks,
resampling arrays and fitted checkpoints are not included. Summary tables
support inspection and aggregate display reproduction; they do not replace
those inputs for a cell-level reanalysis or refitting. The analysis implementations
that require those inputs are supplied for inspection and future use, with no
claim that this release alone executes the complete cell-level workflow.

The current manuscript and supplementary PDF are submitted separately to the
journal. They are not published by this software release. The earlier
[DOI 10.5281/zenodo.23435592](https://doi.org/10.5281/zenodo.23435592) identifies
the preceding three-cohort software edition, not the current release. No new
Zenodo DOI is claimed. The versioned current release is [https://github.com/PeterPonyu/hematopoietic-readouts/releases/tag/v2.0.0](https://github.com/PeterPonyu/hematopoietic-readouts/releases/tag/v2.0.0).

## Attribution

The scoped MIT license covers original software and documentation. Upstream
notices remain in `LICENSES/` and the included source folders. The Python FateID
port and the original R FateID are different implementations; the supplied
GPL-3 notice is preserved. Source datasets, aggregate scientific data, figures
and fonts retain their applicable terms. Biological replication, independent
fate validation and model retraining are not established by a packaging or
aggregate display check.
