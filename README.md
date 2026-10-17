# Hematopoietic readouts: scientific code (v1.0.1)

Code and aggregate source tables for **Cell composition and endpoint proximity constrain model-derived fate interpretations in three hematopoietic single-cell datasets**, by Zeyu Fu ([ORCID 0009-0001-8329-0108](https://orcid.org/0009-0001-8329-0108)). Correspondence: fuzeyu99@tmmu.edu.cn; fuzeyu99@126.com.

Software edition **1.0.1** updates figure presentation and adds editable SVG export alongside PDF and 600 dpi PNG for all eight figures. Text is black and regular, panel letters are bold at 13 pt, subpanel titles are centered and regular, and the minimum native text size is 10 pt. Scientific inputs, aggregate numerical results and the original 378-value, supplementary 376-value and five-value Figure 3 context ledgers are unchanged. See [CHANGELOG.md](CHANGELOG.md). This edition is prepared for release; no archive DOI has been assigned to v1.0.1.

The study compares captured composition, fixed transcriptional programs and conditional model readouts in TPO culture, Dapp1 and sleep cohorts. Normal marrow is a separate technical control. Program-based cell labels are provisional; computational seeds are not biological replicates.

Three reproducibility tiers have different inputs and claims:

| Tier | Inputs and command | What it establishes |
| --- | --- | --- |
| Aggregate displays | This repository; `python run.py tables --output-root ../readout-tables` or `python run.py display --output-root ../readout-displays` | Formats saved summary tables and redraws six main and two supplementary figures. Checks the original 378 main-figure values, five complementary Figure 3a values and 376 supplementary display values. It does not recompute cell-level science. |
| Retained-artifact scientific replay | Separately supplied Supplementary Code 1 and Supplementary Data 1–8; commands below | Recalculates 28 scientific summary tables from retained cell-level outputs; checks expression/program reconstruction. Optional `--checkpoints` exports deterministic posterior means and bottlenecks for three retained encoders. The additional sensitivity command recomputes single-marker and leave-one-marker-out summaries under frozen groups. |
| Raw preparation and model retraining | Original public matrices, source preparation choices and complete training environment | Documented limits and source accessions are provided. This release does not establish raw preparation, model retraining, downstream model-checkpoint inference, historical in-memory code identity, independent annotation or biological validation. |

Start with `python run.py verify`. This standard-library check validates the repository file inventory and SHA-256 checksums; it does not execute an analysis. All output paths must be new and outside this repository.

For scientific replay, extract **all nine independent ZIP archives** into one parent directory, preserving their common `scientific-reports-hematopoietic-readouts` folder. Then use that folder as `BUNDLE`:

```bash
python run.py verify-supplements --bundle /path/to/BUNDLE
python run.py replay --bundle /path/to/BUNDLE --output-root ../readout-replay \
  --cohorts ucb dapp1 sleep marrow dapp1_spliced --checkpoints --figures
python run.py sensitivity --bundle /path/to/BUNDLE --output-root ../readout-sensitivity
```

The replay wrapper verifies the supplied scientific edition, binds this repository's replay code to the separately supplied inputs in a temporary workspace, and records the actually executed checks. It downloads nothing. Missing supplements are reported as missing inputs. The repository contains no per-cell identifiers, count matrices, retained model outputs or checkpoints. These separate supplements must be distributed through the article's approved data-access route; obtaining the GitHub source alone does not provide tier 2 inputs.

Use an existing compatible Python environment. `requirements-display.txt` and `analysis_replay/requirements-replay.txt` record observed versions, rather than a complete transitive installation lock. Integrity and basic table formatting use the standard library; figures need NumPy, pandas, Matplotlib and the article's Arial font. Scientific replay also needs SciPy and scikit-learn; encoder export needs PyTorch. Marker sensitivity uses NumPy and pandas. Arial is not redistributed. No clean-install or full historical training-environment claim is made.

`source_data/` holds aggregate scientific inputs; `supplement_tables/` holds full-precision CSVs and formatted TeX; `sensitivity_data/` holds aggregate sensitivity outputs. `reference_data/` contains display-value references. See [PROVENANCE.md](PROVENANCE.md), [NOTICE.md](NOTICE.md) and [DATA_SOURCES.md](DATA_SOURCES.md) for source bindings and terms. The scoped [MIT license](LICENSE) covers original software and its documentation; aggregate data, source datasets and upstream code retain their separate terms and attribution.

Maintainers can refresh the allowlisted release from the final article folder with:

```bash
python tools/refresh_from_article.py --source /path/to/scientific-reports-hematopoietic-readouts
python run.py verify
```

The refresh records source and released checksums, updates the associated manuscript title, and removes superseded generated copies. It does not copy the manuscript, author forms, review records or general project history. `CITATION.cff` and `.zenodo.json` describe software edition 1.0.1 without assigning it a DOI. The **previous version 1.0.0** is frozen at [Zenodo DOI 10.5281/zenodo.23435592](https://doi.org/10.5281/zenodo.23435592); that identifier belongs to v1.0.0. Software archiving does not assert journal publication. The dedicated source repository is [PeterPonyu/hematopoietic-readouts](https://github.com/PeterPonyu/hematopoietic-readouts).
