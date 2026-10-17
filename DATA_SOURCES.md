# Data sources and terms

The aggregate tables derive from these public GEO studies, as identified in the article's frozen acquisition manifest:

| Article role | Accession | Source record |
| --- | --- | --- |
| TPO cord-blood culture | GSE280270 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280270 |
| Dapp1 WT/KO and spliced-expression sensitivity | GSE277292 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277292 |
| Sleep control/deprivation | GSE280145 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280145 |
| Normal-marrow technical control | GSE117498 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117498 |

`provenance/source_acquisition.json` retains source filenames, raw-file checksums and acquisition URLs from that manifest. These are source provenance records, not a claim that an online acquisition was repeated for this release. Cite the original source studies and their accessions when reusing derived material. The software license does not grant a new dataset license or determine source consent, ethics, institutional secondary-use status or ownership.

This source repository contains aggregate summaries only. Program/marker counts, cell identifiers, frozen coordinates, downstream outputs and encoder artifacts needed for retained-artifact replay are distributed separately as Supplementary Data 1–8 with Supplementary Code 1. Preserve the common extraction folder and supplied integrity manifest. Original full raw matrices are not duplicated here.
