# Reference check

`refs.bib` is shared with paper 1. The entries added for paper 2 are listed
below with the identifier each was checked against (Crossref by DOI, DataCite
for the PhysioNet database, arXiv for preprints, publisher page for
proceedings). Title, author list, venue and year of every entry were compared
with the record, not with memory.

| Key | Venue | Identifier |
|---|---|---|
| `ioffe2015bn` | ICML 2015 | arXiv:1502.03167 |
| `wu2018group` | International Journal of Computer Vision | 10.1007/s11263-019-01198-w |
| `ulyanov2016instance` | arXiv | arXiv:1607.08022 |
| `ba2016layer` | arXiv | arXiv:1607.06450 |
| `li2018adabn` | Pattern Recognition 80:109--117 | 10.1016/j.patcog.2018.03.005 |
| `nado2020` | arXiv | arXiv:2006.10963 |
| `wang2021tent` | ICLR 2021 | arXiv:2006.10726 |
| `sun2020ttt` | NeurIPS 2020 | proceedings |
| `frankle2021bn` | ICLR 2021 | arXiv:2007.05395 |
| `nasiri2021subgrouping` | Sleep 44(6) | 10.1093/sleep/zsab027 |
| `ghasemigarjan2025` | Journal of Neural Engineering | 10.1088/1741-2552/adeec7 |
| `liao2020trifeaturenet` | IEEE ISCAS 2020 | 10.1109/iscas45731.2020.9180501 |
| `katthi2021cca` | IEEE ICASSP 2021 | 10.1109/icassp39728.2021.9414274 |
| `hmc2022` | PhysioNet, HMC Sleep Staging Database v1.1 | 10.13026/t79q-fr32 |
| `cafa2023` | ICCV 2023 | 10.1109/ICCV51070.2023.01747 |
| `fairadabn2023` | MICCAI 2023, LNCS | publisher record |
| `unmixtns2024` | ICLR 2024 | arXiv:2401.08328 |
| `psdnorm2026` | ICLR 2026 | arXiv:2503.04582 |
| `s3dnet2026` | DASFAA 2026, LNCS 16537 | publisher record |
| `cross2026` | ACM SAC 2026, pp. 1917--1924 | ACM DL record |

## Added during the 2026-11-09 Neurocomputing reviewer pass

| Key | Venue | DOI and verification |
|---|---|---|
| `guo2025tta` | Biomedical Signal Processing and Control 100:106980, 2025 | 10.1016/j.bspc.2024.106980; Crossref metadata checked 2026-11-09 |
| `metasth2026` | Neurocomputing 701:134535, 2026 | 10.1016/j.neucom.2026.134535; Crossref metadata checked 2026-11-09 |
| `sedaeeg2025` | Neurocomputing 622:129315, 2025 | 10.1016/j.neucom.2024.129315; Crossref metadata checked 2026-11-09 |
| `transformeeg2026` | Neurocomputing 664:132075, 2026 | 10.1016/j.neucom.2025.132075; Crossref metadata checked 2026-11-09 |

The final manuscript cites 41 unique references.

The HMC database DOI is the v1.1 record; an earlier v1.0 DOI is superseded.

## Unused entries

`refs.bib` also carries entries used only by paper 1. BibTeX prints only the
cited ones, so they do not appear in this manuscript's reference list.

## Check

```bash
python scripts/check_citations.py paper2   # cited keys missing from refs.bib
cd paper2/tex && latexmk -pdf main.tex     # undefined citations in the build
```
