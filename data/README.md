# Data

The raw and processed cohorts are not redistributed in this package. Both
databases are public and can be obtained from PhysioNet.

| Cohort | Source | Licence | Used for |
|---|---|---|---|
| Sleep-EDF expanded (sleep-cassette) | https://physionet.org/content/sleep-edfx/1.0.0/ | ODC-BY 1.0 | Sleep-EDF-20: 20 subjects, 39 recordings, Fpz-Cz, R&K |
| HMC Sleep Staging Database | https://physionet.org/content/hmc-sleep-staging/1.0.0/ | CC-BY 4.0 | 151 recordings, 151 patients, C4-M1, AASM |

## Preprocessing

Both cohorts are band-pass filtered between 0.5 and 30 Hz, resampled to
100 Hz, standardised per recording, and split into 30 s epochs with the five
stages wake, N1, N2, N3 and REM. The preprocessing scripts write one `.npz`
file per recording.

```bash
# Sleep-EDF-20
python scripts/prepare_sleep_edf.py \
    --data-dir data/sleep-edf/sleep-cassette \
    --out-dir data/processed/sleep-edf-20 \
    --subset sleep-edf-20 \
    --channel Fpz-Cz \
    --resample 100

# HMC
python scripts/prepare_hmc.py \
    --data-dir data/hmc/recordings \
    --out-dir data/processed/hmc \
    --channel C4-M1 \
    --resample 100 \
    --workers 4
```

The processed arrays are deterministic functions of the public recordings and
the scripts above; they are omitted from the archive only because of their
size (about 42 GB).
