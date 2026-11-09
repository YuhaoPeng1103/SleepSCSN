# Neurocomputing submission record

Target journal: **Neurocomputing** (Elsevier, ISSN 0925-2312).

## Authors

| Role | Name | Affiliation |
|---|---|---|
| First author | Zhiwen Song | School of Mathematics and Physics, China University of Geosciences (Wuhan), Wuhan 430074, China |
| Corresponding author | Yuhao Peng | Same affiliation; `pngyuo@cug.edu.cn`; ORCID `0009-0004-3302-9050` |

## Submission choices

- Article type: **Research paper**
- Open access: **No**
- Funding: **None**
- Suggested reviewers: **None**
- Data sharing: **Yes**, PhysioNet Sleep-EDF and HMC records
- Code: `https://github.com/YuhaoPeng1103/SleepSCSN`

## Title

Stage-conditional shift normalization for subject-robust EEG sleep staging

**Running title:** Stage-conditional normalization for sleep staging

## Abstract

The abstract is generated from `tex/sections/00_abstract.tex` and must remain
at or below 250 words.

## Keywords

Sleep staging; Test-time adaptation; Normalization; Subject variability;
Electroencephalography; Conformal prediction

## Highlights

The five final highlights are stored in `highlights.txt`; each is at most 85
characters including spaces.

## Declarations

The authors declare no competing interests and received no specific funding.
The study used public, de-identified PhysioNet recordings. A generative-AI
disclosure is included before the references. The code and released result
summaries are public.

## Build

```bash
cd paper2/tex
latexmk -pdf main.tex
latexmk -pdf manuscript_review.tex
latexmk -pdf title_page.tex
latexmk -pdf cover_letter.tex
```

## Final checks

- [ ] Abstract at most 250 words
- [ ] Highlights: 3--5 bullets, each at most 85 characters
- [ ] No undefined citations, references, or overfull boxes
- [ ] All figures and tables inspected at final resolution
- [ ] At least 30 unique references
- [ ] First author has no email in the manuscript
- [ ] Corresponding author and ORCID verified
- [ ] Public code URL verified
