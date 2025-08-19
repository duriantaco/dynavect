# DynaVect User Study (MTurk) – Dataset & Reproduction

This folder contains the stimuli, HTML, and anonymized results for the DynaVect two-alternative forced-choice (2AFC) study comparing **ours** vs **StyleCLIP baselines**.

## Files
- `survey.html` – the exact HTML shown to the anonymous workers (58 questions). Contains the A/B image order for every question.
- `dynavect-survey-images-final.csv` – prompt + image URLs used to generate the pages (columns: `prompt_text, image_A_url, image_B_url`).
- `Survey.csv` – MTurk batch results export (**58 submitted assignments**, 58 unique workers). Each response is recorded in columns from `Answer.q1`-> `Answer.q58` with values of either `A` or `B`. Other columns are just MTurk's metadata. You can ignore these.


## Directory layout
```
/Supplementary/
  survey.html
  dynavect-survey-images-final.csv
  Survey.csv
  README.md
```

## Column dictionary (Survey.csv)
- `WorkerId` – anonymized worker id
- `AssignmentId`, `AssignmentStatus` – MTurk identifiers; use `AssignmentStatus == "Submitted"` to filter valid responses.
- `Answer.qN` – choice for question N (`"A"` or `"B"`). The column will be blank if the worker skipped the item.
- Remaining columns are MTurk metadata

## How A/B maps to models
Use survey.html to map `A`/`B` to a model for each question.

## Quick stats from these files
(Computed from `Survey.csv` with `AssignmentStatus=="Submitted"`; see code below.)

- **Per-response preference**
  - Ours vs **StyleCLIP-Optimizer**: **58.9%** of individual votes for *Ours* (n=1571 responses).
  - Ours vs **StyleCLIP-Global**: **60.3%** of individual votes for *Ours* (n=1576 responses).

- **Per-question majority wins**
  - Against **StyleCLIP-Optimizer**: *Ours* wins **24/29** questions (**82.8%**).
  - Against **StyleCLIP-Global**: *Ours* wins **26/29** questions (**89.7%**).

> Note: Per-response (%) and per-question-majority (%) answer slightly different research questions; report the one you pre-registered. The HTML determines which baseline is on A vs B for each question.

## Attention checks
No explicit attention-check items detected in this HTML.

## License

- **Metadata (CSV) and annotations:** © 2025 Oh Aaron. Licensed under **CC BY 4.0**.
- **Generated images (all PNGs referenced by `survey.html`):** © 2025 Oh Aaron. Licensed under **CC BY-NC 4.0**.
- **Survey scaffolding / HTML snippets:** Licensed under **MIT**.

> Note: The “Ethical Use Guidelines” below are **non-binding requests**, not additional legal terms on top of these licenses.

## Ethical Use Guidelines (non-binding)
To reduce misuse, we ask downstream users to:
- **Do not** use these images or derivatives for biometric identification, face recognition, or surveillance.
- **Do not** attempt to re-identify real individuals (images are synthetic).
- Avoid applications that could cause harm or discrimination; limit use to research, teaching, and reproducibility.

## NOTICE
If you use this dataset, please attribute:
> Oh, Aaron. *DynaVect: Context-Aware Modulation of Global Edit Directions for Controllable GAN Editing*. 2025.

## Citation
If you use this data, please cite:  
> Oh, Aaron. *DynaVect: Context-Aware Modulation of Global Edit Directions for Controllable GAN Editing*. 2025.

## Contact
For questions: aaronoh2015@gmail.com
