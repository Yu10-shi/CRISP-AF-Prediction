# UCTT-RP — AF & All-Cause Mortality Risk Prediction (Shiny App)

Interactive R Shiny application for individualized prediction of atrial fibrillation
(AF) and all-cause mortality under competing risks, using a TabTransformer + DeepHit
survival model. The interface shows per-patient event-probability and survival curves,
an **Aalen–Johansen–recalibrated upper predicted bound (UPB)** for each outcome, and a
test-case-level permutation feature-importance module.

> ⚠️ **The trained model and all patient/clinical data are NOT included in this
> repository for data-privacy reasons.** The code is fully provided. The trained
> model weights and the underlying data cannot be shared publicly; they are
> **available by request** — please contact **Yu Shi** (ysherry.shi@mail.utoronto.ca).
> Once you have the model file, place it in this `ShinyApp/` folder (see
> **"Required model file"** below) to run the app.

---

## Repository contents

```
.
├── SETUP.md                                # this file
├── app.R                                   # main Shiny app
├── predict_model.py                        # Python inference (called via reticulate)
├── model_meta.json                         # feature order, categ/cont indices, risk names
├── cp_outputs/
│   └── ucttrp_aj_upb_recalibration.npz      # aggregate AJ calibration constants (no patient data)
└── env/
    ├── requirements_app.txt                # Python (pip) dependencies
    ├── install_r_packages.R                # R dependencies
    └── environment.yml                     # conda spec (reference)
```

Not in the repo (excluded by `.gitignore`): the model `.pt`, all `.csv`/`.xlsx` data,
and figures.

---

## Required model file (not included)

The app loads a trained model that is **not distributed here**. After cloning, place
your model weights file in this `ShinyApp/` folder with this exact name:

```
end-lr0.0001-wd0.01-eta0.8-alpha0.2-bs256-drop0.1-epochs100-dur30-embed500.pt
```

This is the local path `app.R` falls back to (see the `model_candidates` vector near
the top of `app.R`). To use a different filename or location, edit that vector.

The model must be a 30-time-bin (`dur30`) TabTransformer + DeepHit model consistent
with `model_meta.json`; the bundled `cp_outputs/ucttrp_aj_upb_recalibration.npz`
(delta + 30-bin time grid) is calibrated for that model.

**Optional data:** permutation feature-importance needs reference rows to sample
replacement values from. Provide them by uploading a multi-row CSV (the other rows
are used as the reference), or by adding an `example_input.csv` (one or more records
with the model's feature columns) to this `ShinyApp/` folder as a fallback.

---

## Setup

### 1. Python environment (conda, Python 3.10)

```bash
conda create -n hu_ssc python=3.10 -y
conda activate hu_ssc
pip install -r env/requirements_app.txt
```

### 2. R packages (R >= 4.2, reticulate >= 1.38)

```bash
Rscript env/install_r_packages.R
```

> `reticulate` must be **>= 1.38** — older versions cannot talk to NumPy 2.x and fail
> at prediction time with *"Required version of NumPy not available"*.

### 3. Point the app at your conda env

Edit line 5 of `app.R`:

```r
use_condaenv("/path/to/your/conda/envs/hu_ssc", required = TRUE)
```

Find the path with `conda env list`.

---

## Run

```bash
Rscript -e 'shiny::runApp("app.R", launch.browser = TRUE)'
```

(or open `app.R` in RStudio → **Run App**). Run from inside this `ShinyApp/` folder so
the relative paths (`model_meta.json`, `cp_outputs/...`) resolve.

---

## Method note

The uncertainty quantification is an Aalen–Johansen–recalibrated upper predicted bound
(γ = 0.10, nominal 90% coverage), computed separately for AF and all-cause death. It is
a post-hoc recalibration of the predicted cumulative incidence functions, **not** a
finite-sample conformal interval.
