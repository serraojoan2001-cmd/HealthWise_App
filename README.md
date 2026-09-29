# HealthWise Smart Premium Engine: Streamlit dashboard

A seven-page dashboard for the HealthWise Insurance machine-learning lab. It explains the two-stage pricing
engine (classify the customer into a risk tier, then price them inside that tier) and lets you play with it live.

## Pages

| Page | What it is | Lab source |
|---|---|---|
| **Boardroom Brief** | Executive summary, headline numbers, board memo | Part D |
| **Data Detective** | Cleaning, EDA, feature audit, hypothesis tests, VIF, Lasso cross-check | Part A |
| **Risk Sorter** | Stage 1 classifier: accuracy, confusion matrix, overfit check | Part B |
| **Price Engine** | Stage 2 regression: flat baseline, per-tier models, tuning, verdict | Part C |
| **Price Lab** | Live simulator: change any factor, watch tier and premium update. Pin a baseline, see why the price is what it is, sweep one factor, map two factors | Task E2 |
| **Batch Scorer** | Score a CSV of customers and apply what-if changes | Task E3 |
| **Guardrails** | What happens when the tier is wrong, limits, model card | Bonus 1 and 2, task E1 |

Every page ends with a **Key insights** box, and has "How I thought about it" notes and the lab's questions with answers.

## What is in this folder

```
healthwise-dashboard/
|-- app.py                     ENTRY POINT. Streamlit runs this file. Builds the navigation.
|-- requirements.txt           Packages Streamlit Cloud installs to run the app (versions pinned).
|-- requirements-train.txt     Extra packages needed only to run train_model.py.
|-- train_model.py             BUILD STEP. Rebuilds the two files in artifacts/ from healthwise.csv.
|-- README.md                  This file.
|-- .gitignore                 Keeps healthwise.csv (private course data) out of GitHub.
|-- .streamlit/
|   `-- config.toml            Colours and light settings.
|-- artifacts/
|   |-- healthwise_model.joblib   The trained two-stage engine (lab task E1).
|   `-- report.json               Every number and chart input shown in the app (no customer rows).
|-- views/                     One file per page (each has a render() function).
|   |-- boardroom_brief.py, data_detective.py, risk_sorter.py, price_engine.py,
|   |-- price_lab.py, batch_scorer.py, guardrails.py
|-- utils/
|   |-- engine.py              Loads the model; predict(), batch prediction, price breakdown, sweeps.
|   |-- charts.py              Every Plotly chart.
|   `-- theme.py               Fonts, colours, Key-insights box and other UI pieces.
|-- content/
|   `-- qa.py                  The lab questions and your answers (edit text here).
|-- data_samples/
|   `-- sample_customers.csv   Eight made-up customers for the Batch Scorer demo.
`-- tests/
    `-- smoke_test.py          Optional: opens every page headlessly and checks for errors.
```

The app **never reads `healthwise.csv`**. It only reads `artifacts/`, so the course dataset does not need to be on GitHub.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501.

## Deploy it (GitHub + Streamlit Community Cloud)

1. **Create a GitHub repository** and upload everything in this folder, keeping the folder structure.
   Use *Add file, Upload files* on github.com, or `git add . && git commit -m "dashboard" && git push`.
   Do **not** upload `healthwise.csv` (the `.gitignore` already excludes it).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app** (or **Create app**), choose your repository and branch (`main`), and set **Main file path** to `app.py`.
4. Open **Advanced settings** and choose **Python 3.12** (or 3.13) if it is offered.
5. Click **Deploy**. After a few minutes you get a public link like `https://your-name.streamlit.app`.

Later changes: commit and push to GitHub and the app redeploys automatically.

## Rebuild the models (only if you change the data or code)

```bash
pip install -r requirements-train.txt
python train_model.py --csv healthwise.csv
```

This overwrites `artifacts/healthwise_model.joblib` and `artifacts/report.json`. It prints the scikit-learn version
used. **`requirements.txt` must pin that same scikit-learn version.**

In Colab: upload `train_model.py` and `healthwise.csv`, run `!pip install statsmodels` if needed, run
`!python train_model.py --csv healthwise.csv`, then download the `artifacts/` folder, and set the scikit-learn line in
`requirements.txt` to the version Colab printed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "The model file could not be loaded" | scikit-learn version mismatch. Pin the version that trained the file (see above). |
| Build fails on `pandas==` or `numpy==` | The chosen Python is too old. Pick Python 3.12+ in Advanced settings, or loosen those pins to `>=`. |
| "artifacts/... is missing" | The `artifacts/` folder was not uploaded. Upload it, or run `train_model.py`. |
| App shows "zzzz / wake up" | Free apps sleep when unused. Click the wake-up button; the first load takes a moment. |
| Fonts look different offline | The page falls back to Georgia and Segoe UI. Nothing is broken. |

## Notes

- Educational project. The engine is trained on a course dataset and is not for pricing real policies.
- The deployed Stage 1 classifier is a Decision Tree (lab task E1). Part B evaluates Logistic Regression; the Risk Sorter
  page shows both on the same split.
