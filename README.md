# Hospital Appointment No-Show Prediction

**Capstone Project — AI/ML Fundamentals**
Field-Based Scenario: MED-01 (MedTech)

## 1. Problem Statement

Private healthcare clinics lose significant operational capacity when patients miss scheduled appointments without cancelling in advance ("no-shows"). This project builds a machine learning model that estimates, **before the appointment takes place**, the probability that a given scheduled appointment will result in a no-show. Clinic operations staff can use this risk score to prioritize reminder calls, SMS confirmations, or manual follow-up for high-risk appointments.

## 2. Dataset

- **Source:** [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) — Kaggle, public dataset
- **Size:** 110,527 appointment records, Brazilian public healthcare system, April–June 2016
- **Target variable:** `No-show` (binary — did the patient miss the appointment?)
- **License:** Public, available for research/educational use

## 3. Repository Structure

```
├── notebooks/
│   └── EDA_AND_MODELING.ipynb    # Full pipeline: EDA, cleaning, modeling, evaluation
├── models/
│   ├── noshow_model.joblib       # Trained Random Forest model
│   └── model_columns.joblib      # Column order required for inference
├── data/
│   └── KaggleV2-May-2016.csv     # Raw dataset (see Setup below)
├── requirements.txt
└── README.md
```

## 4. Setup & How to Run (Google Colab)

1. Open `notebooks/EDA_AND_MODELING.ipynb` in Google Colab.
2. Download the dataset from the Kaggle link above (or use the copy in `data/`) and upload `KaggleV2-May-2016.csv` to the Colab session storage (left sidebar → Files → Upload).
3. Run all cells in order: **Runtime → Run all**.
4. The notebook will reproduce the full pipeline: data cleaning → EDA → preprocessing → baseline model → Random Forest → evaluation → saved model artifacts → inference demo.

No API keys or paid services are required. All dependencies are listed in `requirements.txt`.

## 5. Methodology Summary

| Step | Approach |
|---|---|
| Data cleaning | Removed 1 record with invalid negative age; removed 5 records with invalid negative waiting time; converted `Handcap` to binary |
| Feature engineering | Derived `WaitingDays` (days between scheduling and appointment); one-hot encoded `Neighbourhood`; binary-encoded `Gender` |
| Split | Stratified 80/20 train/test split (preserves ~80/20 class balance) |
| Baseline | Logistic Regression, compared against a majority-class Dummy Classifier |
| Main model | Random Forest Classifier (`n_estimators=200`, `max_depth=10`, `class_weight='balanced'`) |
| Primary metric | Recall / F1-score (prioritized over accuracy due to class imbalance and the higher operational cost of missing a true no-show) |

## 6. Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (majority class) | 0.798 | – | 0.000 | 0.000 | – |
| Logistic Regression | 0.796 | 0.395 | 0.017 | 0.033 | 0.659 |
| Logistic Regression (balanced) | 0.653 | 0.308 | 0.576 | 0.401 | 0.665 |
| **Random Forest (final model)** | 0.568 | 0.298 | **0.839** | **0.439** | **0.717** |

**Final model choice:** Random Forest was selected as the final model. Although its accuracy is lower than the naive baseline, this is expected and acceptable for an imbalanced classification problem: accuracy alone is misleading here, since a model that always predicts "will show up" achieves ~80% accuracy while being operationally useless. Random Forest correctly identifies 83.9% of true no-shows (recall), which is the metric that matters most for the clinic's use case — missing a genuine no-show has a higher operational cost than an unnecessary reminder call.

**Top predictive features:** `WaitingDays` (68%), `Age` (11%), `SMS_received` (11%) — the number of days between scheduling and the appointment is by far the strongest predictor of no-show behavior.

## 7. Inference Demo

The notebook includes a reusable `predict_noshow()` function that accepts a new appointment's details and returns a predicted class plus a no-show probability. It validates inputs (rejects invalid ages, missing fields) and handles unseen categories (e.g., unfamiliar neighbourhoods) gracefully. See the "Inference Pipeline" section of the notebook for usage examples and edge-case tests.

## 8. Limitations & Responsible AI

- **Generalizability:** the dataset is from a single healthcare system (Brazil, 2016) and may not directly generalize to other regions or time periods.
- **Fairness:** `Neighbourhood` and demographic fields are used as predictors; if deployed, model outputs should be periodically audited across subgroups to avoid systematically over-targeting certain neighbourhoods or demographics with reminder calls.
- **Not a clinical tool:** this model estimates operational no-show risk only. It must not be used to deny care, alter clinical decisions, or automatically cancel appointments.
- **Threshold sensitivity:** the current model favors recall over precision; the decision threshold should be revisited based on the clinic's actual cost of a missed reminder vs. an unnecessary one.
- **Privacy:** the dataset is anonymized; no real patient-identifiable information is used in this project.

## 9. Author

Capstone project developed as part of the AI/ML Fundamentals course.
