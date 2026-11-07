# Residency Project Day 1: Data Preprocessing and Initial Model Development

Real-world AI-driven analytics solution using Python. Heart disease prediction with the UCI Heart Disease (Cleveland) dataset.

**Team:** Damaries Ongole, Iman Parwaiz Sayeda, Sanjeev Koushik Sennam Setty

## Problem

Can routinely collected patient attributes (age, sex, chest pain type, blood pressure, cholesterol, and exercise test results) predict whether a patient has heart disease?

Binary classification: target is 1 for heart disease present, 0 for absent. Because a missed diagnosis is more costly than a false alarm, we prioritized recall for the disease class alongside accuracy and ROC-AUC. The model is decision support only, never a replacement for clinical judgment.

## Dataset

- UCI Heart Disease (Cleveland), 303 records, 13 predictors and one binary target
- No missing values, 1 duplicate record removed, leaving 302 records
- Classes are fairly balanced: 164 patients (54.3%) with disease, 138 (45.7%) without
- Ages 29 to 77, mean 54.4

## Preprocessing

- Checked missing values and removed 1 duplicate with drop_duplicates()
- One-hot encoded categorical variables: sex, cp, fbs, restecg, exang, slope, ca, thal
- Standardized numeric variables with StandardScaler: age, trestbps, chol, thalach, oldpeak
- 75/25 stratified train/test split, 226 training and 76 test records, random_state = 42
- Scaling and encoding were fitted inside a scikit-learn Pipeline on training data only, so no test information leaked into training

## Models and results

Logistic Regression baseline and Random Forest, evaluated with 5-fold cross-validation and on the held-out test set. Random Forest was tuned with GridSearchCV (n_estimators 100 or 200, max_depth 10, 20, or None). Best settings: max_depth = 10, n_estimators = 100.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.803 | 0.783 | 0.878 | 0.828 | 0.895 |
| Random Forest (default) | 0.750 | 0.739 | 0.829 | 0.782 | 0.879 |
| Random Forest (tuned) | 0.737 | 0.723 | 0.829 | 0.773 | 0.861 |

Logistic Regression performed best on the test set and is preferred for transparency. Most influential features: chest pain type, thal, number of vessels (ca), ST depression (oldpeak), and maximum heart rate (thalach).

## Files

- `heart_disease_analysis.py` - EDA, preprocessing, Logistic Regression and Random Forest, tuning, confusion matrices, and feature importance
- `Residency_Project_Day1.docx` - Day 1 report
- `Residency_Project_Day_1_Team_Kickoff.pptx` - Team kickoff slides

## How to run

The script expects the UCI Cleveland data as `heart.csv` in the same folder (it is written to run in Google Colab: upload heart.csv first).

```
pip install pandas numpy matplotlib seaborn scikit-learn
python heart_disease_analysis.py
```

Outputs: eda1.png, eda2.png, cm.png, plus printed metrics.

## Team process

- Shared Google Colab notebook for coding and this GitHub repository for version control
- One branch per task, a teammate reviews each pull request before merging
- Roles: exploration and visualization (Damaries Ongole), preprocessing and evaluation (Iman Parwaiz Sayeda), problem framing, documentation, and modeling (Sanjeev Koushik Sennam Setty)
- Key decisions: remove the duplicate record, use a stratified split, prioritize recall

## Limitations

Data is from a single clinic with 302 records, so results may not generalize. Before any real use, compare recall and false-negative rates by sex and age group, keep the model interpretable, and handle patient data under HIPAA-compliant controls.
