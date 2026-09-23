# 🎓 Student Dropout Prediction

Predicting which online learners are likely to **withdraw from a course**, so instructors can step in before it's too late.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 📌 Problem

Learning and training management systems (LMS/TMS) usually find out a student has dropped out only after it happens. Flagging at-risk students early lets instructors intervene, which raises completion rates and improves learning outcomes.

**Goal:** a binary classifier that predicts `dropout = 1` (final result is *Withdrawn*) from a student's demographics, assessment performance, and learning-platform activity.

Because missing an at-risk student costs more than sending an unnecessary check-in, the model is tuned for **recall**.

## 🏆 Results

All three models used a 0.3 decision threshold, chosen to favour recall. Scores are on a stratified 20% hold-out set (7,880 rows).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.789** | **0.649** | 0.858 | **0.739** | **0.887** |
| Random Forest | 0.765 | 0.617 | 0.851 | 0.715 | 0.864 |
| Logistic Regression (balanced) | 0.756 | 0.604 | **0.859** | 0.709 | 0.851 |

XGBoost also held up under **5-fold stratified cross-validation** at its default 0.5 threshold, with a ROC-AUC of **0.889 ± 0.001** and an F1 of **0.725 ± 0.004**.

**What that means:** at the 0.3 threshold, the XGBoost model catches about **86% of students who go on to withdraw**.

## 🗂️ Dataset

The project uses the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset):

| File | Rows | Used for |
|---|---|---|
| `studentInfo.csv` | 32,593 | Demographics and the target (`final_result`) |
| `studentAssessment.csv` | 173,912 | Assessment scores and submission timing |
| `studentVle.csv` | 10,655,280 | Daily click activity on the learning platform |
| `studentRegistration.csv` | 32,593 | Registration timing |

> The CSVs are not included in this repo. Download OULAD and place the files in a `dataset/` folder next to the notebook.

## 🔧 Approach

1. **Cleaning.** Drop nulls and duplicates, keep the relevant columns, and build the target `dropout = (final_result == "Withdrawn")`.
2. **Feature engineering** (aggregated per student):
   - *Assessment:* average, max and min score, score standard deviation, number of assessments, last submission day
   - *Engagement:* total clicks, active days, first and last activity day, activity span, clicks per day
   - *Registration:* registration day and a `registered_late` flag
3. **EDA and preprocessing.** Log-transform the skewed features, winsorize outliers, drop features that are highly correlated with each other (|r| > 0.8), one-hot encode the categoricals, and standardize.
4. **Modeling.** Start with a Logistic Regression baseline, add class-weight tuning, then compare against Random Forest and XGBoost.
5. **Evaluation.** Precision/recall tables across thresholds, ROC curves, cross-validation, and false-negative error analysis.

## 💡 Key Insights

- **Engagement is the strongest signal.** How late a student is still active and submitting work correlates with dropout at about −0.58. *Recommendation:* send automated alerts when a student goes inactive for a long stretch.
- **Academic performance matters.** Lower average scores mean higher withdrawal risk. *Recommendation:* offer targeted academic support when a student's grades start to slip.
- **Blind spot.** The students the model misses (false negatives) look academically healthy and stay active late into the course. Their withdrawals are probably driven by factors outside the data, such as personal, financial, or work reasons.

## ⚠️ Limitations & Next Steps

- Features like `last_activity_day` summarize the *whole* course. A real early-warning system should only use data available up to a cut-off week (for example, the first 4–8 weeks).
- Students enrolled in several modules are aggregated across all of them. Modeling each student-module pair separately would be more precise.
- Next steps: SHAP explanations, hyperparameter search, and a small dashboard that surfaces at-risk students.

## 🚀 Run It

```bash
git clone https://github.com/NHDungxNo1/Dropout-Prediction.git
cd Dropout-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost jupyter
# put the OULAD CSVs in ./dataset/
jupyter notebook DROP_OUT_PRED.ipynb
```

## 👤 Author

**Dung Ngo** · [GitHub @NHDungxNo1](https://github.com/NHDungxNo1)
