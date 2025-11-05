# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using **Random Forest** and **XGBoost** classifiers. The notebook performs **data analysis, visualization, feature selection**, and **model comparison** to determine which algorithm performs better at identifying fraud.

---

## 📂 Dataset

The dataset used is `creditcard_2023.csv`, containing **568,630 transactions** with **31 columns**, including:
- **Features:** V1–V28, `Amount`
- **Target:** `Class` (0 = Non-Fraud, 1 = Fraud)
- **ID:** Transaction identifier

**Key Notes:**
- No missing values.
- No duplicates.
- Dataset is **perfectly balanced** (284,315 fraud and 284,315 non-fraud).


---

## 🔍 Exploratory Data Analysis (EDA)

- Inspected column names, datatypes, and unique counts.
- Verified absence of null and duplicate records.
- Visualized class distribution using `Seaborn`.
- Generated **correlation heatmap** to identify feature relationships.

### 📈 Observations
- Features most correlated with `Class`: **V1, V3, V4, V9, V10, V11, V12, V14, V16**
- Strong inter-feature correlations found between **V10–V14–V16–V17–V18**.

---

## 🧠 Feature Selection

Selected features with correlation above `0.5` or below `-0.5` with the target:

```
['V2', 'V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17']
```

These features were used for training and testing.

---

## 🧩 Data Splitting

```python
X = df[feature_name]
y = df[['Class']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Training & Comparison

### 1️⃣ Random Forest Classifier
- **Hyperparameter tuning** with `RandomizedSearchCV`
- Evaluated using **accuracy**, **recall**, and **AUC**.

### 2️⃣ XGBoost Classifier
- Used same features and split.
- Evaluated using **accuracy**, **recall**, and **AUC**.
---

## 📊 Results

| Model | Accuracy | Recall | AUC |
|:------|:---------:|:------:|:---:|
| Random Forest | 0.973 | 0.970 | 0.98 |
| XGBoost | 0.981 | 0.978 | 0.99 |

✅ **XGBoost** outperformed Random Forest with higher accuracy and recall.


---

## 🧾 Conclusion
- Both models achieved high performance.
- **XGBoost** proved superior in recall, which is crucial in fraud detection.
- The project demonstrates practical handling of class balance, feature selection, and model evaluation.

---

## 🧩 Future Improvements
- Integrate **SHAP** or **LIME** for model explainability.
- Deploy best model using **Streamlit** or **FastAPI**.
- Include **real-time prediction API**.
