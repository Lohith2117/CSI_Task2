# CSI_Task2
This README file provides a clear, professional overview of your Credit Card Anomaly Detection project, covering the "why" and "how" of your implementation.

---

# Credit Card Transaction Anomaly Detection

This project implements a hybrid machine learning approach to detect fraudulent credit card transactions using both **unsupervised** (Isolation Forest) and **supervised** (XGBoost) methods.

## 📌 Project Overview
Fraud detection presents a significant challenge due to extreme **class imbalance** (fraud typically represents <2% of data). This project demonstrates how to handle such imbalance using specialized algorithms and resampling techniques like **SMOTE**.

### Key Features:
* **Exploratory Data Analysis (EDA):** Visualizing class distributions, transaction amounts, and fraud categories.
* **Feature Engineering:** Calculating distances between merchants and cardholders and extracting temporal data (hour/day).
* **Dual-Model Approach:**
    * **Isolation Forest:** An unsupervised model that identifies anomalies based on how "easy" they are to isolate from the rest of the data.
    * **XGBoost + SMOTE:** A supervised gradient boosting model trained on balanced data generated via Synthetic Minority Over-sampling Technique.
* **Performance Metrics:** Focuses on **Precision-Recall**, **F1-Score**, and **ROC-AUC** instead of simple accuracy.

---

## 📊 Dataset
The project uses the [Credit Card Transactions Dataset from Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset).
* **File Required:** `final_dataset.csv`
* **Target Variable:** `is_fraud` (0 = Normal, 1 = Fraud)

---

## 🚀 Installation & Setup
1. **Clone or Download** this repository.
2. **Install Dependencies** via terminal:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```
3. **Place the Dataset** in the same directory as the script.
4. **Run the Script:**
   ```bash
   python credit_card_anomaly.py
   ```

---

## 📈 Methodology
### 1. Preprocessing
* Removal of non-predictive identifiers (names, street addresses, etc.).
* Haversine distance approximation for merchant-to-home distance.
* Label encoding for categorical variables and standard scaling for numerical features.

### 2. Models
| Model | Type | Logic |
| :--- | :--- | :--- |
| **Isolation Forest** | Unsupervised | Isolates anomalies through random partitioning; frauds typically have shorter path lengths. |
| **XGBoost** | Supervised | Optimized gradient boosting; uses SMOTE to balance the training set for better minority class recognition. |

---

## 🖼️ Generated Visualizations
The script automatically generates and saves the following plots:
1.  `plot_1_class_distribution.png`: Shows the degree of imbalance.
2.  `plot_2_amount_distribution.png`: Compares spending patterns.
3.  `plot_4_iso_confusion.png` & `plot_5_xgb_confusion.png`: Confusion matrices for both models.
4.  `plot_6_feature_importance.png`: Identifies the strongest predictors of fraud.
5.  `plot_7_model_comparison.png`: ROC and Precision-Recall curves.

---

## 🏁 Results & Analysis
The Supervised model (XGBoost) generally outperforms the unsupervised model because it utilizes the historical labels. However, the Isolation Forest remains a robust baseline for scenarios where labels are unavailable or fraud patterns shift rapidly.

**Key takeaway:** In fraud detection, the **Average Precision (AP)** is our "North Star" metric, as it focuses on the model's ability to find the rare positive cases without flagging too many false alarms.

---

## 🛠️ Limitations & Future Scope
* **Synthetic Samples:** SMOTE may create unrealistic data points; future versions could test ADASYN or simple undersampling.
* **Temporal Features:** Adding "velocity" features (e.g., number of transactions in the last hour) would likely improve detection of card-testing attacks.
