## Predicting Fake Job Postings Using Machine Learning  

### Overview  
This project aims to **detect fraudulent job postings** using **machine learning models**. By analyzing text-based job descriptions, we apply various **classification algorithms** to distinguish between real and fake job listings.  

### 🛠️ Methodology  

#### **1️⃣ Data Preprocessing**
- **Tokenization & Normalization**  
- **Lemmatization & Stop-word Removal**  
- **TF-IDF Vectorization** for feature extraction  

#### **2️⃣ Models Used & Evaluation**
- **Naive Bayes** (Baseline)  
- **Support Vector Machine (SVM)**  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Random Forest Classifier** 🌲  
- **Stacked Classifier - Final Model 🚀**  
  - Uses **Random Forest & KNN**  
  - **Gradient Boosting Classifier** as the final estimator  

#### **3️⃣ Hyperparameter Tuning**
- **Manual tuning** for Logistic Regression, KNN, and Random Forest  
- **Automated tuning** using **RandomizedSearchCV** for **Gradient Boosting Classifier**  

### 📊 Results  
- **Final Model:** Stacking Classifier  
- **Best F1 Score:** `0.7542`  
- **Best Recall Score:** `0.82`  
- **Key Observations:**
  - **Gradient Boosting Classifier** improved F1 Score from `0.658` → `0.748`  
  - **Stacking Classifier** further optimized accuracy and recall  

### 📂 Repository Structure  
┣ 📜 Data/ # Dataset files <br>
┣ 📜 Code/ # .py files 

### 🛠️ Technologies Used  
- **Python** (Pandas, NumPy, Scikit-learn, SciPy)  
- **Machine Learning** (SVM, Logistic Regression, Random Forest, Gradient Boosting)  
- **Feature Engineering** (TF-IDF Vectorization)  
- **Hyperparameter Optimization** (RandomizedSearchCV)  
