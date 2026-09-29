# ❤️ Heart Disease Prediction Using Machine Learning

## CardioPredict — Machine Learning Based Heart Disease Prediction Web Application

---

## 📌 Project Overview

Heart disease is one of the major health-related challenges worldwide. Early identification of individuals who may be at risk can support further medical examination and clinical decision-making.

**CardioPredict** is a Machine Learning-based Heart Disease Prediction System developed using **Python, NumPy, Pandas, SciPy, Scikit-learn, imbalanced-learn, Flask, HTML, and CSS**.

The system accepts selected patient health parameters as input and uses a trained **Gaussian Naive Bayes classification model** to generate a prediction.

This project demonstrates a complete Machine Learning workflow, starting from data preprocessing and continuing through variable transformation, feature selection, class balancing, feature scaling, model training, model evaluation, model serialization, and Flask-based web deployment.

---

# 🎯 Objectives

The main objectives of this project are:

1. To analyze a heart disease dataset.
2. To preprocess the raw dataset.
3. To perform variable transformation using the Yeo-Johnson technique.
4. To select relevant features for machine learning.
5. To handle class imbalance using SMOTE.
6. To standardize the selected features.
7. To train a Gaussian Naive Bayes classification model.
8. To evaluate the trained model using classification metrics.
9. To save the trained model and preprocessing objects.
10. To develop a Flask-based backend.
11. To create a responsive HTML/CSS frontend.
12. To integrate the trained Machine Learning model with the Flask application.
13. To allow users to enter patient information through a web form.
14. To display the generated prediction through the web interface.

---

# 🧠 Project Concept

The complete project follows the workflow below:

```text
                    HEART DISEASE DATASET
                              │
                              ▼
                     DATA PREPROCESSING
                              │
                              ▼
                  VARIABLE TRANSFORMATION
                     YEO-JOHNSON
                              │
                              ▼
                     FEATURE SELECTION
                              │
                              ▼
                           SMOTE
                    CLASS BALANCING
                              │
                              ▼
                       FEATURE SCALING
                       STANDARD SCALER
                              │
                              ▼
                    MODEL TRAINING
                GAUSSIAN NAIVE BAYES
                              │
                              ▼
                    MODEL EVALUATION
                              │
                              ▼
                    SAVE MODEL FILES
                              │
                              ▼
                       FLASK BACKEND
                              │
                              ▼
                    WEB APPLICATION
                              │
                              ▼
                    PATIENT INPUT FORM
                              │
                              ▼
                       ML PREDICTION
                              │
                              ▼
                     RESULT DISPLAY
```

---

# 🚀 Key Features

- Heart disease prediction using Machine Learning
- Data preprocessing
- Yeo-Johnson variable transformation
- Feature selection
- SMOTE-based class balancing
- StandardScaler feature scaling
- Gaussian Naive Bayes classification
- Saved trained ML model
- Flask backend
- HTML5 frontend
- CSS3 styling
- Responsive user interface
- Patient information form
- Prediction result display
- Machine Learning pipeline information
- Professional healthcare-oriented interface
- Local execution through PyCharm

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                 ┌──────────────────┐
                 │   WEB INTERFACE  │
                 │    HTML + CSS    │
                 └─────────┬────────┘
                           │
                           │ Patient Data
                           ▼
                 ┌──────────────────┐
                 │  FLASK BACKEND   │
                 │     app.py       │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ INPUT PROCESSING │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ FEATURE PIPELINE │
                 │ Transformation   │
                 │ Selection        │
                 │ Scaling          │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  TRAINED MODEL   │
                 │ Gaussian NB      │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │    PREDICTION    │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │   RESULT PAGE    │
                 └──────────────────┘
```

---

# 📊 Dataset

The project uses a Heart Disease Dataset containing patient-related medical and physiological attributes.

The raw dataset is processed before being supplied to the Machine Learning algorithm.

The dataset contains numerical and categorical variables, which require appropriate preprocessing before model training.

The preprocessing pipeline consists of:

- Data inspection
- Data cleaning
- Variable transformation
- Feature selection
- Train-test splitting
- Class balancing
- Feature scaling
- Model training

---

# 📋 Features Used for Prediction

The web application accepts seven selected features.

| No. | Feature | Dataset Name | Description |
|---:|---|---|---|
| 1 | Age | `age` | Age of the patient |
| 2 | Sex | `sex` | Sex of the patient |
| 3 | Chest Pain Type | `cp` | Type of chest pain |
| 4 | Maximum Heart Rate | `thalach` | Maximum heart rate achieved |
| 5 | ST Depression | `oldpeak` | ST depression induced by exercise |
| 6 | Slope | `slope` | Slope of peak exercise ST segment |
| 7 | Thalassemia | `thal` | Thalassemia classification |

The feature order must remain consistent with the order expected by the trained preprocessing pipeline and model.

---

# 📌 Feature Details

## 1. Age

The `age` feature represents the patient's age in years.

Example:

```text
Age = 33
```

## 2. Sex

The `sex` feature represents the sex encoding used by the dataset.

Common encoding:

```text
0 → Female
1 → Male
```

Example:

```text
Sex = 1
```

## 3. Chest Pain Type

The `cp` feature represents the type of chest pain.

Dataset encoding:

```text
0 → Typical Angina
1 → Atypical Angina
2 → Non-anginal Pain
3 → Asymptomatic
```

Example:

```text
cp = 3
```

## 4. Maximum Heart Rate

The `thalach` feature represents the maximum heart rate achieved during the exercise test.

Example:

```text
thalach = 169
```

## 5. ST Depression

The `oldpeak` feature represents ST depression induced by exercise relative to rest.

Example:

```text
oldpeak = 0.7
```

## 6. Slope

The `slope` feature represents the slope of the peak exercise ST segment.

Dataset encoding:

```text
0 → Downsloping
1 → Flat
2 → Upsloping
```

Example:

```text
slope = 0
```

## 7. Thalassemia

The `thal` feature represents the thalassemia-related classification used in the dataset.

Example:

```text
thal = 2
```

The interpretation of categorical values must remain consistent with the encoding used during model training.

---

# 🧹 Data Preprocessing

Data preprocessing is an important stage of the project.

The preprocessing pipeline is:

```text
Raw Dataset
     │
     ▼
Data Inspection
     │
     ▼
Data Cleaning
     │
     ▼
Variable Transformation
     │
     ▼
Feature Selection
     │
     ▼
Train-Test Split
     │
     ▼
SMOTE
     │
     ▼
Feature Scaling
     │
     ▼
Model Training
```

---

# 🔄 Variable Transformation

The project uses the **Yeo-Johnson Transformation** for selected numerical variables.

Yeo-Johnson is a power transformation technique that can handle zero and negative values.

It can be useful when numerical variables have skewed distributions.

Example:

```python
from scipy.stats import yeojohnson

transformed_data, lambda_value = yeojohnson(data)
```

---

# 🎯 Feature Selection

Feature selection is performed to retain relevant features and remove unnecessary variables.

The project removes the following transformed variables:

```text
fbs_yeo_trim
trestbps_yeo_trim
chol_yeo_trim
exang_yeo_trim
ca_yeo_trim
restecg_yeo_trim
```

Feature selection can help:

- Reduce unnecessary variables
- Reduce dimensionality
- Reduce computational requirements
- Remove potentially irrelevant information
- Simplify the Machine Learning pipeline
- Improve model interpretability

---

# ⚖️ Handling Class Imbalance

A classification dataset may contain different numbers of samples in different classes.

This situation is called **class imbalance**.

The project uses:

## SMOTE

SMOTE stands for:

> Synthetic Minority Over-sampling Technique

SMOTE creates synthetic observations for the minority class.

Conceptually:

```text
Minority Class Samples
          │
          ▼
   Find Nearest Neighbors
          │
          ▼
 Generate Synthetic Samples
          │
          ▼
Balanced Training Dataset
```

### Important

SMOTE should be applied to the training data, not to a single patient input.

Correct:

```text
Original Dataset
       │
       ▼
Train-Test Split
       │
       ▼
Training Data
       │
       ▼
     SMOTE
       │
       ▼
Balanced Training Data
```

For a new patient:

```text
Patient Input
      │
      ▼
Preprocessing
      │
      ▼
Scaling
      │
      ▼
Trained Model
      │
      ▼
Prediction
```

---

# 📏 Feature Scaling

The project uses:

```python
StandardScaler
```

from Scikit-learn.

The standardization formula is:

```text
z = (x - μ) / σ
```

Where:

```text
x  = Original value
μ  = Mean
σ  = Standard deviation
z  = Standardized value
```

The scaler fitted during training should be reused during prediction.

---

# 🤖 Machine Learning Model

The primary classification algorithm used in this project is:

## Gaussian Naive Bayes

Gaussian Naive Bayes is a supervised Machine Learning classification algorithm based on Bayes' theorem.

It assumes conditional independence between features.

For numerical features, Gaussian Naive Bayes assumes that feature values approximately follow a Gaussian distribution within each class.

---

# 🧮 Bayes' Theorem

Bayes' theorem is represented as:

```text
P(C | X) = P(X | C) × P(C)
           ----------------
                 P(X)
```

Where:

```text
P(C | X) = Posterior probability
P(X | C) = Likelihood
P(C)     = Prior probability
P(X)     = Evidence
```

---

# 🏋️ Model Training

The training workflow is:

```text
Processed Dataset
       │
       ▼
Feature Selection
       │
       ▼
Train-Test Split
       │
       ▼
SMOTE
       │
       ▼
StandardScaler
       │
       ▼
Gaussian Naive Bayes
       │
       ▼
Trained Model
```

---

# 📊 Model Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Accuracy

```text
                 TP + TN
Accuracy = -------------------------
           TP + TN + FP + FN
```

## Precision

```text
              TP
Precision = ---------
             TP + FP
```

## Recall

```text
            TP
Recall = ---------
          TP + FN
```

## F1 Score

```text
                 Precision × Recall
F1 = 2 × -----------------------------
                   Precision + Recall
```

---

# 📊 Confusion Matrix

```text
                         Predicted
                      0          1

Actual 0             TN         FP

Actual 1             FN         TP
```

Where:

| Term | Meaning |
|---|---|
| TP | True Positive |
| TN | True Negative |
| FP | False Positive |
| FN | False Negative |

---

# 📋 Classification Report

Example:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

The classification report provides:

- Precision
- Recall
- F1-score
- Support

for each class.

---

# 💾 Model Serialization

After training, the trained model and required preprocessing objects can be serialized.

This allows the Flask application to load the trained artifacts without retraining the model every time the application starts.

The project contains:

```text
navi_bayes_model.pkl
Heart_Disease_Prediction_Model.pkl
```

These files are used according to the project's preprocessing and prediction pipeline.

---

# 🌐 Flask Web Application

Flask is used as the backend framework.

The Flask application connects the frontend with the Machine Learning model.

The user enters patient information through an HTML form.

The form sends the information to Flask.

Flask processes the information and passes the processed input to the trained model.

The prediction is then returned to the frontend.

---

# 🔗 Flask Request Flow

For the home page:

```text
Browser
   │
   │ GET /
   ▼
Flask
   │
   ▼
index.html
```

For prediction:

```text
Browser
   │
   │ POST /predict
   ▼
Flask
   │
   ▼
Read Form Data
   │
   ▼
Convert to Numerical Values
   │
   ▼
Apply Required Preprocessing
   │
   ▼
Apply Scaling
   │
   ▼
Load Trained Model
   │
   ▼
Generate Prediction
   │
   ▼
Return Result
   │
   ▼
Display Result
```

---

# 🖥️ Frontend

The frontend is developed using:

- HTML5
- CSS3
- Jinja2

The interface contains:

- Navigation bar
- Hero section
- Patient prediction form
- Prediction result
- Machine Learning pipeline
- About section
- Footer

---

# 📝 Prediction Form

The prediction form contains seven input fields:

```text
Age
Sex
Chest Pain Type
Maximum Heart Rate
ST Depression
Slope
Thalassemia
```

---

# 🧪 Example Test Input

A sample input used for testing is:

```python
import numpy as np

testing = np.array([
    [33, 1, 3, 169, 0.7, 0, 2]
])
```

Feature mapping:

```text
33   → Age
1    → Sex
3    → Chest Pain Type
169  → Maximum Heart Rate
0.7  → ST Depression
0    → Slope
2    → Thalassemia
```

Input shape:

```text
(1, 7)
```

This means:

```text
1 patient
7 features
```

---

# 🔢 Feature Order

The input feature order is:

```text
1. age
2. sex
3. cp
4. thalach
5. oldpeak
6. slope
7. thal
```

Therefore:

```python
testing = np.array([
    [33, 1, 3, 169, 0.7, 0, 2]
])
```

means:

```text
age     = 33
sex     = 1
cp      = 3
thalach = 169
oldpeak = 0.7
slope   = 0
thal    = 2
```

The order must not be changed.

---

# 🔄 Prediction Workflow

```text
User Enters Information
          │
          ▼
     HTML Form
          │
          ▼
    POST /predict
          │
          ▼
   Flask Receives Data
          │
          ▼
   Convert Input Values
          │
          ▼
 Create Input NumPy Array
          │
          ▼
 Apply Required Processing
          │
          ▼
 Apply Saved Scaler
          │
          ▼
 Load Saved ML Model
          │
          ▼
 Generate Prediction
          │
          ▼
 Send Result to Flask Template
          │
          ▼
 Display Result
```

---

# 🔐 Training vs Prediction Pipeline

## Training Pipeline

```text
Raw Dataset
     ↓
Preprocessing
     ↓
Yeo-Johnson Transformation
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
SMOTE on Training Data
     ↓
StandardScaler
     ↓
Gaussian Naive Bayes
     ↓
Save Model
```

## Prediction Pipeline

```text
New Patient Input
     ↓
Required Preprocessing
     ↓
Feature Transformation
     ↓
Feature Selection
     ↓
Previously Fitted StandardScaler
     ↓
Saved Gaussian Naive Bayes Model
     ↓
Prediction
```

---

# ⚠️ Important ML Implementation Rules

## 1. Do Not Apply SMOTE to a Single Patient

SMOTE is a training-data balancing technique.

Do not perform:

```text
Patient Input
     ↓
SMOTE
     ↓
Prediction
```

Instead:

```text
Training Dataset
     ↓
SMOTE
     ↓
Model Training
```

## 2. Maintain Feature Order

Correct:

```python
[age, sex, cp, thalach, oldpeak, slope, thal]
```

## 3. Use Compatible Preprocessing

The preprocessing applied during prediction must correspond to the preprocessing used during training.

## 4. Use the Saved Scaler

If the model was trained using scaled data, new input should be transformed using the scaler fitted during training.

## 5. Do Not Retrain During Prediction

The Flask application should load the already-trained model.

---

# 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
│
├── heart.csv
│
├── navi_bayes_model.pkl
│
├── Heart_Disease_Prediction_Model.pkl
│
├── requirements.txt
│
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

# 📄 File Descriptions

## `app.py`

Main Flask application.

Responsibilities:

- Start Flask
- Load the trained model
- Load preprocessing artifacts
- Receive form data
- Convert input values
- Perform required preprocessing
- Generate predictions
- Render HTML templates

## `heart.csv`

Heart disease dataset used for Machine Learning development.

## `navi_bayes_model.pkl`

Serialized Gaussian Naive Bayes model.

## `Heart_Disease_Prediction_Model.pkl`

Serialized preprocessing/scaling object used according to the project's Machine Learning pipeline.

## `templates/index.html`

Frontend HTML file containing:

- Navigation
- Hero section
- Prediction form
- Result section
- About section
- Pipeline
- Footer

## `static/style.css`

CSS file containing:

- Layout
- Colors
- Fonts
- Cards
- Forms
- Buttons
- Responsive design
- Navigation
- Animations

## `requirements.txt`

Python dependencies required by the application.

## `README.md`

Complete project documentation.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| SciPy | Statistical transformation |
| Scikit-learn | Machine Learning |
| imbalanced-learn | SMOTE |
| Flask | Backend web framework |
| Jinja2 | Dynamic HTML |
| HTML5 | Frontend structure |
| CSS3 | Frontend styling |
| Pickle | Model serialization |
| PyCharm | Development environment |

---

# 📦 Installation

## Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
```

Move into the project directory:

```bash
cd heart-disease-prediction
```

## Step 2 — Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install flask numpy pandas scikit-learn scipy imbalanced-learn xgboost
```

## Step 4 — Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

---

# 💻 Running in PyCharm

1. Open the project in PyCharm.
2. Verify the project structure.
3. Open the PyCharm terminal.
4. Install the dependencies.
5. Right-click `app.py`.
6. Select **Run 'app'**.
7. Open:

```text
http://127.0.0.1:5000/
```

To stop the server:

```text
CTRL + C
```

---

# 🧪 Example Evaluation Code

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

# 🔬 Project Methodology

## Phase 1 — Data Collection

The heart disease dataset is loaded into the Python environment.

## Phase 2 — Data Analysis

The dataset is inspected to understand:

- Number of observations
- Number of variables
- Data types
- Missing values
- Feature distributions
- Target distribution

## Phase 3 — Data Preprocessing

The raw dataset is prepared for Machine Learning.

## Phase 4 — Variable Transformation

Yeo-Johnson transformation is applied to selected numerical variables.

## Phase 5 — Feature Selection

Relevant features are retained and unnecessary transformed features are removed.

## Phase 6 — Train-Test Split

The dataset is divided into training and testing data.

## Phase 7 — Class Balancing

SMOTE is applied to the training data.

## Phase 8 — Feature Scaling

StandardScaler is fitted using the training data.

## Phase 9 — Model Training

Gaussian Naive Bayes is trained.

## Phase 10 — Model Evaluation

The model is evaluated using classification metrics.

## Phase 11 — Model Serialization

The trained model and required preprocessing objects are saved as `.pkl` files.

## Phase 12 — Flask Integration

The saved model is loaded into the Flask application.

## Phase 13 — Web Prediction

Users enter patient information and receive a model-generated prediction.

---

# 🔒 Security Considerations

For a real-world deployment, additional security measures should be implemented:

- HTTPS
- Authentication
- Authorization
- Input validation
- Secure database storage
- Encryption
- Secure model storage
- Logging
- Monitoring
- Protection of patient information

The current project is primarily intended for academic and educational purposes.

---

# 🌍 Deployment

The Flask application can potentially be deployed to a cloud platform supporting Python applications.

Possible architecture:

```text
User Browser
     │
     ▼
Cloud Hosting
     │
     ▼
Flask Application
     │
     ▼
Machine Learning Model
     │
     ▼
Prediction
     │
     ▼
User
```

Before production deployment, additional testing, security, privacy, reliability, and clinical validation would be required.

---

# 🔮 Future Enhancements

## 1. Multiple Machine Learning Models

Additional algorithms can be trained and compared:

```text
Logistic Regression
K-Nearest Neighbors
Decision Tree
Random Forest
Support Vector Machine
Gradient Boosting
XGBoost
LightGBM
```

## 2. Hyperparameter Optimization

Possible methods:

```text
GridSearchCV
RandomizedSearchCV
Bayesian Optimization
```

## 3. Explainable AI

Possible techniques:

```text
SHAP
LIME
Feature Importance
```

## 4. Prediction Probability

The application can display model probability estimates where supported.

## 5. Database Integration

A database can store:

- Prediction history
- User records
- Timestamps
- Model results

## 6. Authentication

Possible flow:

```text
Registration
     ↓
Login
     ↓
Dashboard
     ↓
Prediction
     ↓
Prediction History
```

## 7. Interactive Dashboard

Possible components:

- Dataset statistics
- Model performance
- Confusion matrix
- Feature distributions
- Prediction history
- Model comparison
- Charts

---

# 📊 Possible Dashboard

```text
┌─────────────────────────────────────────┐
│          HEART DISEASE ANALYTICS        │
├─────────────────────────────────────────┤
│                                         │
│  Total Patients       Model Accuracy    │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│  Class Distribution    Confusion Matrix │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│  Feature Analysis      Model Comparison │
│                                         │
└─────────────────────────────────────────┘
```

---

# 🎓 Learning Outcomes

This project provides practical experience in:

## Python

- NumPy
- Pandas
- Functions
- Arrays
- Data processing
- Model serialization

## Data Science

- Data preprocessing
- Data analysis
- Variable transformation
- Feature selection
- Feature scaling
- Class balancing

## Machine Learning

- Classification
- Gaussian Naive Bayes
- Train-test split
- SMOTE
- Model evaluation
- Confusion matrix
- Classification report

## Web Development

- Flask
- HTML5
- CSS3
- Jinja2
- Forms
- Backend integration

## Software Development

- Project organization
- Virtual environments
- Requirements management
- Model serialization
- Local deployment

---

# 📌 Project Limitations

The current project has several limitations:

1. The model depends on the dataset used for training.
2. Prediction quality depends on the quality and representativeness of the training data.
3. The model may not generalize equally to all populations.
4. The application is an academic demonstration.
5. The system has not been presented as a clinically validated diagnostic system.
6. Only selected features are exposed through the web interface.
7. Additional validation would be required before real-world healthcare deployment.
8. Model performance may change when the input data distribution differs from the training data.

---

# ⚠️ Medical Disclaimer

This project is developed for:

```text
Academic
Educational
Research
Demonstration
```

purposes.

The predictions generated by this application should **not be considered a medical diagnosis**.

The application should not be used as a replacement for:

- Medical professionals
- Clinical examinations
- Laboratory tests
- Medical imaging
- Professional diagnosis
- Medical treatment

For health-related concerns, users should consult qualified healthcare professionals.

---

# 📸 Screenshots

You can add screenshots of the application using:

```text
screenshots/
├── home.png
├── prediction.png
└── result.png
```

Add them to the README using:

```markdown
## Home Page

![Home Page](screenshots/home.png)

## Prediction Page

![Prediction Page](screenshots/prediction.png)

## Prediction Result

![Prediction Result](screenshots/result.png)
```

---

# 🎥 Project Demonstration

A project demonstration can show:

1. Opening the application
2. Navigating to the prediction section
3. Entering patient details
4. Submitting the form
5. Flask receiving the request
6. Preprocessing the input
7. Loading the saved model
8. Generating the prediction
9. Displaying the prediction on the webpage

---

# 📝 requirements.txt

The `requirements.txt` file can contain:

```text
Flask
numpy
pandas
scikit-learn
scipy
imbalanced-learn
xgboost
```

---

# 🔗 GitHub Repository Description

```text
A complete Machine Learning-based Heart Disease Prediction web application developed using Python, Scikit-learn, Gaussian Naive Bayes, Yeo-Johnson transformation, feature selection, SMOTE, StandardScaler, Flask, HTML, and CSS.
```

---

# 🏷️ GitHub Topics

```text
machine-learning
machine-learning-project
heart-disease
heart-disease-prediction
python
data-science
flask
scikit-learn
naive-bayes
gaussian-naive-bayes
smote
feature-selection
feature-scaling
pandas
numpy
scipy
html
css
healthcare
healthcare-analytics
btech-project
final-year-project
```

---

# 🎓 Academic Project Information

```text
Project Title:
Heart Disease Prediction Using Machine Learning

Project Name:
CardioPredict

Domain:
Machine Learning / Data Science

Application Domain:
Healthcare Analytics

Programming Language:
Python

Machine Learning Algorithm:
Gaussian Naive Bayes

Backend:
Flask

Frontend:
HTML5 + CSS3

Development Environment:
PyCharm
```

---

# 🧠 Complete Machine Learning Pipeline

```text
                       DATASET
                          │
                          ▼
                    DATA ANALYSIS
                          │
                          ▼
                  DATA PREPROCESSING
                          │
                          ▼
              YEO-JOHNSON TRANSFORMATION
                          │
                          ▼
                   FEATURE SELECTION
                          │
                          ▼
                   TRAIN-TEST SPLIT
                          │
                          ▼
                         SMOTE
                          │
                          ▼
                    STANDARD SCALER
                          │
                          ▼
               GAUSSIAN NAIVE BAYES
                          │
                          ▼
                   MODEL EVALUATION
                          │
                          ▼
                    SAVE MODEL
                          │
                          ▼
                  FLASK INTEGRATION
                          │
                          ▼
                   WEB APPLICATION
                          │
                          ▼
                    USER INPUT
                          │
                          ▼
                     PREDICTION
                          │
                          ▼
                    RESULT DISPLAY
```

---

# 🔄 Complete Application Flow

```text
                     USER
                      │
                      ▼
              Open Web Application
                      │
                      ▼
              Enter Patient Data
                      │
                      ▼
                Submit Form
                      │
                      ▼
                Flask Backend
                      │
                      ▼
              Validate Input
                      │
                      ▼
             Prepare Input Array
                      │
                      ▼
             Apply Preprocessing
                      │
                      ▼
              Apply StandardScaler
                      │
                      ▼
             Load Saved ML Model
                      │
                      ▼
                  Prediction
                      │
                      ▼
              Return Prediction
                      │
                      ▼
                Web Interface
                      │
                      ▼
                 User Result
```

---

# 🏆 Project Highlights

```text
✔ End-to-End Machine Learning Project
✔ Heart Disease Prediction
✔ Data Preprocessing
✔ Yeo-Johnson Transformation
✔ Feature Selection
✔ SMOTE
✔ StandardScaler
✔ Gaussian Naive Bayes
✔ Model Evaluation
✔ Model Serialization
✔ Flask Backend
✔ HTML5 Frontend
✔ CSS3 Styling
✔ Responsive Design
✔ PyCharm Development
✔ Web-Based Prediction
```

---

# 📚 Conclusion

The **Heart Disease Prediction Using Machine Learning** project demonstrates how a Machine Learning classification model can be developed and integrated into a web application.

The project begins with a raw heart disease dataset and applies preprocessing, variable transformation, feature selection, class balancing, and feature scaling.

A Gaussian Naive Bayes model is then trained and evaluated.

The trained model and required preprocessing objects are saved and integrated with a Flask backend.

A responsive HTML/CSS interface allows users to enter selected patient parameters and obtain a Machine Learning prediction.

The project demonstrates an end-to-end workflow:

```text
Data
  ↓
Preprocessing
  ↓
Transformation
  ↓
Feature Selection
  ↓
SMOTE
  ↓
Scaling
  ↓
Machine Learning
  ↓
Evaluation
  ↓
Model Saving
  ↓
Flask
  ↓
Web Application
  ↓
Prediction
```

---

# ❤️ CardioPredict

## Heart Disease Prediction Using Machine Learning

**Data → Preprocessing → Feature Engineering → Machine Learning → Flask → Prediction**

---

# 👩‍💻 Author

**Your Name**

**B.Tech – Information Technology**

**Final Year Project**

---

## ⭐ Project Summary

CardioPredict is an academic Machine Learning project that demonstrates the integration of:

```text
Python
+
Data Science
+
Machine Learning
+
Statistical Processing
+
Feature Engineering
+
SMOTE
+
Gaussian Naive Bayes
+
Flask
+
HTML
+
CSS
```

into a complete web-based Heart Disease Prediction application.
