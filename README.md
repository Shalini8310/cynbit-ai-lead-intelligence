# AI Lead Intelligence Assistant

An AI/ML-powered prototype that analyzes customer enquiries, classifies service requirements, estimates lead priority, and recommends suitable follow-up actions.

## Project Overview

Businesses receive customer enquiries across different services. Manually reviewing and categorizing these enquiries can take time.

This project demonstrates how Natural Language Processing (NLP) and Machine Learning can help automate the initial enquiry analysis process.

## Key Features

- **Service Classification:** Classifies enquiries into AI Development, Web Development, App Development, Cloud Services, and Digital Marketing.
- **Lead Priority Prediction:** Estimates whether an enquiry is High, Medium, or Low priority.
- **Confidence Scores:** Displays model prediction probabilities.
- **Recommended Actions:** Suggests suitable next steps based on the predicted service and priority.
- **Interactive Dashboard:** Provides a simple interface using Streamlit.

## Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

## Project Structure

```text
cynbit-ai-lead-intelligence/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── enquiries.csv
├── models/
│   ├── service_model.pkl
│   └── priority_model.pkl
├── reports/
└── screenshots/
```

## Dataset

The prototype uses 57 synthetic customer enquiries across five service categories.

The dataset includes:
- Customer enquiry message
- Service category
- Lead priority

No real customer information is used.

## Machine Learning Approach

Two separate text classification models are developed using Scikit-learn Pipelines.

1. TF-IDF converts enquiry text into numerical features.
2. Logistic Regression predicts the service category.
3. A separate Logistic Regression model estimates lead priority.

The dataset is split into training and testing subsets using a stratified 75:25 split.

## Initial Model Evaluation

| Model | Test Accuracy |
|---|---:|
| Service Classification | 86.67% |
| Priority Classification | 33.33% |

These are preliminary results from a small synthetic dataset and should not be interpreted as real-world performance.

Priority classification requires further improvement using a larger, representative dataset and additional evaluation.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shalini8310/cynbit-ai-lead-intelligence.git
cd cynbit-ai-lead-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\ven
