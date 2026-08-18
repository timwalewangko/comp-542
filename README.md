# Breast Cancer Prediction

HistGradientBoostingClassifier analysis and a Streamlit demo for predicting cancer vs. no cancer.

## Run the demo

1. Clone the repo and enter the project directory.

```bash
git clone https://github.com/timwalewangko/comp-542.git
cd comp-542
```

2. Create and activate a virtual environment.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install the requirements.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Start the Streamlit app.

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal, usually http://localhost:8501.

The form starts with values from the first row of the dataset. Change any inputs and click **Predict** to see the predicted class and cancer probability.

## Notebook

`breast_cancer.ipynb` is the main notebook. It contains data exploration, feature selection, HGBC training, hyperparameter tuning, testing, and experimental results. Its last cell saves the fitted pipeline used by the demo to `models/breast_cancer_hgbc.joblib`.

The other `Breast_Cancer_*.ipynb` notebooks contain the individual model experiments used for comparison.

The previous bird-species work is kept in `_old/`.

This project is for coursework and is not a medical diagnostic tool.
