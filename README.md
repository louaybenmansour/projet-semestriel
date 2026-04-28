# Student Performance Predictor - Production ML Web App

End-to-end ML system with:
- multi-model training and automatic best-model selection,
- FastAPI inference backend (`POST /predict`),
- modern frontend (`HTML + CSS + JavaScript`).

## Structure

```text
project/
├── data/
│   └── dataset.csv (or student_performance_factors.csv)
├── models/
│   └── model.pkl
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── train.py
├── requirements.txt
└── src/
```

## API Contract

### Request: `POST /predict`
```json
{
  "StudyHours": 16,
  "SleepHours": 7,
  "Attendance": 82,
  "StressLevel": 35
}
```

### Response
```json
{
  "prediction": 85.4,
  "label": "Pass",
  "interpretation": "High Score",
  "threshold": 60
}
```

## Training Pipeline

`train.py` handles:
- data loading and cleaning,
- preprocessing (imputation, encoding, scaling),
- train/test split,
- model training and evaluation with:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost (if installed),
- metrics comparison table (MAE, RMSE, R2),
- best model selection using lowest RMSE,
- model save to `models/model.pkl`.

## Run Instructions

1. Install requirements
```bash
pip install -r requirements.txt
```

2. Train and save model artifacts
```bash
python train.py
```

3. Start backend
```bash
uvicorn backend.main:app --reload
```

4. Open the frontend
- Open `frontend/index.html` directly in your browser.
- The frontend calls `http://127.0.0.1:8000/predict`.

## Notes

- Backend loads `models/model.pkl` by default.
- `models/feature_importance.csv` is generated when the best model supports importances.
- Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
