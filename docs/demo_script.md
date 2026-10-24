# Demo script (about 3 minutes)

Record with any screen recorder (QuickTime, OBS, Loom). Covers every demo requirement in the rubric.

## 0. Setup shot (15 s)
- Terminal: `pytest tests/ -v` scrolled to the green summary line.
- MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`): the experiment table with all runs and the AUC / recall columns.
- Terminal: `python -m src.compare_runs` output showing the ranked table and the selected run.

## 1. Happy path (60 s)
Run `streamlit run src/app.py` and type:

> 58-year-old man with asymptomatic chest pain, resting BP 140, cholesterol 289, fasting blood sugar normal, normal resting ECG, max heart rate 125, angina on exercise, oldpeak 1.5, flat ST slope.

Point out:
- the sidebar checklist filling in (parsing worked),
- the probability and the factor chart (the real model ran),
- the explanation in plain language,
- the **"What the system parsed"** expander (raw LLM extraction).

Follow-up: *"What if his max heart rate were 165?"* and show that the estimate changes.

## 2. Edge case: incomplete input (40 s)
Click **New patient / reset**, then type:

> I'm a 45 year old woman with a blood pressure of 130/85. What's my risk?

Show that it lists the missing measurements instead of predicting. Then supply them:

> Atypical chest pain, cholesterol 210, fasting sugar 95, normal ECG, max HR 160, no exercise angina, oldpeak 0, upsloping ST.

It now predicts, combining both messages.

## 3. Edge cases: out of scope and ambiguous (30 s)
- *"What medication should I take for my blood pressure?"* → scope message.
- *"He's in his 50s with high cholesterol"* → asks for exact values.

## 4. Close (15 s)
One sentence on the architecture: the LLM handles language, deterministic code validates, and the trained model, chosen with MLflow, makes every prediction.
