Model artifacts for the AI Differential Diagnosis System.

The supplied malaria and COVID model/scaler/feature artifacts are retained from the original project.

Additional artifact:
  covid_label_encoders.pkl

This encoder was generated from the supplied COVID dataset to match the original training code's
LabelEncoder mapping (Female=0, Male=1). The original training code did not save this encoder,
which made raw gender strings impossible to transform reliably during inference. Future COVID
retraining through src/covid_model.py saves the encoder automatically.

Serialized scikit-learn/XGBoost artifacts should be loaded with compatible library versions.
