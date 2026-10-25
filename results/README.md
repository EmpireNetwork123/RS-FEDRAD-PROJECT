# Results

| Folder | Content |
|---|---|
| `tables/` | results reported in the RS-FEDRAD article (CSV, values in %) |
| `figures/` | figures of the RS-FEDRAD article (Figs. 1 and 3 to 6), as published |
| `logs/federated/` | original per-client logs of the FedAvg run (10 clients, 10 rounds, 10 local epochs) |
| `explainability/` | LIME and SHAP outputs of `python -m rsfedrad.explain` |

## Client logs

`client_XX_metrics.csv` holds one row per round. *Accuracy* and *Loss* are the client's training values at the end of the round's last local epoch; *Precision*, *Recall* and *F1-score* are computed on the client's held-out 20% split. `training_history_client_XX.csv` lists every local epoch (10 rounds x 10 epochs). The round-wise results in the article are the mean over the ten clients and are rebuilt with `python -m rsfedrad.aggregate_logs`.

## Explainability outputs

| File | Content |
|---|---|
| `summary.json` | number of explained samples, misclassifications, SHAP completeness check, group shares |
| `test_predictions.csv` | prediction of the global model for every held-out sequence |
| `lime_explanations.csv` | ten strongest LIME weights for each of the eight explained cases |
| `shap_values.npy`, `shap_samples.csv` | SHAP values (400 x 530, logit scale) and the samples they belong to |
| `shap_location_importance.csv` | mean absolute SHAP value per sequence position |
| `shap_group_contribution.csv` | share of the attribution per feature group |
| `shap_value_attributions.csv` | mean SHAP value of every API call, DLL, mutex and technique seen at least 20 times |
| `shap_ttp_techniques.csv` | mean SHAP value per ATT&CK technique |
| `fig_lime_local_explanations.*` | LIME explanations of two correct and two incorrect decisions |
| `fig_shap_location_based.*` | location-based global explanation (twenty most influential positions) |
| `fig_shap_feature_based.*` | feature-based global explanation of each class |

Figures are stored as PNG (300 dpi) and PDF. They are redrawn from the saved explanations with `python -m rsfedrad.explain --figures-only`.
