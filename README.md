# Freight Rate Prediction Challenge

## Approach
- Target: `posted_rate`
- Development data: `data/train-test.csv`
- Chronological holdout: October 2025, because final validation is future November 2025.
- Model: CatBoost regression with MAE loss.
- Categorical features: pickup, delivery, equipment, route.
- Numerical/date features: distance, weight, coordinates, date components, log distance, weight-per-mile, distance-weight interaction, and cyclical calendar features.
- Missing weight is median-imputed with a missingness flag.
- `market_index` and `quote_signal` are not used because they are absent from the fixed December chart inputs.
- For December inputs, city coordinates are recovered from the development data.

## Run
```bash
pip install -r requirements.txt
python train.py
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

The scorer creates `scorer_results/candidate_december.png`.
