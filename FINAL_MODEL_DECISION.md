# Final Model Decision

## Selected model
**CatBoostRegressor predicting rate-per-mile (RPM).**

The final prediction is:

`predicted_rate = predicted_RPM × distance`

## Validation
- Train period: January–September 2025
- Holdout: October 2025
- MAE: $102.79
- RMSE: $646.89
- R²: 0.8209
- Selected iterations: 170

## Why this model
The data contains mixed categorical and numerical variables, including pickup, delivery, equipment and route. CatBoost handles categorical features directly and the RPM target normalizes the strong dependence of total rate on distance.

## Final fit
After selecting the iteration count using the chronological October holdout, the model was refit on all labeled development rows and used to produce the 12,000 validation predictions and 31 fixed December predictions.
