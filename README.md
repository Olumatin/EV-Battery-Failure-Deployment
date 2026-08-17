# EV Battery Failure Prediction API

## Project Overview

This project deploys a neural network model developed from the EV Battery Failure dataset. The model predicts whether an electric vehicle battery is at risk of failure based on selected battery health, usage, charging, and aging features.

## Model Features

The model uses the following 10 features:

1. battery_health_percent
2. state_of_health
3. charge_efficiency
4. charging_quality_score
5. voltage_imbalance
6. odometer_km
7. cycle_count
8. cell_voltage_std
9. aging_score
10. capacity_loss_percent

## Model Performance

The model was evaluated on an unseen test dataset.

- Accuracy: 90.65%
- Precision: 52.25%
- Recall: 92.42%
- AUC: 97.28%

The high recall is particularly useful for an early-warning system because the model identifies most batteries that are at risk of failure.

## Deployment

The application is packaged using Docker and deployed as a web service on Railway.

## Files

- `app.py` - Flask prediction API
- `ev_battery_failure_model.keras` - trained neural network
- `ev_battery_scaler.pkl` - fitted feature scaler
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration

## Disclaimer

This model is a prototype for early battery failure warning and should undergo further validation with real-world operational data before unrestricted production deployment.
