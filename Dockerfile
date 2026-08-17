FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY ev_battery_failure_model.keras .
COPY ev_battery_scaler.pkl .

EXPOSE 8080

CMD ["python", "app.py"]
