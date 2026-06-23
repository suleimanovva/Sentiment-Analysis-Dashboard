FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install pandas scikit-learn streamlit joblib
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]