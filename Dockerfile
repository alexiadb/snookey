FROM tensorflow/tensorflow:latest
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8051
COPY . .
CMD ["streamlit", "run",  "app.py"]
