FROM python:3.12.7

WORKDIR /app

COPY MLProject/requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "MLProject/modelling.py"]