FROM python:3.11.7-alpine

WORKDIR /app

RUN pip install numpy pandas scikit-learn matplotlib
RUN pip install soundfile

COPY . /app

CMD [ "python", "main.py" ]

