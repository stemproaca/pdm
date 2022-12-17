
#FROM civisanalytics/datascience-python
FROM python:3.10

RUN pip install joblib lightgbm pandas scikit-learn seaborn statsmodels

# WORKDIR /opt/app

RUN for i in data model scripts; do  mkdir "/home/$i"; done

WORKDIR /home

ENV MODEL_DIR=/home/model
ENV METADATA_FILE=metadata.json
ENV MODEL_FILE=rul_lgbm.joblib


COPY requirements.txt requirements.txt
COPY train.py ./scripts/train.py
COPY predict.py ./scripts/predict.py

#RUN pip install -r requirements.txt

COPY train.csv ./data/train.csv
COPY test.csv ./data/test.csv


#RUN python3 train.py
