FROM civisanalytics/datascience-python


COPY requirements.txt /opt/weixiuai/requirements.txt

WORKDIR /home/weixiuai
RUN pip install --upgrade pip
RUN for i in data docs models model_results notebooks raw references reports src; do  mkdir "/home/weixiuai/$i"; done
RUN for i in featurizing modeling preparing raw serving visualizing; do  mkdir "/home/weixiuai/src/$i"; done


ENV MODEL_DIR=/home/weixiuai/model_results
ENV SCRIPT_DIR=/home/weixiuai/src
ENV RAW_DATAFOLDER=/home/weixiuai/raw
ENV REPORT_FOLDER=/home/weixiuai/reports

ENV MODEL_FILE_FIRST=rul_lstm_n_first.h5
ENV MODEL_FILE_SECOND=rul_lstm_n_second.h5
ENV MODEL_RESULT_FILE_FIRST=rul_lstm_n_first.csv
ENV MODEL_RESULT_FILE_SECOND=rul_lstm_n_second.csv
ENV FEATURE_FILE = FeatureCSV.csv


RUN pip install -r requirements.txt
