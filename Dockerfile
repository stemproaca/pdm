FROM civisanalytics/datascience-python

COPY requirements.txt /opt/weixiuai/requirements.txt
WORKDIR /opt/weixiuai
RUN pip install -r requirements.txt