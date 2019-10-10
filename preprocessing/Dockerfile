FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY preprocess.py ./

ENTRYPOINT ["python3", "-u", "preprocess.py"]

# docker run --rm -u $(id -u):$(id -g) -v /home/james/Downloads/DR-data/Data/DRIVE/test/images:/data python-n4:0.1 /data/01_test.tif -n4 