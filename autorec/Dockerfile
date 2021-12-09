FROM python:3.8.0-slim as builder


COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 

COPY . .

WORKDIR /app

# RUN chmod +x entrypoint.sh

# CMD bash entrypoint.sh 
CMD python /app/serve.py
