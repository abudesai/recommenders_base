#!/bin/bash

docker build -t abudesai/rec_base_mf:v1.0.0 .

docker push abudesai/rec_base_mf:v1.0.0

docker rmi abudesai/rec_base_mf:v1.0.0