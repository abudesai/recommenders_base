#!/bin/bash

docker build -t abudesai/rec_base_mf:latest .

docker push abudesai/rec_base_mf:latest

docker rmi abudesai/rec_base_mf:latest