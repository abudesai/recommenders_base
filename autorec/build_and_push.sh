#!/bin/bash

docker build -t abudesai/rec_base_autorec:latest .

docker push abudesai/rec_base_autorec:latest

docker rmi abudesai/rec_base_autorec:latest