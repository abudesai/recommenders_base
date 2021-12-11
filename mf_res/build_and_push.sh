#!/bin/bash

docker build -t abudesai/rec_base_mf_res:latest .

docker push abudesai/rec_base_mf_res:latest

docker rmi abudesai/rec_base_mf_res:latest