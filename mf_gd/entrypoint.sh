#!/bin/sh


files=$(ls /app/*)
echo "Files in Docker mount location:  $files"

echo "test" > /app/algorithm/ml_vol/input/train/test.txt

