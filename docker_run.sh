#!/bin/bash

docker build -t digits:v1 -f docker/Dockerfile .
rm models/*
echo "Models before execution"
ls -lh models
echo "Run docker image"
docker run -v ./models:/digits/models digits:v1
echo "Models after execution"
ls -lh models