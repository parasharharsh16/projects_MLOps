FROM python:3.9.17
#copy the whole code directory
COPY . /digits/
WORKDIR /digits
RUN pip install -r requirements.txt

RUN chmod +x docker_run.sh
# Define a volume to store the trained models
VOLUME /digits/models

# Train the models when the container runs
CMD ["./docker_run.sh"]