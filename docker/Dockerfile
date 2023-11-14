FROM python:3.9.17
#copy the whole code directory
COPY . /digits/
WORKDIR /digits
RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh
# Define a volume to store the trained models
VOLUME /digits/models

# # Run flask app when the container runs
CMD ["./entrypoint.sh"]
# RUN export FLASK_APP=api/main

# RUN flask run --host=0.0.0.0
