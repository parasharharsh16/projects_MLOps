FROM python:3.9.17
#copy the whole code directory
COPY . /digits/
WORKDIR /digits
RUN pip install -r requirements.txt
# # Run flask app when the container runs
EXPOSE 8000
CMD ["sh","./entrypoint.sh"]

#execute app in azure
#ENTRYPOINT [ "python","api/main.py" ]
# Define a volume to store the trained models
#VOLUME /digits/models
# RUN export FLASK_APP=api/main
# RUN flask run --host=0.0.0.0
