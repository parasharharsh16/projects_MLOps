FROM python:3.9.17
#copy the whole code directory
COPY . /digits/
WORKDIR /digits
RUN pip install -r requirements.txt
CMD ["pytest"]