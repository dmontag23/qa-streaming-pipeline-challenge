FROM python:3.7.9-slim-stretch
RUN mkdir -p /usr/src/app
COPY . /usr/src/app
WORKDIR /usr/src/app/api
RUN pip install -r ./requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]

