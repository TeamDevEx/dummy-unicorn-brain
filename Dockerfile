FROM python:3.11.3

RUN mkdir /usr/app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/app
WORKDIR /usr/app
EXPOSE 8080

# Run the command to start uWSGI
CMD ["uvicorn", "--app-dir", "/usr/app/src/api", "main:app", "--host", "0.0.0.0", "--port", "8080"]
