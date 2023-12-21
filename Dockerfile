FROM python:3.11-slim

COPY scripts/requirements.txt /root/src/requirements.txt

RUN chown -R root:root /root/

WORKDIR /root/src
RUN pip3 install -r requirements.txt

COPY src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]