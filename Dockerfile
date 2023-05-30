# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install pip requirements
COPY packages/ /packages
COPY requirements.txt .
COPY Internal_CBiTT_CA_.crt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
USER root
SHELL ["/bin/bash", "-c"]
RUN mv Internal_CBiTT_CA_.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# RUN export GIT_PYTHON_REFRESH=quiet

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["python", "run.py"]
# CMD ["python", "run.py"]
# CMD ["python", "manager_example.py"]

