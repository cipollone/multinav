
FROM python:3.7-buster

	# python
ARG PYTHONUNBUFFERED=1
ARG PYTHONDONTWRITEBYTECODE=1
	# pip
ARG PIP_NO_CACHE_DIR=off
ARG PIP_DISABLE_PIP_VERSION_CHECK=on
ARG PIP_DEFAULT_TIMEOUT=100
	# poetry
ARG POETRY_NO_INTERACTION=1
ENV POETRY_VERSION=1.1.4
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false
	# install files
ARG PROJECT_TEMP_PATH="/opt/multinav"

# Apt-get installs
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
	curl libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx graphviz

# Install poetry 
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Bug with non-bultin typing package
RUN pip uninstall -y typing

# Install all
RUN mkdir ${PROJECT_TEMP_PATH}
WORKDIR ${PROJECT_TEMP_PATH}
COPY . ./
RUN poetry install --extras tf

# Entry point
ENTRYPOINT ["bash", "-l", "docker/tfrun-entry.sh"]
