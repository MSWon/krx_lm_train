FROM python:3.9.16 AS base

ENV PACKAGE_NAME=krx_lm_train
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VERSION=1.4.0
ENV POETRY_HOME="/home/jbk4860/apps/poetry"
ENV POETRY_NO_INTERACTION=1
ENV SETUP_PATH="/home/jbk4860/apps/${PACKAGE_NAME}"
ENV VENV_PATH="${SETUP_PATH}/.venv"
ENV PATH="${POETRY_HOME}/bin:${VENV_PATH}/bin:$PATH"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER root

RUN wget -qO - http://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
RUN echo "deb http://linux.mellanox.com/public/repo/mlnx_ofed/24.04-0.6.6.0/debian11.3/x86_64 ./" \
    | tee /etc/apt/sources.list.d/mlnx_ofed.list
RUN apt-get update \
    && apt-get install -y mlnx-ofed-all \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${SETUP_PATH}
WORKDIR ${SETUP_PATH}

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN python -m venv ${VENV_PATH}

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install --no-cache --no-root --without dev

COPY ${PACKAGE_NAME} ${PACKAGE_NAME}
COPY configs ./configs

# ------------------------
# development image for CI
# ------------------------
FROM base AS development

COPY tests ./tests
RUN poetry install --no-cache


# ------------------------
# production image for distribution
# ------------------------
FROM base AS production
