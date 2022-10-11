FROM python:3.9-slim as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

# create user with a home directory
ARG NB_USER=experimenter
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN apt-get update && apt-get install -y gcc libffi-dev g++

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.2.0

# create user with a home directory
ARG NB_USER=experimenter
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER}

USER ${USER}

RUN python -m venv .venv
ENV PATH="$HOME/.venv/bin:$PATH"
RUN pip install --verbose "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false --local \
    & poetry install --only main --no-root --no-interaction

FROM base as final

# create user with a home directory
ARG NB_USER=experimenter
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER}

USER ${USER}

COPY --from=builder $HOME/.venv $HOME/.venv

ENV PATH="$HOME/.venv/bin:$PATH"

CMD ["bash"]