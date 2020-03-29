FROM amd64/python:3.6-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1 \
	SHELL=/bin/bash

ARG USER='ds'
ARG GROUP='datascience'
ARG UID='1000'
ARG GID='1000'

RUN apt-get update && \
	apt-get install -y git && \
	apt-get install -y \
		libglib2.0-0 \
		libsm6 \
		libxrender1 \
		libxext6

RUN addgroup --gid ${GID} ${GROUP}
RUN adduser \
	--gecos '' \
    --disabled-password \
    --shell ${SHELL} \
    --ingroup ${GROUP} \
    --uid ${UID} \
    ${USER}

RUN pip install --upgrade pip && \
	pip install -U virtualenv

RUN pip install jupyter jupyterlab

WORKDIR /opt

RUN mkdir covidnet && \
	chown -R ${USER}.${GROUP} covidnet

RUN apt-get clean && \
	apt-get autoremove && \
	apt-get -o APT::Clean-Installed=true autoclean && \
	rm -Rf /var/cache/* && \
	rm -Rf /root/.cache

USER ${USER}

COPY ./opt .

RUN virtualenv covidnet && \
	covidnet/bin/pip install ipykernel && \
	covidnet/bin/pip install -r covidnet-requirements.txt

RUN jupyter kernelspec install covidnet-kernel --user

RUN rm -Rf /home/${USER}/.cache

WORKDIR /home/${USER}

ENTRYPOINT ["/opt/start-notebook.sh"]
