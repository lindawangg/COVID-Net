#!/usr/bin/env bash

# Please notice that is running without anykind of authentication
jupyter lab --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
