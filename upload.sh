#!/usr/bin/env bash
MY_BUCKET=gs://haifeng-tf-example
gsutil cp -r ${PWD}/cifar-10-data $MY_BUCKET/