#!/usr/bin/env bash
MY_BUCKET=gs://haifeng-tf-example
gsutil cp -r ${PWD}/tf_record_data $MY_BUCKET/