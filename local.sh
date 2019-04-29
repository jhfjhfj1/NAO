#!/usr/bin/env bash
JOB_NAME=test-nao-1
BUCKET_NAME=haifeng-tf-example
OUTPUT_PATH=models
REGION=us-central1
gcloud ai-platform local train \
    --module-name cnn.train_search \
    --package-path NAO-WS/cnn/ \
    --job-dir $OUTPUT_PATH \
    -- \
    --base_dir ../ \
    --output_dir $OUTPUT_PATH \
    --dataset="cifar10" \
    --child_use_aux_heads \
    --child_lr_cosine \
    --controller_attention \
    --controller_time_major \
    --controller_symmetry
