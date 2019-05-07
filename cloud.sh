#!/usr/bin/env bash
JOB_NAME=testnao10
BUCKET_NAME=haifeng-tf-example
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-west1
gcloud ai-platform jobs submit training $JOB_NAME\
    --job-dir $OUTPUT_PATH \
    --python-version 3.5 \
    --runtime-version 1.13 \
    --module-name cnn.train_search \
    --package-path NAO-WS/cnn/ \
    --region $REGION \
    --config config.yaml \
    -- \
    --base_dir gs://$BUCKET_NAME \
    --output_dir $OUTPUT_PATH \
    --dataset="mnist" \
    --child_use_aux_heads \
    --child_lr_cosine \
    --controller_attention \
    --controller_time_major \
    --controller_symmetry
