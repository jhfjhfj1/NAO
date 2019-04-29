#!/usr/bin/env bash
JOB_NAME=testnao3
BUCKET_NAME=haifeng-tf-example
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-central1
gcloud ai-platform jobs submit training $JOB_NAME\
    --job-dir $OUTPUT_PATH \
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
