#!/usr/bin/env bash

# variables
CONTAINER_RANK=0
BATCH_SIZE=32
LEARNING_RATE=0.01
NUM_GROUPS=10
CLIENTS_PER_GROUP=10
SAMPLER="gbp-cs"
NUM_SYNCS=50
NUM_ROUNDS=500
DATASET="femnist"
MODEL="cnn"
EVAL_EVERY=1
NUM_GPU_AVAILABLE=2
NUM_GPU_BEGIN=0

# container
IMAGE_NAME="fedgs"
CONTAINER_NAME="fedgs.${CONTAINER_RANK}"
HOST_NAME=${CONTAINER_NAME}
USE_GPU=$[${NUM_GPU_BEGIN}+${CONTAINER_RANK}%${NUM_GPU_AVAILABLE}]

# shellcheck disable=SC2034
sudo docker run -dit \
                --name ${CONTAINER_NAME} \
                -h ${HOST_NAME} \
                --gpus all \
                -e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
                -e MXNET_ENFORCE_DETERMINISM=1 \
                -v /etc/localtime:/etc/localtime \
                -v /home/lizh/fedgs:/root \
                ${IMAGE_NAME}

sudo docker exec -di ${CONTAINER_NAME} bash -c \
        "python main.py \
            -dataset ${DATASET} \
            -model ${MODEL} \
            --num-rounds ${NUM_ROUNDS} \
            --eval-every ${EVAL_EVERY} \
            --clients-per-group ${CLIENTS_PER_GROUP} \
            -sampler ${SAMPLER} \
            --batch-size ${BATCH_SIZE} \
            --num-syncs ${NUM_SYNCS} \
            --num-groups ${NUM_GROUPS} \
            -lr ${LEARNING_RATE} \
            --log-rank ${CONTAINER_RANK} \
            -ctx ${USE_GPU}"