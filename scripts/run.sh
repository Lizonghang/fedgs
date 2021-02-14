#!/usr/bin/env bash

# variables
CONTAINER_RANK=10
BATCH_SIZE=5
LEARNING_RATE=0.01
NUM_EPOCHS=3
NUM_ROUNDS=500
CLIENTS_PER_ROUND=100
DATASET="femnist"
MODEL="cnn"
EVAL_EVERY=20
NUM_GPU_AVAILABLE=2
NUM_GPU_BEGIN=0

# container
IMAGE_NAME="leaf-mx:mxnet1.4.1mkl-cu101-py3.7"
CONTAINER_NAME="leaf-mx.${CONTAINER_RANK}"
HOST_NAME=${CONTAINER_NAME}
USE_GPU=$[${NUM_GPU_BEGIN}+${CONTAINER_RANK}%${NUM_GPU_AVAILABLE}]

# shellcheck disable=SC2034
sudo docker run -dit \
                --name ${CONTAINER_NAME} \
                -h ${HOST_NAME} \
                --gpus all \
                -e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
                -v /etc/localtime:/etc/localtime \
                -v /home/lizh/leaf-mx:/root \
                ${IMAGE_NAME}

sudo docker exec -di ${CONTAINER_NAME} bash -c \
        "python main.py \
            -dataset ${DATASET} \
            -model ${MODEL} \
            --num-rounds ${NUM_ROUNDS} \
            --eval-every ${EVAL_EVERY} \
            --clients-per-round ${CLIENTS_PER_ROUND} \
            --batch-size ${BATCH_SIZE} \
            --num-epochs ${NUM_EPOCHS} \
            -lr ${LEARNING_RATE} \
            --log-rank ${CONTAINER_RANK} \
            -ctx ${USE_GPU} \
            --count-ops"