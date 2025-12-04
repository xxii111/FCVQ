# # #!/bin/bash

EMBEDDING_DIM=64
NUM_EMBEDDINGS=16
NUM_CHUNKS=1
BATCH_SIZE=32
LR=1e-5
SEED=3407
EPOCH=1
LMBDA=1.0
PRETRAINED=False
CHECKPOINT="./ckpt_cls/"
FREQUENCY_PATH="${CHECKPOINT}epoch_${EPOCH}num_${NUM_EMBEDDINGS}chunk_${NUM_CHUNKS}.pt"
VQ_PATH="${CHECKPOINT}epoch_${EPOCH}num_${NUM_EMBEDDINGS}chunk_${NUM_CHUNKS}.pth.tar"
PRETRAINED_PATH="./ckpt_cls/epoch_${EPOCH}num_${NUM_EMBEDDINGS}chunk_${NUM_CHUNKS}.pth.tar"

echo "prertrained path: ${PRETRAINED_PATH}"
echo "Frequency path: ${FREQUENCY_PATH}"
echo "VQ path: ${VQ_PATH}"
echo "VQ codebook size: ${NUM_EMBEDDINGS}"
echo "VQ dim: ${EMBEDDING_DIM}"
echo "VQ chunk: ${NUM_CHUNKS}"
echo "pretrained: ${PRETRAINED}"
echo "r+lmbda*d: ${LMBDA}"



python train_cls.py \
    --embedding_dim "$EMBEDDING_DIM" \
    --num_embeddings "$NUM_EMBEDDINGS" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --num_chunks "$NUM_CHUNKS" \
    --lmbda "$LMBDA" \
    --pretrained_path "$PRETRAINED_PATH" \
    --pretrained "$PRETRAINED" \
    -e "$EPOCH" | tee /output/log_train_multi_vq.txt


python test_bits_cls.py \
    --embedding_dim "$EMBEDDING_DIM" \
    --num_embeddings "$NUM_EMBEDDINGS" \
    --vq_path "$VQ_PATH" \
    --num_chunks "$NUM_CHUNKS" \
    --lmbda "$LMBDA" \
    # --frequency_path "$FREQUENCY_PATH" | tee /output/log_test_bits.txt