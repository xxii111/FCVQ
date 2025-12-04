# # # # #!/bin/bash

EMBEDDING_DIM=64
NUM_EMBEDDINGS=16
NUM_CHUNKS=1
BATCH_SIZE=1
LR=1e-3
SEED=3407
EPOCH=1
LMBDA=1.0
PRETRAINED=False
FEATURE_SPLIT_SIZE=8

CHECKPOINT="./ckpt_seg/"
VQ_PATH="${CHECKPOINT}epoch_${EPOCH}num_${NUM_EMBEDDINGS}chunk_${NUM_CHUNKS}.pth.tar"
PRETRAINED_PATH="./ckpt_seg/epoch_${EPOCH}num_${NUM_EMBEDDINGS}chunk_${NUM_CHUNKS}.pth.tar"

echo "prertrained path: ${PRETRAINED_PATH}"
echo "VQ path: ${VQ_PATH}"
echo "VQ codebook size: ${NUM_EMBEDDINGS}"
echo "VQ dim: ${EMBEDDING_DIM}"
echo "VQ chunk: ${NUM_CHUNKS}"
echo "pretrained: ${PRETRAINED}"
echo "r+lmbda*d: ${LMBDA}"



python train_seg.py \
    --pretrained "$PRETRAINED"\
    --pretrained_path "$PRETRAINED_PATH"\
    --embedding_dim "$EMBEDDING_DIM" \
    --num_embeddings "$NUM_EMBEDDINGS" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --num_chunks "$NUM_CHUNKS" \
    --lmbda "$LMBDA" \
    --feature_split_size "$FEATURE_SPLIT_SIZE" \
    -e "$EPOCH" | tee /output/log_train_seg.txt


python test_bits_seg.py \
    --embedding_dim "$EMBEDDING_DIM" \
    --num_embeddings "$NUM_EMBEDDINGS" \
    --vq_path "$VQ_PATH" \
    --num_chunks "$NUM_CHUNKS" \
    --lmbda "$LMBDA" \
    # --frequency_path "$FREQUENCY_PATH" | tee /output/test_seg.txt