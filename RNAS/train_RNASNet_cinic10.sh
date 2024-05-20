# Train the architecture discovered by NAO-WS, with channel size of 36, noted as NAONet-B-36
nvidia-smi
MODEL=RNASNet_36_cinic10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data
DATASET=cinic10
mkdir -p $OUTPUT_DIR

# 2nd Best arch from BTP OLD 2
fixed_arc="0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 2 0 0 3 1 3 1 0 1 0 1 0 1 0 1"

python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset=$DATASET \
  --arch="$fixed_arc" \
  --use_aux_head \
  --cutout_size=16 | tee -a $OUTPUT_DIR/train.log
