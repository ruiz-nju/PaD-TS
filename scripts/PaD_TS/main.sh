# NAME=stock
# # python run.py -d $NAME > results/$NAME.txt
# python run.py -d $NAME

DATA=./data
MODEL=PaD_TS
DATASET=$1 # stock
CFG=$2 # "default"
WINDOW=$3 # 24
SEED=$4  # 1

# sh scripts/main.sh fMRI default 24 1
# sh scripts/main.sh stock default 24 1
# sh scripts/main.sh sine default 24 1
DIR=output/${DATASET}/${MODEL}/${CFG}/seed_${SEED}
python main.py \
    --root ${DATA} \
    --output-dir ${DIR} \
    --model ${MODEL} \
    --config-file configs/models/${MODEL}/${CFG}.yaml \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --window ${WINDOW} \
    --seed ${SEED} \
    --period train
