.PHONY: help test prep prep-1m prep-10m baselines baselines-1m baselines-10m cf cf-1m cf-10m cf-10m-resume neumf quick all

PYTHON ?= python3
SEED ?= 42
KS ?= 5,10,20
NUM_NEGATIVES ?= 99
LAMBDA_REG ?= 25
NUM_ITERS ?= 10
KNN_NEIGHBORS ?= 20,50
ALS_RANKS ?= 10,20
ALS_REGS ?= 0.1,1.0
ALS_MAX_ITER ?= 8
ALS_BIAS_LAMBDA ?= 25
SPARK_MASTER ?= local[4]
SPARK_DRIVER_MEMORY ?= 4g
SPARK_EXECUTOR_MEMORY ?= 4g
SPARK_DEFAULT_PARALLELISM ?= 4
SPARK_SHUFFLE_PARTITIONS ?= 4
ALS_USER_BLOCKS_1M ?= 10
ALS_ITEM_BLOCKS_1M ?= 10
ALS_USER_BLOCKS_10M ?= 40
ALS_ITEM_BLOCKS_10M ?= 40
NEUMF_EPOCHS ?= 6
NEUMF_BATCH_SIZE ?= 2048
NEUMF_LR ?= 0.001
NEUMF_WEIGHT_DECAY ?= 1e-6
NEUMF_NEG_RATIO ?= 1
NEUMF_EMBED_DIM ?= 32
NEUMF_MLP ?= 64,32,16
NEUMF_DROPOUT ?= 0.2
PYSPARK_MEM = PYSPARK_SUBMIT_ARGS='--driver-memory $(SPARK_DRIVER_MEMORY) --executor-memory $(SPARK_EXECUTOR_MEMORY) pyspark-shell'

help:
	@echo "Common targets:"
	@echo "  make quick          # Fast reproducibility path: 1m prep + baselines + CF + NeuMF"
	@echo "  make all            # Full rerun: 1m and 10m prep/baselines/CF + 1m NeuMF"
	@echo "  make test           # Unit tests"
	@echo "  make cf-10m-resume  # Resume long 10m CF run if interrupted"

test:
	$(PYTHON) -m unittest discover -s tests -q

prep: prep-1m prep-10m

prep-1m:
	$(PYTHON) scripts/prepare_data.py --dataset 1m --num-negatives $(NUM_NEGATIVES) --seed $(SEED)

prep-10m:
	$(PYTHON) scripts/prepare_data.py --dataset 10m --num-negatives $(NUM_NEGATIVES) --seed $(SEED)

baselines: baselines-1m baselines-10m

baselines-1m:
	$(PYTHON) scripts/baselines.py --dataset 1m --ks $(KS) --lambda-reg $(LAMBDA_REG) --num-iters $(NUM_ITERS) --num-negatives $(NUM_NEGATIVES) --seed $(SEED)

baselines-10m:
	$(PYTHON) scripts/baselines.py --dataset 10m --ks $(KS) --lambda-reg $(LAMBDA_REG) --num-iters $(NUM_ITERS) --num-negatives $(NUM_NEGATIVES) --seed $(SEED)

cf: cf-1m cf-10m

cf-1m:
	$(PYTHON) scripts/collab_filtering.py --dataset 1m --ks $(KS) --knn-neighbors $(KNN_NEIGHBORS) --als-ranks $(ALS_RANKS) --als-regs $(ALS_REGS) --als-max-iter $(ALS_MAX_ITER) --als-bias-lambda $(ALS_BIAS_LAMBDA) --als-user-blocks $(ALS_USER_BLOCKS_1M) --als-item-blocks $(ALS_ITEM_BLOCKS_1M) --spark-master '$(SPARK_MASTER)' --spark-driver-memory $(SPARK_DRIVER_MEMORY) --spark-executor-memory $(SPARK_EXECUTOR_MEMORY) --spark-default-parallelism $(SPARK_DEFAULT_PARALLELISM) --spark-shuffle-partitions $(SPARK_SHUFFLE_PARTITIONS) --num-negatives $(NUM_NEGATIVES) --seed $(SEED)

cf-10m:
	env $(PYSPARK_MEM) $(PYTHON) scripts/collab_filtering.py --dataset 10m --ks $(KS) --knn-neighbors $(KNN_NEIGHBORS) --als-ranks $(ALS_RANKS) --als-regs $(ALS_REGS) --als-max-iter $(ALS_MAX_ITER) --als-bias-lambda $(ALS_BIAS_LAMBDA) --als-user-blocks $(ALS_USER_BLOCKS_10M) --als-item-blocks $(ALS_ITEM_BLOCKS_10M) --spark-master '$(SPARK_MASTER)' --spark-driver-memory $(SPARK_DRIVER_MEMORY) --spark-executor-memory $(SPARK_EXECUTOR_MEMORY) --spark-default-parallelism $(SPARK_DEFAULT_PARALLELISM) --spark-shuffle-partitions $(SPARK_SHUFFLE_PARTITIONS) --num-negatives $(NUM_NEGATIVES) --seed $(SEED)

cf-10m-resume:
	env $(PYSPARK_MEM) $(PYTHON) scripts/collab_filtering.py --dataset 10m --ks $(KS) --knn-neighbors $(KNN_NEIGHBORS) --als-ranks $(ALS_RANKS) --als-regs $(ALS_REGS) --als-max-iter $(ALS_MAX_ITER) --als-bias-lambda $(ALS_BIAS_LAMBDA) --als-user-blocks $(ALS_USER_BLOCKS_10M) --als-item-blocks $(ALS_ITEM_BLOCKS_10M) --spark-master '$(SPARK_MASTER)' --spark-driver-memory $(SPARK_DRIVER_MEMORY) --spark-executor-memory $(SPARK_EXECUTOR_MEMORY) --spark-default-parallelism $(SPARK_DEFAULT_PARALLELISM) --spark-shuffle-partitions $(SPARK_SHUFFLE_PARTITIONS) --num-negatives $(NUM_NEGATIVES) --seed $(SEED) --resume

neumf:
	$(PYTHON) scripts/neumf.py --dataset 1m --epochs $(NEUMF_EPOCHS) --batch-size $(NEUMF_BATCH_SIZE) --lr $(NEUMF_LR) --weight-decay $(NEUMF_WEIGHT_DECAY) --neg-ratio $(NEUMF_NEG_RATIO) --embed-dim $(NEUMF_EMBED_DIM) --mlp $(NEUMF_MLP) --dropout $(NEUMF_DROPOUT) --seed $(SEED) --ks $(KS)

quick: prep-1m baselines-1m cf-1m neumf

all: prep baselines cf neumf
