export LOGURU_LEVEL=INFO
tether run -j 8 -d grayskull -i tok_bert_base_squad.yaml > /data/logs/log-tok-bert-base-squad.txt 2>&1
