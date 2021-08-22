export LOGURU_LEVEL=DEBUG
tether run -j 8 -d grayskull -i bert-large-cased-finetuned-squad.yaml > /data/inf-log-dump-bert.txt 2>&1
