export LOGURU_LEVEL=INFO
tether run -j 8 -c --save-ttg-path bert-large-cased-finetuned-squad.ttg -d grayskull -i bert-large-cased-finetuned-squad.yaml > /data/ttg-build-log-bert-lcf.txt 2>&1


