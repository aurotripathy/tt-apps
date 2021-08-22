export LOGURU_LEVEL=INFO
# tether run -d grayskull --run-count=1000 --queue=50 --measure-time -i hello-tt.yaml
tether run -d grayskull --run-count=10 --queue=5 --measure-time -i hello-tt.yaml

