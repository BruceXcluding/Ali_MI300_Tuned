#!/user/bin/bash
set -ex

KIT_PATH=$(dirname "$PWD")
TUNE_PATH=$KIT_PATH"/pytorch_afo_testkit/afo/tools/tuning/tune_from_rocblasbench.py"

for d in */; do
    cd $d
    for f in *.yaml; do
	echo "Tuning for $d/$f"
	python $TUNE_PATH $f --cuda_device 0 1 2 3
    done
    cd ..
done

python3 combine_yamls.py
