## Install 
### Option-1. pull from docker hub
```sh
docker pull twosubplace/krx_lm_train
```

### Option-2. build docker image in local
```sh
docker build --tag krx_lm_train:0.1.0 .
```

## Train

- **Step1**
```
docker run -it --entrypoint bash -v $(pwd):$(pwd) -w $(pwd) krx_lm_train:0.1.0
```

- **Step2**
```
cd krx_lm_train
python krx_lm_train/sft_train.py --config configs/sft_config.yaml
```