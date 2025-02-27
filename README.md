## Install 
### Option-1. Pull from docker hub (Recommended)
- docker hub: https://hub.docker.com/r/twosubplace/krx_lm_train/tags
```sh
docker pull twosubplace/krx_lm_train:0.3.0
```

### Option-2. Build docker image in local
```sh
docker build --tag krx_lm_train:0.1.0 .
```

## Train

- **Run with docker**
```
docker run -it --entrypoint bash twosubplace/krx_lm_train:0.3.0
```

- **SFT train**
```
python krx_lm_train/sft_train.py --config configs/sft_config.yaml
```

- **CPO train**
```
python krx_lm_train/cpo_train.py --config configs/cpo_config.yaml
```


## Docker attach

```
docker ps -a

CONTAINER ID   IMAGE                            COMMAND   CREATED      STATUS          PORTS     NAMES
17e399a8e7ca   twosubplace/krx_lm_train:0.3.0   "bash"    2 days ago   Up 31 minutes             funny_kepler
```

- attach to container
```
docker attach 17e399a8e7ca
```