## Install

```
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
python train.py
```