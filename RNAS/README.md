## CIFAR-10
To search the best CNN architecture for CIFAR-10, using RNAS-WS:
```
bash train_search_RNAS-WS_cifar10.sh
```

## Train on CIFAR-10
You can train the best architecture we discovered by RNAS-WS:
```
bash train_RNASNet_36_cifar10.sh
```

## Transferability to CINIC-10
To check the transferability, you can train the best architecture discovered by RNAS-WS on CINIC-10:
```
bash train_RNASNet_cinic10.sh
```
