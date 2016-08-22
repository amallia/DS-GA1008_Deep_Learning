# ds-ga-1008-a2

NYU course 2016 Spring 2016 Assignment 2

stl-10 image classification with CNN, K-Means pretraining and data augmentation. This set of code is modified from Sergey Zagoruyko's cifar.torch tutorial: [cifar.torch](https://github.com/szagoruyko/cifar.torch/blob/master/README.md).

Data processing:

```bash
th -i provider.lua
```

```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```

Training:
```bash
th train.lua --model vgg_bn_drop -s logs/vgg
```




