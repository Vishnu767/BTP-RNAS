# Randomized Neural Architecture Search

# Abstract #
Neural network architectures are very critical in determining optimal model performance across various datasets. Traditional approaches to architecture design often involve manual exploration or heuristic methods, which can be time-consuming and resource-intensive. In this paper, we propose a randomized neural architecture search algorithm aimed at efficiently discovering high-performing architectures. Our approach involves embedding the unstructured space of neural architectures within a manifold, enabling constrained exploration of architectures with promising performance potential. By diligently sampling architectures from regions exhibiting high performance values within this constrained space, our algorithm focuses its search efforts on promising candidates.

# Requirements #
* Python 3.7.13+
* PyTorch 1.13+
* TorchVision 0.14+





## Results
### CIFAR-10

| Model | Error (%) | Params (M) |
|-------|-----------|------------|
| RNASNet-WS + Cutout | 2.78 | 2.5 |  
| NAONet-WS + Cutout | 2.93 | 2.5 |
| DARTS + Cutout | 2.83 | 4.6 |
| AmoebaNet-B + Cutout | 2.13 | 34.9 |

The architecture discovered by RNASNet-WS resulted in a competitive error rate of 2.78% on CIFAR-10, outperforming previous neural architecture search methods like ENAS, PNAS, DARTS, while being over 10x more parameter efficient than AmoebaNet-B.

### CINIC-10 

| Model | Top-1 Acc (%) | Params (M) |
|-------|----------------|------------|
| RNASNet-WS + Cutout  | 91.48 | 2.5 |
| NAONet-WS + Cutout | 90.83 | 2.5 |  
| NAT-M4 | 94.8 | 9.1 |

On the transfer task of CINIC-10, RNASNet-WS achieved 91.48% top-1 accuracy, outperforming NAONet-WS and being competitive with NAT models that were directly trained on CINIC-10.

### Penn Treebank

| Model | Test Perplexity | 
|-------|-----------------|
| RNASNet-WS | 57.5 |
| NAONet-WS | 56.6 |
| DARTS | 56.1 |

On the Penn Treebank language modeling task, the RNN architecture discovered by RNASNet-WS achieved a test perplexity of 57.5, surpassing various neural architecture search methods like ENAS and being on par with state-of-the-art results.

The code for PTB: (BTP-RNAS-RNN)[https://github.com/prathik8794/BTP-RNAS-RNN]

See the paper for more details on the experimental setup and full results.

# Acknowledgements #
We thank the authors of Neural Architecture Optimization (NAO) for the code base of the encoder, decoder, and performance predictor used in our work.
We also thank the Department of CSE, Indian Institute of Technology Tirupati for providing the GPU resources for this project.
