# Inferring network structure with unobservable nodes from time series data

This repository contains the official implementation of: ***[Inferring network structure with unobservable nodes from time series data](https://aip.scitation.org/doi/10.1063/5.0076521)*** in Chaos 2022.

## Abstract
Network structures play important roles in social, technological, and biological systems. However, the observable nodes and connections in real cases are often incomplete or unavailable due to measurement errors, private protection issues, or other problems. Therefore, inferring the complete network structure is useful for understanding human interactions and complex dynamics. The existing studies have not fully solved the problem of the inferring network structure with partial information about connections or nodes. In this paper, we tackle the problem by utilizing time series data generated by network dynamics. We regard the network inference problem based on dynamical time series data as a problem of minimizing errors for predicting states of observable nodes and proposed a novel data-driven deep learning model called Gumbel-softmax Inference for Network (GIN) to solve the problem under incomplete information. The GIN framework includes three modules: a dynamics learner, a network generator, and an initial state generator to infer the unobservable parts of the network. We implement experiments on artificial and empirical social networks with discrete and continuous dynamics. The experiments show that our method can infer the unknown parts of the structure and the initial states of the observable nodes with up to 90% accuracy. The accuracy declines linearly with the increase of the fractions of unobservable nodes. Our framework may have wide applications where the network structure is hard to obtain and the time series data is rich.
<p align="center">
  <img src="./NEDMP_vis.png" width="450" title="hover text">
</p>

## Requirements
OS:
- Ubuntu

Python packages:
- troch==1.9.1

## Code

We provide bash scripts (at `./GIN/train_wscml.sh`) for trainning cml datasets used in our paper  (as well as hyper parameters) conducted in our paper,

Also we provide bash scripts (at `./GIN/voter_trainba_300.sh`) for trainning cml datasets used in our paper  (as well as hyper parameters) conducted in our paper.


## Citation

```
@article{chen2022inferring,
  title={Inferring network structure with unobservable nodes from time series data},
  author={Chen, Mengyuan and Zhang, Yan and Zhang, Zhang and Du, Lun and Wang, Shuo and Zhang, Jiang},
  journal={Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume={32},
  number={1},
  pages={013126},
  year={2022},
  publisher={AIP Publishing LLC}
}
```

# seed graph matching
seedgraphmatching(sgm.py) python  seed graph matching python reproduce 


reference: 
- https://github.com/youngser/VN  
- https://www.sciencedirect.com/science/article/pii/S0031320318303431

