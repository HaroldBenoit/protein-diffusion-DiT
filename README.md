# Training from scratch a protein structure diffusion model on CATH dataset 


- The goal of this repo was to get familiar with protein & diffusion model training.

- The CATH dataset can be obtained from this [script](https://github.com/Mickdub/gvp/blob/master/data/getCATH.sh) or by downloading the files just below.

```bash
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json
```

- Architecture is a DiT (Diffusion Transformer) from [Peebles & Xie (2022)](https://arxiv.org/abs/2212.09748)
    - DiT code inspired by https://github.com/facebookresearch/DiT

 - Diffusion tips and tricks inspired by [Würstchen (ICLR 2024, oral)](https://arxiv.org/abs/2306.00637) (for architecture and loss) and [Kadkhodaie et al.](https://arxiv.org/abs/2310.02557) (for choosing model size, Appendix B.4)
    - Diffusion code inspired by https://github.com/pabloppp/pytorch-tools/blob/master/torchtools/utils/diffusion.py and https://github.com/dome272/Wuerstchen/tree/main


- Some resulting generations:

![generation1](images/protein_1.png)
![generation2](images/protein_2.png)
![generation3](images/protein_3.png)
![generation4](images/protein_4.png)
