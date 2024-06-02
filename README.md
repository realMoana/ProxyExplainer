# Generating In-Distribution Proxy Graphs for Explaining Graph Neural Networks
This is a PyTorch implementation for ProxyExplainer. 


## Requirements

- `python==3.9`
- `torch==2.0.1`

## Datasets
Real-world datasets:
- [Mutag](https://arxiv.org/pdf/2011.04573.pdf)
- [Benzene](https://www.nature.com/articles/s41597-023-01974-x)
- [Alkane-Carbonyl](https://www.nature.com/articles/s41597-023-01974-x)
- [Fluoride-Carbonyl](https://www.nature.com/articles/s41597-023-01974-x)

Synthetic datasets:
- [BA-2motifs](https://arxiv.org/pdf/2011.04573.pdf)
- [BA-3motifs](https://arxiv.org/abs/2310.19321)

## Baselines
- [Grad](https://arxiv.org/pdf/1903.03894)
- [GNNExplainer](https://arxiv.org/pdf/1903.03894)
- [PGExplainer](https://arxiv.org/pdf/2011.04573)
- [ReFine](https://papers.nips.cc/paper_files/paper/2021/file/99bcfcd754a98ce89cb86f73acc04645-Paper.pdf)
- [MixupExplainer](https://arxiv.org/pdf/2307.07832)

## Usage
- By default, the experiment will use the pretrained models that are saved in `ExplanationEvaluation/models/pretrained/GNN/`.

## Citation  
If you find this resource helpful, please consider starting this repository and cite our research:

```bibtex
@inproceedings{chen2024proxy,
      title={Generating In-Distribution Proxy Graphs for Explaining Graph Neural Networks}, 
      author={Zhuomin Chen, Jiaxing Zhang, Jingchao Ni, Xiaoting Li, Yuchen Bian, Md Mezbahul Isam, Ananda Mondal, Hua Wei, Dongsheng Luo},
      year={2024},
      booktitle={Proceedings of the 41st International Conference on Machine Learning}
}
```

## Using robust fidelity as metric

If you want to use robust fidelity for evaluation, please refer to: 
https://github.com/AslanDing/Fidelity.

## Further reading

For the most comprehensive collection of graph explainability papers, please refer to:
https://github.com/flyingdoog/awesome-graph-explainability-papers.

## Acknowledgement

Our code are based on [[Re] Parameterized Explainer for Graph Neural Network](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks). Thanks to the original authors for open-sourcing their work.