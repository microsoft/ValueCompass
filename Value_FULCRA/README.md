# Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Values
<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
📄 <a href="https://aclanthology.org/2024.naacl-long.486.pdf" target="_blank" style="margin-right: 15px; margin-left: 10px">Paper</a>
</p>

## Overview
<p align="center"> <img src="figures/introduction.png" style="width: 95%;" id="title-icon"></p>
Leveraging basic values established in humanity and social science that are compatible with values across cultures, this paper introduces a novel value space spanned by multiple basic value dimensions and proposes BaseAlign, a corresponding value alignment paradigm. Applying the representative Schwartz’s Theory of Basic Values as an instantiation, we construct FULCRA, a dataset consisting of (LLM output, value vector) pairs. LLMs’ outputs are mapped into the K-dim value space beyond simple binary labels, by identifying their underlying priorities for these value dimensions. Extensive analysis and experiments on FULCRA: (1) reveal the essential relation between basic values and LLMs’ behaviors, (2) demonstrate that our paradigm with basic values not only covers existing risks but also anticipates the unidentified ones, and (3) manifest BaseAlign’s superiority in alignment performance with less data.

## The FULCRA Dataset


## Value Evaluator
- main.py

## Value Alignment
- main.py
- evaluation (script)

## Citation
If you find Value Fulcra and BaseAlign useful:
```
@article{yao2023value,
  title={Value fulcra: Mapping large language models to the multidimensional spectrum of basic human values},
  author={Yao, Jing and Yi, Xiaoyuan and Wang, Xiting and Gong, Yifan and Xie, Xing},
  journal={arXiv preprint arXiv:2311.10766},
  year={2023}
}
```