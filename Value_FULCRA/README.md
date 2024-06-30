# Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Values
<!-- <p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
📄 <a href="https://aclanthology.org/2024.naacl-long.486.pdf" target="_blank" style="margin-right: 15px; margin-left: 10px">Paper</a>
</p> -->
Paper Link: https://aclanthology.org/2024.naacl-long.486.pdf

## Overview
<p align="center"> <img src="figures/introduction.png" style="width: 95%;" id="title-icon"></p>
Leveraging basic values established in humanity and social science that are compatible with values across cultures, this paper introduces a novel value space spanned by multiple basic value dimensions and proposes BaseAlign, a corresponding value alignment paradigm. Applying the representative Schwartz’s Theory of Basic Values as an instantiation, we construct FULCRA, a dataset consisting of (LLM output, value vector) pairs. LLMs’ outputs are mapped into the K-dim value space beyond simple binary labels, by identifying their underlying priorities for these value dimensions. Extensive analysis and experiments on FULCRA: (1) reveal the essential relation between basic values and LLMs’ behaviors, (2) demonstrate that our paradigm with basic values not only covers existing risks but also anticipates the unidentified ones, and (3) manifest BaseAlign’s superiority in alignment performance with less data.

## The FULCRA Dataset
This dataset contains 20k pairs of an LLM's output and the underlying values reflected by the response. Multiple domains involving values and multiple models for response generation are included, with statistics and examples illustrated in the Figure.
<p align="center"> <img src="figures/dataset.png" style="width: 95%;" id="title-icon"></p>

You can access the dataset in the directory ./data. More details about each split are introduced there.

## Value Evaluator
First, we train the core basic value evaluator, which can 1) measure LLMs' underlying values to reveal potential risks, and 2) perform as a reward model for subsequent alignment.
```
cd src/value_evaluator
bash scripts/run_classifier.sh or scripts/run_regression.sh
```

The trained ckpt would be saved as ./output/evaluator/

## Value Alignment
This step utilized BaseAlign for alignment. We provide several target values in main.py, you can design more kinds of values (Human-defined values, Cultural or national values) referring to the approched mentioned in our paper.
```
cd src/value_alignment
accelerate launch main.py
```
Evaluate the helpful and harmless scores of the aligned models.
```
bash eval_scripts/inference_model.sh  # infer the evaluation results
bash eval_scripts/eval_model.sh   # compute the evaluation scores
```


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