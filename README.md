ReVEAL
===============================

ReVEAL is a framework that combines Graph Neural Networks (GNNs) for architecture-level reverse engineering of optimized multipliers to assist in formal verification. The framework leverages GNN-based predictions to identify stage templates in optimized multiplier designs, which are then formally verified using SAT solving to check equivalence between the circuit under test and template library circuits. Additionally, ReVEAL employs offline computer algebra verification for the template library to ensure correctness.

Requirements
------------
* python 3.9
* pytorch 1.12 (CUDA 11.3)
* torch_geometric 2.1

Environment Setup
------------
To set up the environment, run the following commands:

```bash
cd env
bash setup_env.sh
bash activate_env.sh
```

Datasets and Pre-trained Models
------------
Pre-processed benchmarks, datasets, and reproducible pre-trained models are available at:

**https://huggingface.co/cc1997cc/ReVEAL**

If you want to play with and test the models as well as perform combinational equivalence checking (CEC), please fully download the contents from the above link first.

Running Experiments
------------
### Model Prediction

For testing the prediction accuracy of multiplier stage PPA and stage FSA using trained models:

```bash
cd REVEAL
bash run_total_test.sh
```

For retraining the models from scratch:

```bash
cd REVEAL
bash run_total_train.sh
```

### Formal Verification

To perform formal verification of optimized multipliers using predicted stage template types:

Navigate to the experiment directory and run the verification:

```bash
cd artifact/ReVEAL/exp1_reveal
```

The framework will utilize the model-predicted stage template types to conduct formal verification of the optimized multipliers.

Citation
------------
If you use ReVEAL in your research, please cite our work published in TACAS'26.

```
@inproceedings{chen2026reveal,
  title={ReVEAL: GNN-Guided Reverse Engineering for Formal Verification of Optimized Multipliers},
  author={Chen Chen and Daniela Kaufmann and Chenhui Deng and Zhan Song and Hongce Zhang and Cunxi Yu},
  booktitle={TACAS},
  year={2026},
}
```
