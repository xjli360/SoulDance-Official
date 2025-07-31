# SoulDance: Music-Aligned Holistic 3D Dance Generation

[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://arxiv.org/abs/2507.14915)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.14915)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://xjli360.github.io/SoulDance/)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](https://www.youtube.com/watch?v=ND0aJVdBBao)
[![Code](https://img.shields.io/badge/Github-Code-pink)](https://github.com/xjli360/SoulDance-Official)

![SoulDance cover image](./assets/teaser.png)

> **Abstract**: Well-coordinated, music-aligned holistic dance enhances emotional expressiveness and audience engagement. However, generating such dances remains challenging due to the scarcity of holistic 3D dance datasets, the difficulty of achieving cross-modal alignment between music and dance, and the complexity of modeling interdependent motion across the body, hands, and face.
> To address these challenges, we introduce **SoulDance**, a high-precision music-dance paired dataset captured via professional motion capture systems, featuring meticulously annotated holistic dance movements. Building on this dataset, we propose **SoulNet**, a framework designed to generate music-aligned, kinematically coordinated holistic dance sequences.

## News
- [2025-08] Source code is currently undergoing an internal open-source compliance review at ByteDance and will be open-sourced once the review is complete.  
- [2025-07] The **SoulDance Dataset** is now available for **academic use**.  
- [2025-06] SoulDance** has been **accepted to ICCV 2025**.


## Key Features

- **Hierarchical Residual Vector Quantization**: Models complex, fine-grained motion dependencies across body, hands, and face
- **Music-Aligned Generative Model**: Composes hierarchical motion units into expressive and coordinated holistic dance
- **Music-Motion Retrieval Module**: Pre-trained cross-modal model ensuring temporal synchronization and semantic coherence

## Requirements

### System Requirements
- **OS**: 64-bit Python 3.10
- **Framework**: PyTorch 2.0.0
- **Memory**: At least 24 GB RAM per GPU
- **GPU**: 1–6 high-end NVIDIA GPUs with at least 24 GB of GPU memory
- **CUDA**: NVIDIA drivers, CUDA 12.4 toolkit

### Dependencies

This repository depends on the following specialized libraries:

- [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index) - Distributed training acceleration
- [librosa](https://github.com/librosa/librosa) - Audio analysis
- [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib) - Music information retrieval
- [jukebox](https://github.com/openai/jukebox) - Music generation models

### Installation

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

**Note**: This project uses the [SMPL-X](https://smpl-x.is.tue.mpg.de/) human body model and the [FLAME](https://flame.is.tue.mpg.de/) for human face model.


## Dataset Access

> **This dataset is available only for the academic use.** Out of respect and protection for the original data providers, we have collected all the links to the raw data for users to download from the original data creators. Please show your appreciation and support for the work of the original data creators by liking and bookmarking their content if you use this data. Please adhere to the usage rules corresponding to this original data; any ethical or legal violations will be the responsibility of the user. 

**License Requirements**:
1. Sign the EULA form located at `assets/SoulDance-EULA-20250728.pdf`
2. Send the signed form to: [fangshukai@bytedance.com](mailto:fangshukai@bytedance.com) or [beichuan@bytedance.com](mailto:beichuan@bytedance.com)
3. Upon approval, you will receive the download link

**Dataset Setup**:
- Download the SoulDance dataset and place it in the `SoulDance_data/` folder
- Organize files as:
  - Motion data: `data/souldance/motion/slice*.npz`
  - Music data: `data/souldance/music/slice*.mp4`

## Data Preparation


## Training Pipeline


## Evaluation


## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{li2025souldance,
    title={Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling}, 
    author={Xiaojie Li and Ronghui Li and Shukai Fang and Shuzhao Xie and Xiaoyang Guo and Jiaqing Zhou and Junkun Peng and Zhi Wang},
    year={2025},
    eprint={2507.14915},
    archivePrefix={arXiv},
    primaryClass={cs.MM},
    url={https://arxiv.org/abs/2507.14915}
}
```

## Acknowledgements

We thank the open-source community for their foundational contributions. Our work builds upon [EDGE](https://github.com/Stanford-TML/EDGE), [HumanTOMATO](https://github.com/IDEA-Research/HumanTOMATO), [EMAGE](https://github.com/PantoMatrix/PantoMatrix/blob/main/train_emage_audio.py), and [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for data processing; [MoMask](https://github.com/EricGuo5513/momask-codes), [TMR](https://github.com/Mathux/TMR), and [FineDance](https://github.com/li-ronghui/FineDance) for generative frameworks. Please cite these works if you use this codebase.

---

<div align="center">
  <b>🕺 Generate expressive holistic dances with SoulDance! 💃</b>
</div>
