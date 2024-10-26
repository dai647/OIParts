# One-shot In-context Part Segmentation
<img src="assets/framework.png" width="100%">

# Visual Results
<img src="assets/visual.png" width="100%">

# Environment Setup
conda env create -f environment.yaml

# Run demo
1. python Save_features.py
2. python Get_Select_IDs.py
3. python demo_main.py

## Acknowledgement
Our code is based on the following open-source projects: [sd-dino](https://github.com/Junyi42/sd-dino), [VLPart](https://github.com/facebookresearch/VLPart), [FBS](https://github.com/poolio/bilateral_solver). we sincerely thanks to the developers of these resources!

# Citation
```bibtex
@inproceedings{dai2024oiparts,
  author = {Dai, Zhenqi and Liu, Ting and Zhang, Xingxing and Wei, Yunchao and Zhang, Yanning},
  title = {One-shot In-context Part Segmentation},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages = {10966–10975},
  year={2024}
}
```
