[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/threshold-matters-in-wsss-manipulating-the/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=threshold-matters-in-wsss-manipulating-the)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/threshold-matters-in-wsss-manipulating-the/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=threshold-matters-in-wsss-manipulating-the)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/threshold-matters-in-wsss-manipulating-the/weakly-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-4?p=threshold-matters-in-wsss-manipulating-the) 

## Threshold Matters in WSSS: Manipulating the Activation for the Robust and Accurate Segmentation Model Against Thresholds (CVPR 2022)

<!-- ![](https://github.com/gaviotas/AMN/blob/main/figure/AMN.png?raw=true){: .center} -->
<p align="center">
<img src="https://github.com/gaviotas/AMN/blob/main/figure/AMN.png?raw=true">
</p>

__Official pytorch implementation of "Threshold Matters in WSSS: Manipulating the Activation for the Robust and Accurate Segmentation Model Against Thresholds"__

> [__Threshold Matters in WSSS: Manipulating the Activation for the Robust and Accurate Segmentation Model Against Thresholds__ ](https://arxiv.org/abs/2203.16045)<br>
> Minhyun Lee<sup>* </sup>, Dongseob Kim<sup>* </sup>, Hyunjung Shim <br>
> School of Integrated Technology, Yonsei University <br>
> <sub>* </sub> indicates an equal contribution. <br>
>
> __Abstract__ _Weakly-supervised semantic segmentation (WSSS) has recently gained much attention for its promise to train segmentation models only with image-level labels. Existing WSSS methods commonly argue that the sparse coverage of CAM incurs the performance bottleneck of WSSS. This paper provides analytical and empirical evidence that the actual bottleneck may not be sparse coverage but a global thresholding scheme applied after CAM. Then, we show that this issue can be mitigated by satisfying two conditions; 1) reducing the imbalance in the foreground activation and 2) increasing the gap between the foreground and the background activation. Based on these findings, we propose a novel activation manipulation network with a per-pixel classification loss and a label conditioning module. Per-pixel classification naturally induces two-level activation in activation maps, which can penalize the most discriminative parts, promote the less discriminative parts, and deactivate the background regions. Label conditioning imposes that the output label of pseudo-masks should be any of true image-level labels; it penalizes the wrong activation assigned to non-target classes. Based on extensive analysis and evaluations, we demonstrate that each component helps produce accurate pseudo-masks, achieving the robustness against the choice of the global threshold._


## Updates

31 Mar, 2022: Initial upload


## Requirement 

- This code is tested on Ubuntu 18.04, with Python 3.6, PyTorch 1.7.1, and CUDA 11.1.

### Dataset & pretrained checkpoint

- Download dataset, pretrained checkpoints, and refined seeds for AMN
  - Example directory hierarchy
  ```
  AMN
  |--- sess
  |    |--- res50_amn.pth.pth
  |    |--- res50_irn.pth
  |--- result
  |    |--- ir_label
  |    | ...
  | ...
  ```

- **Dataset**

  - [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 

- **Pretrained checkpoint**

  - [AMN](https://drive.google.com/file/d/1aGrXnjA2M33acP0BWlCNFCrkZ-8h84sw/view?usp=sharing)
  - [Boundary refinement network (IRN) w/ AMN](https://drive.google.com/file/d/1UqzFPQVugX1SgnO9W5FmdFdlo0XBwPlq/view?usp=sharing)

- **Refined seed for AMN (CAM + CRF)**

  - [Refined seed (ir_label)](https://drive.google.com/file/d/1N8lQsnqaKPOSVtuI-MCDeQxp-d9wHx5F/view?usp=sharing)
 

## Execution

### Pseudo-mask generation w/ AMN

- Execute the bash file.
    ```bash
    # Please see these files for the detail of execution.
    bash script/generate_pseudo_mask.sh
    ```

### Segmentation network
Fort the segmentation network, we experimented with [DeepLab-V2](https://github.com/kazuto1011/deeplab-pytorch) and followed the default training settings of [AdvCAM](https://github.com/jbeomlee93/AdvCAM)


## Acknowledgement
This code is highly borrowed from [IRN](https://github.com/jiwoon-ahn/irn). Thanks to Jiwoon, Ahn.


## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{Lee2022AMN,
    author    = {Lee, Minhyun, Kim, Dongseob, and Shim, Hyunjung},
    title     = {Threshold Matters in WSSS: Manipulating the Activation for the Robust and Accurate Segmentation Model Against Thresholds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2022}
}
```
