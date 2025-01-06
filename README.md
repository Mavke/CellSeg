# CellSegmentation

# About this project
Cell segmentation is a crucial task for biomedical image analysis. With the introduction of the Transformer architecture for computer vision tasks, numerous novel methods have been introduced. The majority of segmentation models are based on a combination of a Transformer in the encoder and a convolutional neural network, in the decoder, or alternatively, a convolutional neural network with a Transformer in the bottleneck. In this work, we propose a model for instance segmentation and simultaneous classification, which is based on the H2Former and CellViT models. The model encoder is similar to the H2Former encoder and includes a ResNet-18 and a multi-scale channel attention (MSCA) module, which is used to extract local features from different scales. These local features are passed as input to a Swin-Transformer, which enables self-attention and global context to be captured. We divide the Swin-Transformer and the ResNet with the MSCA module into two interconnected encoder paths. This architecture enables the use of a pretrained Swin-Transformer.
The proposed model is evaluated on two different datasets, the PanNuke dataset consisting of H&E stained histopathological images and a private dataset containing bacteria images.
The evaluation shows that the proposed model has a superior performance in detecting cell instances for the PanNuke dataset. In terms of the binary panoptic quality, the method achieves a score of 0.664, which is competitive with state-of-the-art methods. Contrary, the multiclass pantopic quality is inferior to the model for comparison.
The results demonstrate that the model architecture is suitable for instance segmentation and that it achieves competitive results compared to state-of-the-art methods. However, the proposed model is not optimal for simultaneous instance segmentation and classification.

# Model architecture
![fixed_model_](https://github.com/user-attachments/assets/5094875b-13d0-4eb7-92d4-e7114aee3e83)

The encoder consist of two branches, one branch is based on a Transformer and the second branch consists of a ResNet-34 combined with a multiscale channel attention module.
The feature maps are decoded by three seperate decoder branches
1. Decoding into a binary segmentation map
2. Decoding into horizontal and vertical distances
3. Decoding into a classification

For each decoder a loss function consiting of weighted losses is utilized:

$$L_{NB} = \lambda_{NB_{DICE}} L_{DICE} + \lambda_{NB_{FT}} L_{FT}$$
$$L_{ND} = \lambda_{ND_{MSE}} L_{MSE} + \lambda_{ND_{MSGE}} L_{MSGE}$$
$$L_{NC} = \lambda_{NC_{FT}} L_{FT} + \lambda_{NC_{DICE}} L_{DICE} + \lambda_{NC_{BCE}} L_{BCE}$$
with $L_{NB}$ the loss of the binary segmentation branch, $L_{ND}$ the loss for the vertical and horizontal distances, and $L_{NC}$ the loss for the classification branch. These loss functions are unified into one loss:
$$L = L_{NB} + L_{ND} + L_{NC}$$

# Results

The detection quality is measured as $F_{1i} = \frac{2 \cdot TP_i}{2 \cdot TP_i + FP_i + FN_i}$, $P_i = \frac{TP_i}{TP_i + FP_i}$ and $R_i = \frac{TP_i}{TP_i + FN_i}$ with:
- $TP_{i}$: correct identified instances
- $FP_{i}$: predicted instances without corresponding ground truth instance
- $FN_{i}$: unmatched ground truth instances


<p align="center">
  <img src="https://github.com/user-attachments/assets/eee0538c-67bd-4805-9125-f2c7e29cdc0a">
</p>

The detection quality for the PanNukedata set. The letters T,B,L,H indicate the size of the used Transformer. The results for the models indicated by * are taken from Hörst et al. [1].
The proposed model outperforms the reference models in terms of the detection of cell instances.

The panoptic quality is defined as follows:

$$PQ = \underbrace{\frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}}_{\substack{\text{segmentation quality}}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP|+\frac{1}{2}|FN|}}_{\substack{\text{detection quality}}}$$


<p align="center">
  <img src="https://github.com/user-attachments/assets/1b486a51-aefe-4c86-880c-5de9184eeee5">
</p>
As can be seen, the proposed model achieves comparable results in terms of bPQ. However, if the classification performance leading to mPQ is also taken into account, a significant difference to the state-of-the-art model
CellVit can be observed.
These results indicate the potential of the proposed model in terms of instance cell segmentation. However, it is also clear that the proposed model does not have good classification performance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/30cc6ac1-3c1c-4e31-9c81-efb34d687b30">
</p>
Visual results for the H2Former, UNet and the proposed model. The ground truth is shown in the first column, the cell classes are indicated by contour colouring.

In addition, an ablation study showed that the use of an MSCA module did not result in a significant increase in performance. However, the performance is still better than without this module.
This study also shows that there is no advantage using a pre-trained transformer. Instead, the use of a randomly trained transformer leads to an increase in performance compared to the model with a pre-trained transformer.



- [1] *Hörst et al. (2024)*. **CellViT: Vision Transformers for precise cell segmentation and classification** In: Medical Image Analysis vol. 94.
https://doi.org/10.1016/j.media.2024.103143
