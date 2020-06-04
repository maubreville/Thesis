# Figures

These figures, used in my doctoral thesis, are all licensed under creative commons (CC BY 4.0) and are thus free to be used in other works (given proper referencing).


# Theoretical Background
## Machine Learning

![Figure 2.1: Pattern recognition pipeline, showcasing the difference between traditional machine learning and deep learning.](theory/PatternRecognition.svg)
Figure 2.1: Pattern recognition pipeline, showcasing the difference between traditional machine learning and deep learning.


![Figure 2.2: Main machine learning tasks on the example of a microscopy image patch](theory/PatternRecognitionTasks.svg)
Figure 2.2: Main machine learning tasks on the example of a microscopy image patch


## Deep Learning

![Figure 3.1: A single neuron](theory/neuron.svg)

Figure 3.1: A single neuron with inputs ![\mathbf{x}](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bx%7D), weights ![\mathbf{w}](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bw%7D), bias ![b_0](https://render.githubusercontent.com/render/math?math=b_0) and activation function ![h()](https://render.githubusercontent.com/render/math?math=h()).


![Figure 3.2: Activation functions and their derivatives commonly used within neural networks. Left panel shows activation functions and their output, right panel shows derivatives.](theory/activationfunctions_original.svg)
![Figure 3.2: Activation functions and their derivatives commonly used within neural networks. Left panel shows activation functions and their output, right panel shows derivatives.](theory/activationfunctions_derivative.svg)

Figure 3.2: Activation functions and their derivatives commonly used within neural networks. Left panel shows activation functions and their output, right panel shows derivatives.

![Figure 3.3: Visualization of a fully connected layer employing matrix operations](theory/neuron_matrix.svg)

Figure 3.3: Visualization of a fully connected layer employing matrix operations. Note that this can also be thought of as successive application of intermedia functions <b><i>h</i></b> and <b><i>v</i></b>.

![Figure 3.4: Gradient descent applied on a non-convex function. Two initializations (red and blue line) have been chosen, leading to different results](theory/gradientdescent)

Figure 3.4: Gradient descent example]{Gradient descent applied on a non-convex function. Two initializations (red and blue line) have been chosen, leading to different results

![Figure 3.5: Layered backpropagation](theory/LayeredBackprop.svg)

Figure 3.5: Layered backpropagation]{Layered backpropagation. Adapted from Maier et al.: A gentle introduction to deep learning in medical image processing (Zeitschrift fuer medizinische Physik)

![Loss functions commonly used in deep learning for classification and regression. Classification losses](theory/cla_losses.svg)
![Loss functions commonly used in deep learning for classification and regression. Regression losses](theory/reg_losses.svg)

Figure 3.6: Loss functions commonly used in deep learning for classification and regression.

![Figure 3.7: Segmentation loss functions and the definition of true positive, false positive and false negative.](theory/IOUsvg.svg)

Figure 3.7: Segmentation loss functions and the definition of true positive, false positive and false negative.

![Figure 3.8: 2D convolution on the example of a small image of a mitotic figure. The convolution operation is sliding a multiplicative kernel over the complete image.](theory/conv2d.svg)

Figure 3.8: 2D convolution on the example of a small image of a mitotic figure. The convolution operation is sliding a multiplicative kernel over the complete image.

![Figure 3.9: Inception module](theory/Inception.svg)

Figure 3.9: Inception module (redrawn from: Szegedy et al.: Going deeper with convolutions (CVPR 2015))

![Figure 3.10: Residual layer](theory/ResidualLayer.svg)

Figure 3.10: Residual layer (redrawn from: He et al.: Deep Residual Learning for Image Recognition (CVPR 2016))

![Figure 3.11: Comparing Yann LeCun's LeNet-5 architecture from 1989 to Krizhevsky's AlexNet architecture from 2012](theory/AlexNet.svg)

Figure 3.11: Comparing Yann LeCun's LeNet-5 architecture from 1989 to Krizhevsky's AlexNet architecture from 2012

![Figure 3.12: Evolution of R-CNN approaches](theory/RCNNoverview.svg)

Figure 3.12: Evolution of R-CNN approaches: R-CNN, Fast R-CNN and Faster R-CNN.

![Figure 3.13: Anchor boxes of two sizes and three aspect ratios, visualized on a mitotic figure (blue bounding box). Red anchor is only match (IOU$>0.5$).](theory/anchorbox.svg)

Figure 3.13: Anchor boxes of two sizes and three aspect ratios, visualized on a mitotic figure (blue bounding box). Red anchor is only match (IOU>0.5).

![Figure 3.14: Comparison of encoder-decoder structures in auto-encoder, FCN, U-Net and Feature Pyramid Network.](theory/Unet_FPN_AutoEnc.svg)

Figure 3.14: Comparison of encoder-decoder structures in auto-encoder, FCN, U-Net and Feature Pyramid Network.

![Figure 3.15: Basic structure of Generative Adversarial Networks](theory/GAN.svg)

Figure 3.15: Basic structure of Generative Adversarial Networks

![Figure 3.16: Basic structure of CycleGAN](theory/CycleGAN.svg)

Figure 3.16: Cycle consistency loss for target domain images ![L_{cyc}(X_T, G_S(G_T(X_T)))](https://render.githubusercontent.com/render/math?math=L_%7Bcyc%7D(X_T%2C%20G_S(G_T(X_T)))) and adversarial loss for source domain ![L_{adv}(X_S, D_S(X_T))](https://render.githubusercontent.com/render/math?math=L_%7Badv%7D(X_S%2C%20D_S(X_T))) are not shown due to symmetry.


![Figure 3.17: Class activation maps](theory/CAM.svg)

Figure 3.17: Class Activation Maps (CAM) to visually display model attention, calculated for a ResNet-18 model pre-trained on ImageNet. Second to fourth column show bilinearly interpolated class activation maps of class ranked 1st to 3rd (as overlay to original image).

![Figure 3.18: Cell inversion when using domain-transfer of microscopy images using CycleGAN](DomainAdaptation/cyclegan_cells.svg)

Figure 3.18: Cell inversion when using domain-transfer of microscopy images using CycleGAN.


# Confocal Laser Endomicroscopy
## Head and Neck Squamous Cell Carcinoma

![Figure 4.1: Upper layers of the skin with squamous cell carcinoma (SCC), a tumor developing from the upper layers of the epithelium.](SCC/SCC.svg)

Figure 4.1: Upper layers of the skin with squamous cell carcinoma (SCC), a tumor developing from the upper layers of the epithelium.

![Figure 4.2: Occurrence of oral squamous cell carcinoma](SCC/regions_SCC.svg)

Figure 4.2: Occurrence of oral squamous cell carcinoma, based on an image published in Aubreville et al. (2017): Automatic Classification of Cancerous Tissue in Laserendomicroscopy Images of the Oral Cavity using Deep Learning (Scientific Reports) (CC BY) visualization using data from Ariyoshi2008 et al.: Epidemiological study of malignant tumors in the oral and maxillofacial region: Survey of member institutions of the Japanese Society of Oral and Maxillofacial Surgeons, 2002 (International Journal of Clinical Oncology, 2008)

![Figure 4.3: Confocal laser endomicroscopy setup](CLE/intro/cle.svg)
Figure 4.3: Confocal laser endomicroscopy (\gls{CLE}) setup. A laser beam is generated and routed through multiple (partially moving) mirrors towards the tissue. There, incoming light is transformed by fluorescence into a different wavelength, which is then focused onto a photodetector. From: Maier et al., Medical Imaging Systems: An Introductory Guide (p. 79), 2018, CC BY

![Figure 4.4: CLE dynamic range compression: healthy mucosa, 8 bit](CLE/preproc/epithel_compressed.png)
![Figure 4.4: CLE dynamic range compression: input and compressed output histograms](CLE/preproc/epithel_hist.svg)
![Figure 4.4: CLE dynamic range compression: SCC, 8 bit](CLE/preproc/carc_compressed.png)
![Figure 4.4: CLE dynamic range compression: input and compressed output histograms](CLE/preproc/carc_hist.svg)
Figure 4.4: Image histograms and dynamic range compression results.

![Figure 4.5: Comparision between CLE images and histopathology images - SCC](SCC/SCC_CLE_Histo_SCC.svg)
![Figure 4.5: Comparision between CLE images and histopathology images - healthy epithelium](SCC/SCC_CLE_Histo_healthy.svg)

Figure 4.5: Comparision between CLE images and H&E-stained histopathology images showing SCC and healthy epithelium. Note that histopathology images are commonly taken perpendicular to the direction of CLE images. Thanks to Christof Bertram for providing the histopathology images.

![Figure 4.6a: CLE image of healthy mucosa of squamous cells - epithelial structure](CLE/interpretation/cellstructure)
![Figure 4.6b: CLE image of healthy mucosa of squamous cells - mucosa with vessels](CLE/interpretation/healthy2)

Figure 4.6: Healthy mucosa of squamous cells. The left panel image is from the oral cavity while the right panel is from a vocal fold examination.

![Figure 4.7a: epithelial structure](CLE/interpretation/vessellong.svg)
![Figure 4.7b: vessels, longitudinal cross-section](CLE/interpretation/vessel2.svg)
Figure 4.7: Healthy mucosa of squamous cells (vocal folds), both with vessels containing erythrocytes.


<object data="theory/PatternRecognition.pdf" type="application/pdf" width="700px" height="700px"> </object>
![Figure 1: Pattern recognition pipeline, showcasing the difference between traditional machine learning and deep learning.](theory/PatternRecognition.pdf)

