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

![Figure 5.1: Confocal laser endomicroscopy setup](CLE/intro/cle.svg)

Figure 5.1: Confocal laser endomicroscopy (\gls{CLE}) setup. A laser beam is generated and routed through multiple (partially moving) mirrors towards the tissue. There, incoming light is transformed by fluorescence into a different wavelength, which is then focused onto a photodetector. From: Maier et al., Medical Imaging Systems: An Introductory Guide (p. 79), 2018, CC BY

![Figure 5.2: CLE dynamic range compression: healthy mucosa, 8 bit](CLE/preproc/epithel_compressed.png)
![Figure 5.2: CLE dynamic range compression: input and compressed output histograms](CLE/preproc/epithel_hist.svg)
![Figure 5.2: CLE dynamic range compression: SCC, 8 bit](CLE/preproc/carc_compressed.png)
![Figure 5.2: CLE dynamic range compression: input and compressed output histograms](CLE/preproc/carc_hist.svg)

Figure 5.2: Image histograms and dynamic range compression results.

![Figure 5.3a: correlation, clear case](CLE/preproc/corr2.svg)
![Figure 5.3b: correlation, hard case](CLE/preproc/corr87.svg)
![Figure 5.3c: dot product, easy case](CLE/preproc/dotproduct2.svg)
![Figure 5.3d: dot product, hard case](CLE/preproc/dotproduct87.svg)

Figure 5.3: Correlation vectors are not showing clear maxima in all cases, but the length-adjusted product does. The blue line represents the correlation to the first frame, the green line that to the last frame.

![Figure 5.4: Comparision between CLE images and histopathology images - SCC](SCC/SCC_CLE_Histo_SCC.svg)
![Figure 5.4: Comparision between CLE images and histopathology images - healthy epithelium](SCC/SCC_CLE_Histo_healthy.svg)

Figure 5.4: Comparision between CLE images and H&E-stained histopathology images showing SCC and healthy epithelium. Note that histopathology images are commonly taken perpendicular to the direction of CLE images. Thanks to Christof Bertram for providing the histopathology images.

![Figure 5.5a: CLE image of healthy mucosa of squamous cells - epithelial structure](CLE/interpretation/cellstructure.svg)
![Figure 5.5b: CLE image of healthy mucosa of squamous cells - mucosa with vessels](CLE/interpretation/healthy2.svg)

Figure 5.5: Healthy mucosa of squamous cells. The left panel image is from the oral cavity while the right panel is from a vocal fold examination.

![Figure 5.6a: epithelial structure](CLE/interpretation/vessel_long.svg)
![Figure 5.6b: vessels, longitudinal cross-section](CLE/interpretation/vessel2.svg)

Figure 5.6: Healthy mucosa of squamous cells (vocal folds), both with vessels containing erythrocytes.

![Figure 5.7a: Common optical artifacts in CLE images - particle on lens](CLE/interpretation/optical.svg)
![Figure 5.7b: Common optical artifacts in CLE images - mucus](CLE/interpretation/mucus.png)

Figure 5.7: Common optical artifacts in CLE images (a: particle on lens, b: mucus)

![Figure 5.8: Motion artifacts in CLE images - partial](CLE/interpretation/motion.png)
![Figure 5.8: Motion artifacts in CLE images - full](CLE/interpretation/fullmotion.png)

Figure 5.8: Motion artifacts in CLE images (a: partial, b: full)

![Figure 5.9: Noise artifacts in CLE images - partially missing contact](CLE/interpretation/noise_partial.svg)
![Figure 5.9: Noise artifacts in CLE images - weak signal](CLE/interpretation/noisy.png)

Figure 5.9: Noise artifacts in CLE images (a: partially missing contact, b: weak signal)

![Figure 5.10: Recording sites for our oral cavity data set](SCC/regions_percentages.svg)

Figure 5.10: Recording sites for our oral cavity data set. Percentages reflect share of images from recorded side in final data set. Image based on Aubreville et al. (Sci Rep 7(1): 11979) (CC BY), with data from Table 1 of the same publication.

![Figure 6.1a](MotionArtifacts/elongated_vc_carc_89800.jpg)
![Figure 6.1b](MotionArtifacts/carc_elongated_oc_68240.jpg)
![Figure 6.1c](MotionArtifacts/elongated_90697.jpg)
![Figure 6.1d](MotionArtifacts/mild_66615.jpg)

## Detection of Motion Artifacts

Figure 6.1: Stretched cells due to motion artifacts for all conditions and data sets (a: carcinoma VC, b: carcinoma OC, c: healthy mucosa VC, c: healthy mucosa OC)

![Figure 6.2a](MotionArtifacts/carc_91390_streaky.jpg)
![Figure 6.2b](MotionArtifacts/carc_91390_streaky.jpg)
![Figure 6.2c](MotionArtifacts/severe_68199_carc_oc.jpg)
![Figure 6.2d](MotionArtifacts/streaky_66560_healthy_oc.jpg)

Figure 6.2: Streaky motion artifacts for all conditions and data sets. Information regarding malignancy is completely lost. (a: carcinoma VC, b: carcinoma OC, c: healthy mucosa VC, c: healthy mucosa OC)

![Figure 6.3d](MotionArtifacts/compressed_VF_peca.jpg)
![Figure 6.3d](MotionArtifacts/compressed_oc_peca.jpg)
![Figure 6.3d](MotionArtifacts/compressedCell_VC.jpg)
![Figure 6.3d](MotionArtifacts/compressed_OC.jpg)

Figure 6.3: Compressed cell motion artifacts for all conditions and data sets.  (a: carcinoma VC, b: carcinoma OC, c: healthy mucosa VC, c: healthy mucosa OC)

![Figure 6.4a](MotionArtifacts/motionProfile37.jpg)
![Figure 6.4b](MotionArtifacts/motionProfile38.jpg)
![Figure 6.4c](MotionArtifacts/motionProfile39.jpg)
![Figure 6.4d](MotionArtifacts/coveredImage37.jpg)
![Figure 6.4e](MotionArtifacts/coveredImage38.jpg)
![Figure 6.4f](MotionArtifacts/coveredImage39.jpg)
![Figure 6.4g](MotionArtifacts/fakeMotion37.jpg)
![Figure 6.4h](MotionArtifacts/fakeMotion38.jpg)
![Figure 6.4i](MotionArtifacts/fakeMotion39.jpg)
![Figure 6.4j](MotionArtifacts/original37.jpg)
![Figure 6.4k](MotionArtifacts/original38.jpg)
![Figure 6.4l](MotionArtifacts/original39.jpg)

Figure 6.4: Simulation of motion artifact generation.


![Figure 6.6: Visualization of the angle of maximum correlation feature](MotionArtifacts/corrAngle.svg)

Figure 6.6: Visualization of the angle of maximum correlation feature]{Visualization of the angle of maximum correlation feature. Left column shows streaky motion artifact while right column shows an example from healthy epithelium without artifacts.

![Figure 6.7a](MotionArtifacts/image1.png)
![Figure 6.7b](MotionArtifacts/image2.png)
![Figure 6.7c: ](MotionArtifacts/coeffMatrix.svg)
![Figure 6.7d: ](MotionArtifacts/coeffMatrix2.svg)
![Figure 6.7e: ](MotionArtifacts/corrAngleFeature_rad.svg)
![Figure 6.7f: ](MotionArtifacts/corrAngleFeature2_rad.svg)

Figure 6.7: Visualization of the angle of maximum correlation feature in all steps from real image patches.


![Figure 6.8: Overview of the deep convolutional model used for motion artifact detection.](MotionArtifacts/MotionDetection.svg)

Figure 6.8: Overview of the deep convolutional model used for motion artifact detection.

![Figure 6.9: Preprocessing of CLE images to reduce impact of black borders](SCC/PreProcessing.svg)

Figure 6.9: Preprocessing of CLE images to reduce impact of black borders.


![6.10a](MotionArtifacts/ROC_Motion_Diss.svg)
![6.10b](MotionArtifacts/compareMotion_depth_multi.svg)

Figure 6.10: Evaluation of motion artifact detection with the proposed deep neural network architecture vs. conventional approaches based on the described features or combinations thereof. Right panel recreated from Aubreville et al., Deep learning-based detection of motion artifacts in probe-based confocal laser endomicroscopy images, 2019, left panel re-calculated.

![Figure 6.11a](MotionArtifacts/motionDet_66560.jpg)
![Figure 6.11b](MotionArtifacts/motionDet_70436.jpg)
![Figure 6.11c](MotionArtifacts/motionDet_92380_compressed.jpg)
![Figure 6.11d](MotionArtifacts/motionDet_82597_falsepos.jpg)

Figure 6.11: Exemplary results of motion detection from proposed CNN architecture. Panel <b>(d)</b> shows physiologically sheared cells acquired in the vocal folds.


## Diagnosis of Head and Neck Squamous Cell Carcinoma

![Figure 7.1](SCC/ImagePatches.svg)

Figure 7.1: Division of the original, resized image into overlapping patches of size 80x80. From: Aubreville et al., Sci Rep 7(1):11979, 2017, CC BY.

![Figure 7.2](SCC/CNNtoolchain.svg)

Figure 7.2: Patch-based classification of CLE images using a LeNet5 classifier. From: Aubreville et al., Sci Rep 7(1):11979, CC BY.


![Figure 7.3](SCC/patchProbabilityFusion.svg)

Figure 7.3: Fusion of probabilities from the patch classifier. From: Aubreville et al., Sci Rep 7(1):11979, CC BY.

![Figure 7.4](SCC/CNNTFtoolchain.svg)

Figure 7.4: Overview of the transfer learning approach, based on Szegedy's Inception v3, pre-trained on the ImageNet database. From: Aubreville et al., Sci Rep 7(1):11979, CC BY.

![Figure 7.5a](SCC/roc.svg)
![Figure 7.5b](SCC/rocAL.svg)

Figure 7.5: Receiver Operating Characteristic (ROC) curve of cross-validation. All cross-validation runs were combined into a single result vector. Left panel shows overall results, right panel shows results only on alveolar ridge and labium.


![Figure 7.6a](SCC/patchOutcome.svg)
![Figure 7.6b](SCC/classOutcome.svg)

Figure 7.6: Posterior probabilities for class <b>carcinoma</b> on patch-based approach for randomly selected single patches (a) and complete images (b). Top row shows patches/images with high likelihood for being healthy mucosa, middle row shows uncertain picks and bottom row shows high likelihood for being SCC. From Aubreville et al., Sci Rep 7(1):11979, CC BY.

![Figure 7.7](SCC/sameDataset.svg)

Figure 7.7: ROC curve of the application of the patch-based probability fusion method on same data set, using LOPO cross-validation.

![Figure 7.8a](SCC/application_OC.svg)
![Figure 7.8b](SCC/application_VC.svg)

Figure 7.8: ROC curves for detection of malignancies in the respective data sets, when trained on the same data set (using LOPO cross-validation) or on a different data set.

![Figure 7.9a](SCC/OC_patientview.svg)
![Figure 7.9b](SCC/VC_patientview.svg)

Figure 7.9: Accuracy for all patients of the respective data sets, when trained on same or other data set.

![Figure 7.10a](SCC/patient7_gutartig_92350_p_p661_vc_lopo.jpg)
![Figure 7.10b](SCC/patient7_gutartig_92347_p_0436_ocvc.jpg)
![Figure 7.10c](SCC/patient7_gutartig_92347_p_0682_vc_lopo.jpg)
![Figure 7.10d](SCC/patient7_gutartig_92350_p_296_ocvc.jpg)

Figure 7.10: Exemplary color-coded posterior probabilities for the class <b>carcinoma </b>for patient seven of the vocal folds data set when training was performed on the VC data set (a+c) and the combined data set (OC+VC, panels b+d). Gray indicates 0.5, deep red indicates 1.0, deep blue indicates 0.0.

![Figure 7.11a](SCC/patient7oc_labium_73862_p0417_oc.jpg)
![Figure 7.11b](SCC/patient7oc_labium_73862_p0635_ocvc.jpg)
![Figure 7.11c](SCC/patient7oc_labium_73892_p0204_oc.jpg)
![Figure 7.11d](SCC/patient7oc_labium_73892_p0409_ocvc.jpg)

Figure 7.11: Exemplary color-coded posterior probabilities for the class <b>carcinoma</b> for patient seven of the oral cavity dataset, showing CLE images of the labium. Images show varying output for classifiers trained on the OC data set (panels a+c) as well as for the concatenated OC+VC data set (b+d). Gray indicates 0.5, deep red indicates 1.0, deep blue indicates 0.0.

![Figure 7.12](SCC/median.svg)

Figure 7.12: Histogram of median pixel values of images showing healthy mucosa in various anatomical regions from all 23 patients.

![Figure 7.13](SCC/comparison.svg)

Figure 7.13: Comparison of patch-based classification and whole image classification.

![Figure 7.14](SCC/sameDataset_ResNet.svg)

Figure 7.14: ROC curve for LOPO cross-validations for all data sets and the combination thereof, comparing the patch-based approach and the ResNet-based whole image classification approach.

![Figure 7.15](SCC/OC_patientview_ResNet.svg)

Figure 7.15: Individual patient accuracy results on the OC data set.


![Figure 7.16](SCC/patient6_posteriors.svg)

Figure 7.16: Analysis of posteriors by different models for OC patient six (top row). Bottom row shows example image of sequence 56 that was largely responsible for the difference in accuracy for the different models. Left and middle column show ResNet-based methods, visualization is representing malignancy CAM with red indicating 1.0 and blue indicating 0.0, right bottom panel shows posteriors of probability map of ppf method.

![Figure 7.17](SCC/VC_patientview_ResNet.svg)

Figure 7.17: Individual patient accuracy results on the VC data set.

# Bright-field Microscopy
## Introduction to Microscopy

![Figure 8.1](Microscopy/thinlensmodel.svg)

Figure 8.1: Illustration of focus and defocused images in the thin lens model. Adapted from Mualla, Aubreville and Maier: Microscopy (in: Medical Imaging Systems, p.69-90, CC BY)


![Figure 8.2](Microscopy/microscope.svg)

Figure 8.2: Microscope lens setup. Adapted from Mualla, Aubreville and Maier: Microscopy (in: Medical Imaging Systems, p.69-90, CC BY)

![Figure 8.3](Microscopy/processingsteps.svg)

Figure 8.3: The steps of tissue processing for microscopy slides.

![Figure 8.4a](Microscopy/hem_2.png)
![Figure 8.4b](Microscopy/eos_2.png)
![Figure 8.4c](Microscopy/h_e_2.png)

Figure 8.4: Hematoxylin-eosin stain components (a: hematoxylin, b: eosin, c: H&E) (reconstruction using Macenko's method for stain separation)

![Figure 8.5a](Microscopy/he.jpg)
![Figure 8.5b](Microscopy/Azan.jpg)
![Figure 8.5c](Microscopy/ck1.jpg)
![Figure 8.5d](Microscopy/Grocott-versilberung.jpg)
![Figure 8.5e](Microscopy/MGG.jpg)
![Figure 8.5f](Microscopy/turnbull.png)

Figure 8.5: Different stains for histology and cytology. From: Mualla, Aubreville and Maier: Microscopy (in: Medical Imaging Systems, p.69-90, CC BY)

![Figure 8.6](Microscopy/scanningarea.svg)
![Figure 8.6](Microscopy/scanningmode.svg)

Figure 8.6: Process of scanning of whole slide images. a: Glass slide with selected area, b: line-based vs. tile-based scanning.

![Figure 8.7](Microscopy/Pyramid.svg)

Figure 8.7: WSI file pyramid structure.

## Annotation of Whole Slide Images
![Figure 9.1](WSIAnno/Annotated_HPFs.svg)

Figure 9.1: Number of annotated high power fields (HPFs) per patient in publicly available data sets.

![Figure 9.2](WSIAnno/QuPath.png)

Figure 9.2: WSI annotation using QuPath. Annotations set using the brush tool (green annotation) and the point list tool (yellow annotation).

![Figure 9.3](WSIAnno/ScreeningPipeline_red.svg)

Figure 9.3: SlideRunner guided screening pipeline

![Figure 9.4](WSIAnno/PluginInterface.svg)

Figure 9.4: Overview of SlideRunner's plug-in interface.

![Figure 9.5a](WSIAnno/plugin_coloroverlay.png)
![Figure 9.5b](WSIAnno/plugin_annos.png)
![Figure 9.5c](WSIAnno/plugin_rgbimage.png)

Figure 9.5: Operation modes for plug-ins, differing in the objects they feed to the main software.

![Figure 9.6](WSIAnno/SlideRunner_UML.svg)

Figure 9.6: SlideRunner database model.

## A Large Dataset of Mitotic Figures and other Cell Types in Canine Mast Cell Tumors

![Figure 10.1](Dataset/Annotation.svg)

Figure 10.1: Dataset creation approaches for the manually expert-labeled (MEL) data set variant. Modified from Bertram et al., Sci Data 6(1):274, CC BY

![Figure 10.2](Dataset/AugmentedAnnotation-Schrott.svg)

Figure 10.2: CNN-aided division of the ambiguous class into hard examples, true mitotic figures and other cells resulting in the hard-example augmented expert labeled (HEAEL) dataset variant, from Bertram et al., Sci Data 6(1):274, CC BY

![Figure 10.3](Dataset/AugmentedMissingCandidates.svg)

Figure 10.3: Algorithm-aided identification of potentially missed mitotic cells, resulting in the object-detection augmented expert labeled (ODAEL) dataset variant. From Bertram et al., Sci Data 6(1):274, CC BY

![Figure 10.4](Dataset/Missed.svg)

Figure 10.4: Count of additional mitotic figures, identified by the dual-stage object detection pipeline, and assessed by both experts.

![Figure 10.5](Dataset/ConsistencyCheckPipeline.svg)

Figure 10.5: Dataset consistency check pipeline

![Figure 10.6](Dataset/disagreed_consensus_expert1.svg)
![Figure 10.6](Dataset/disagreed_consensus_expert2.svg)

Figure 10.6: Final consensus of disagreed labels in manually labeled data set compared to individual expert opinion from first (left panel) and second (right panel) expert.

![Figure 10.7a](Dataset/ImCenterPeriphery_50c50p_c91a842257ed2add5134_raw.svg)
![Figure 10.7b](Dataset/ImCenterPeriphery_50c50p_dd4246ab756f6479c841_raw.svg)
![Figure 10.7c](Dataset/CenterPeripheryBoxplots.svg)

Figure 10.7: Comparison of mitotic count in center and periphery of the tumor. Example in *a* and *b* show threshold of inner 50% area. Blue line represents border of the tumor while red line represents outline of the inner part and darkened area represents necrotic tissue not included in the area estimation. Panel *c* shows statistical comparison for different ratios.

![Figure 10.8a](Dataset/stage1_stage2_aposterioris_1024.svg)
![Figure 10.8b](Dataset/stage1_stage2_aposterioris_1024_missed.svg)

Figure 10.8: 2D Histogram of dependency between first and second stage model scores for true mitotic figures, evaluated on the train and validation set with patch size 1024.

![Figure 10.9a](Dataset/AblationStudy_HPF.svg)
![Figure 10.9b](Dataset/AblationStudy_WSI.svg)

Figure 10.9: Ablation study on the ODAEL data set. Left panel shows reduction of area used in training, while right panel shows reduction of number of WSIs used in training. From Bertram et al., Sci Data 6(1):274, CC BY

![Figure 10.10](Dataset/FigureAblation.svg)

Figure 10.10: Example of a single tumor case comparing ground truth and detections on a 10HPF-ablated data set and the full data set. The pipeline trained on the ablated data set introduces a large number of false positives (bottom right panel).



