# 2-Stage Facial Landmark Tracking with TAPIR 


We designed a facial landmark tracking pipeline to combine pre-trained facial landmark detectors and TAPIR. Given an input video, our pipeline initializes query points using a facial landmark detector and tracks the points using TAPIR across all frames.

https://github.com/xiaoyezuo/facetrack/assets/50150299/5b6bdb10-9cc7-4064-838d-6a8756f2f2f6

We initialized query points for TAPIR with 4 existing facial landmark detection methods.

*Ensemble of Regression Trees (ERT)

  *This approach predicts facial landmarks based on a cascade of regression trees trained using gradient boosting. First it detects the face in the frame using Viola-Jones Facial Detector. Then the algorithm uses a "mean" face template as an initial approximation and iteratively refines the landmark positions. The main advantage of this algorithm is high speed(1 millisecond per face). However, it had been shown that neural network approaches have better performance than ERT.


*Pixel-in-Pixel Net (PIPNet)

    *PIPNet is a combined heatmap and direct regression approach. It's designed to address three main challenges of existing heatmap regression models: computational expense, lack of explicit global shape constraints, and domain gaps. PIPNet employs a novel detection head based on heatmap regression, enabling simultaneous score and offset predictions on low-resolution feature maps. Additionally, PIPNet includes a neighbor regression module that enhances robustness by introducing local constraints through the fusion of neighboring landmarks' predictions. The model also incorporates a self-training strategy with a curriculum, which effectively utilizes unlabeled data across domains, improving cross-domain generalization capability. The PIPNet was trained on the 300W dataset and tested on 300W, COFW- 68, and WFLW-68.

*Facial Alignment Network (FAN)

FAN utilizes a 2D-Facial Alignment Network to identify and locate facial landmarks. This network uses a stack of four modified state-of-the-art "HourGlass" networks, first used for human pose estimation. Each hourglass network uses a series of convolution and max pooling layers. The network branches off right before each max pooling layer to apply convolutions on the pre-pooled output in addition to the pooled output. Once the main series reaches $4\times 4$ resoultion, the network begins upsampling the lower-resolution outputs to add them with the higher-resolution branches and combine their features. The architecture is constructed in this way due to recent findings that consecutive convolutions of smaller filters can capture more features than fewer convolutions of larger filters. The output is a heatmap for each landmark predicting its position. 

*Self-adapTive Ambiguity Reduction (STAR)

  *STAR incorporates a loss function which aims to account for the semantic ambiguity of manual landmark annotations of faces in training datasets. The authors call Self-adapTive Ambiguity Reduction, or STAR loss. The authors used the four stacked HourGlass networks just as the Facial-Alignment network did, but applying the STAR loss after each HourGlass module. The first step of computing the STAR loss is performing principal component analysis on the predicted distribution of the facial landmarks. Generally, the direction of the first principal component is in the direction of facial contours, which happens to be the axis along which semantic ambiguity is found. The ratio of the eigenvalues of the first and second principal components defines the elliptical area of the ambiguity. The STAR loss takes in the eigenvectors and eigenvalues of these two principal components and makes the model pay less attention to prediction error in that direction since it may be due to semantic ambiguity rather than model inaccuracy.



