# 2-Stage Facial Landmark Tracking with TAPIR 


We designed a facial landmark tracking pipeline to combine pre-trained facial landmark detectors and TAPIR. Given an input video, our pipeline initializes query points using a facial landmark detector and tracks the points using TAPIR across all frames.

https://github.com/xiaoyezuo/facetrack/assets/50150299/5b6bdb10-9cc7-4064-838d-6a8756f2f2f6

We initialized query points for TAPIR with 4 existing facial landmark detection methods.

- Ensemble of Regression Trees (ERT): predicts facial landmarks based on a cascade of regression trees trained using gradient boosting.
  
- Pixel-in-Pixel Net (PIPNet): a combined heatmap and direct regression approach enabling simultaneous score and offset predictions on low-resolution feature maps
  
- Facial Alignment Network (FAN): a heatmap approach that uses a stack of four modified state-of-the-art "HourGlass" networks to predict landmark positions
  
- Self-adapTive Ambiguity Reduction (STAR): incorporates a loss function which aims to account for the semantic ambiguity of manual landmark annotations of faces in training datasets  


