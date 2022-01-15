# RSNA-MICCAI-Brain-Tumor-Radiogenomic-Classification
Code used for kaggle RSNA-MICCAI Brain Tumor Radiogenomic Classification

##255th place solution 

### Data preprocessing steps
- Load T2w Images (T2w images provided best results during training)
- Removed Blank slices
- Crop slices
- SIZ (Spline interpolated zoom) [1] depth 64
- Normalise
- Zero centre

###Training

- Trained a Resnet50 3D with a LR scheduler (factor 10, tolerance 5) Initial LR=1e-2, batch size = 16
- Single fold cross validation
- Generally the results appeared like:

![image](/images/im1.png)

- Chose best 5 models from each training cycle and created a final ensemble for submission



[1] https://link.springer.com/chapter/10.1007%2F978-3-030-59354-4_15