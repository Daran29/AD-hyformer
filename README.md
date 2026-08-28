# AD-HyFormer

AD-HyFormer is a hybrid CNN–Transformer framework for automated Alzheimer's disease classification using structural MRI (sMRI) data.

The project combines convolutional neural networks for extracting local anatomical features with Transformer-based architectures for learning global contextual relationships from MRI data.

## Project Status

🚧 **Ongoing Project**

The model, preprocessing pipeline, and experimental components are currently under development.

## Objectives

- Develop a hybrid CNN–Transformer architecture for Alzheimer's disease classification.
- Process and enhance structural MRI (sMRI) images.
- Utilize multi-slice MRI information for richer anatomical representation.
- Evaluate model performance across different Alzheimer's disease stages.
- Explore explainability techniques for model predictions.

## Project Structure

```text
AD-hyformer/
│
├── datasets/          # Dataset files
├── preprocessing/     # MRI preprocessing and enhancement
├── utils/             # Utility functions
├── tests/             # Testing scripts
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md