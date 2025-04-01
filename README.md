# Diffusion Models: Course Notes & Notebooks

## Course Overview

This repository contains comprehensive notes and Jupyter notebooks from the **"How Diffusion Models Work"** short course by [DeepLearning.AI](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/). The course provides a hands-on introduction to **diffusion models**, a powerful class of generative AI models used in image generation. By the end of this course, you will:

- Understand the **fundamentals of diffusion models** and how they generate data.
- Build and train a diffusion model **from scratch**.
- Implement **efficient sampling techniques** to speed up inference.
- Develop neural networks for **noise prediction** and **context-aware generation**.

This repository includes structured notes and fully functional Jupyter notebooks to guide you through the process.

## Course Content

### 1. [**Diffusion Model Sampling**]()
- **Model Definition**: Implements a **U-Net architecture** with context conditioning for denoising images.
- **Diffusion Process Setup**:
  - Defines **timesteps, noise schedules, and feature dimensions**.
  - Constructs **alpha values** to track signal decay over time.
- **Denoising & Sampling**:
  - Implements **DDPM sampling** to remove noise step by step.
  - Demonstrates incorrect sampling to illustrate failure cases.
- **Visualization**:
  - Generates and animates sampled images.

### 2. [**Diffusion Model Training**]()
- **Model Architecture**:
  - Uses **U-Net** with **residual convolutional blocks**.
  - Embeds time information and optional context.
- **Training Process**:
  - Adds Gaussian noise to images.
  - Trains the model to predict and remove noise.
  - Uses **Mean Squared Error (MSE) loss** for learning.
  - Saves model checkpoints every few epochs.
- **Visualization & Checkpoints**:
  - Loads trained models and generates sample images.
  - Creates animations of generated outputs at different training epochs.

### 3. [**Contextual Control in Diffusion Models**]()
- **Extends U-Net with Context Embeddings**:
  - Enables **controllable image generation** based on class labels.
  - Incorporates **context dropout** during training.
- **Training Process**:
  - Uses an optimizer with learning rate decay.
  - Predicts and removes noise while applying context.
  - Saves models periodically.
- **Sampling & Visualization**:
  - Uses **DDPM sampling with contextual embeddings**.
  - Generates images conditioned on specific categories.
  - Implements **context mixing** to create hybrid images.

### 4. [**Fast Sampling in Diffusion Models**]()
- **Optimized Sampling with DDIM**:
  - Introduces **Denoising Diffusion Implicit Models (DDIM)** to speed up inference.
  - Removes noise **without adding additional stochastic noise**.
- **Comparing DDPM vs. DDIM**:
  - DDPM: Traditional step-by-step denoising.
  - DDIM: **Fast sampling by skipping intermediate steps**.
  - **Benchmarking speed** using `%timeit`.
- **Context-Aware Fast Sampling**:
  - Integrates class labels for controlled sampling.
  - Generates and visualizes images efficiently.

## Notebooks
- 📂 [`L1_Sampling.ipynb`]() – Implements **basic diffusion model sampling**.
- 📂 [`L2_Training.ipynb`]() – Covers **training a diffusion model from scratch**.
- 📂 [`L3_Context.ipynb`]() – Explores **context-aware diffusion models**.
- 📂 [`L4_FastSampling.ipynb`]() – Demonstrates **fast sampling techniques using DDIM**.

## Getting Started

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-repo/diffusion-models.git
cd diffusion-models
```

### 2. **Set Up the Environment**
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. **Run the Notebooks**
Launch Jupyter Notebook and explore the code:
```bash
jupyter notebook
```
Open any of the `.ipynb` files and run the cells to execute the diffusion model pipeline.

## Resources & References
- [DeepLearning.AI Course on Diffusion Models](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
- [Denoising Diffusion Probabilistic Models (DDPM) Paper](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (DDIM) Paper](https://arxiv.org/abs/2010.02502)
- [minDiffusion Repository](https://github.com/cloneofsimo/minDiffusion)
- [LucidRains' Diffusion Model](https://github.com/lucidrains/denoising-diffusion-pytorch)

## Acknowledgments
- Sprites from ElvGames, FrootsnVeggies, kyrise.
- Code adapted from various open-source diffusion model implementations.

Happy experimenting with diffusion models! 🚀

