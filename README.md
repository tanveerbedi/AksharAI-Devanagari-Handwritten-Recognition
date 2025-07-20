# 🧠 AksharAI: CNN-Based Real-Time Devanagari Handwritten Character Recognition

## 🌎 Project Abstract

This repository presents an end-to-end deep learning solution for recognizing handwritten characters in the Devanagari script, a widely used writing system for several Indian languages including Hindi, Sanskrit, and Marathi. The system demonstrates a robust machine learning pipeline, from data preprocessing and model training to real-time deployment via a web interface.

At the core of the application lies a **Convolutional Neural Network (CNN)**, implemented using **TensorFlow/Keras**, trained to identify and classify 46 distinct Devanagari characters. This model is seamlessly deployed using **Streamlit**, allowing users to draw characters on an interactive canvas and receive instant predictions. This project highlights the convergence of deep learning, intuitive UI design, and reproducible research workflows.

## ✨ Core Features & Technical Highlights

* **Interactive Drawing Interface:**
  Built using `streamlit-drawable-canvas`, this feature provides an intuitive drawing space for users to input Devanagari characters using mouse or touch. The canvas output is captured as image data, which is processed and fed into the trained CNN model.

* **Robust CNN Model:**
  A sequential convolutional neural network trained on grayscale 32x32-pixel images. It leverages multiple convolution and pooling layers to capture both low-level and high-level features crucial for accurate classification.

* **Optimized Data Pipeline:**
  The training process utilizes the TensorFlow `tf.data` API for high-performance data loading and augmentation. This ensures efficient training even on limited hardware and supports caching, prefetching, and parallel processing.

* **Modular Codebase:**
  The code is structured with best practices in mind:

  * `train_model.py` handles model building, training, evaluation, and saving.
  * `app.py` manages the Streamlit interface and user interaction logic.

* **Real-Time Inference:**
  The Streamlit app loads a pre-trained model (`devanagari_model.h5`) and performs live inference on user input, returning a prediction within milliseconds.

* **Reproducibility & Clarity:**
  All steps—from data handling to model inference—are clearly documented, ensuring that results can be replicated across machines.

## 🔧 CNN Model Architecture

The CNN is designed to process small 32x32 grayscale character images and output a classification across 46 categories. The architecture is as follows:

| Layer Type     | Parameters / Details                             | Purpose                                              |
| -------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Rescaling      | `scale=1./255, input_shape=(32, 32, 1)`          | Normalize pixel values to range \[0, 1]              |
| Conv2D         | `filters=32, kernel_size=3x3, activation='relu'` | Extract low-level features like edges                |
| MaxPooling2D   | `pool_size=2x2`                                  | Reduce spatial dimensions, retain important features |
| Conv2D         | `filters=64, kernel_size=3x3, activation='relu'` | Extract more complex patterns from image             |
| MaxPooling2D   | `pool_size=2x2`                                  | Further dimensionality reduction                     |
| Dropout        | `rate=0.25`                                      | Regularization to prevent overfitting                |
| Flatten        | -                                                | Convert feature maps to flat vector                  |
| Dense          | `units=128, activation='relu'`                   | Fully connected layer for feature synthesis          |
| Dropout        | `rate=0.5`                                       | Additional dropout for generalization                |
| Dense (Output) | `units=46, activation='softmax'`                 | Output classification over 46 character categories   |

The model is compiled with:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metric:** Accuracy

## 📂 Project Structure

```
.
├── DevanagariHandwrittenCharacterDataset/  # Manually downloaded dataset
│   ├── Train/                              # 46 folders, one for each class
│   └── Test/                               # Optional for model evaluation
├── train_model.py                          # Model training script
├── app.py                                  # Streamlit web interface
├── devanagari_model.h5                     # Trained model artifact
└── requirements.txt                        # List of dependencies (optional)
```

## ⚙️ Installation & Execution Guide

### Prerequisites

* Python 3.7 or higher
* pip (Python package installer)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### Step 2: Download Dataset

1. Visit the Kaggle page: [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase)
2. Download the `archive.zip` file
3. Extract and place the `DevanagariHandwrittenCharacterDataset/` in the project root

### Step 3: Setup Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install streamlit pandas numpy tensorflow opencv-python streamlit-drawable-canvas
```

### Step 4: Train the CNN Model

```bash
python train_model.py
```

This will preprocess data, train the model, and save the output as `devanagari_model.h5`.

### Step 5: Launch the Application

```bash
streamlit run app.py
```

Browser will open a GUI where you can draw a character and view its predicted class.

## 🤖 Future Work & Enhancements

* **Hyperparameter Optimization:** Utilize tools like `KerasTuner` or `Optuna` for automated architecture and parameter tuning.
* **Live Data Augmentation:** Integrate `ImageDataGenerator` or use `tf.image` transformations for on-the-fly training variations.
* **Transfer Learning:** Fine-tune pre-trained CNN architectures like ResNet-50 or MobileNetV2.
* **Model Explainability:** Add saliency maps or Grad-CAM to visualize what parts of the input influence model decisions.
* **Cloud Hosting:** Deploy the app publicly via Streamlit Community Cloud, Hugging Face Spaces, or Heroku.


---

## 📸 Application Screenshot

![App Screenshot](https://github.com/tanveerbedi/AksharAI-Devanagari-Handwritten-Recognition/blob/f4d53876bd589bfc896d65b8c930579488048830/Interface%20Glimpse.png)

*Figure: The Streamlit interface showing the prediction result for a handwritten Devanagari character.*

---

## 👨‍💻 Author

**Tanveer Singh Bedi**

Machine Learning and Data Science Practitioner

[LinkedIn](https://www.linkedin.com/in/tanveer-singh-bedi-a8b811177) • [GitHub](https://github.com/tanveerbedi)

---

## 📄 License

Distributed under the **MIT License**. See the `LICENSE` file for more details.

## 🙏 Acknowledgements

* **Dataset Source:** [Kaggle - Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase)
* **Tooling:** Built using open-source libraries including TensorFlow, Streamlit, NumPy, and OpenCV.
* Special thanks to the open-source community and contributors to Streamlit and TensorFlow ecosystems.
