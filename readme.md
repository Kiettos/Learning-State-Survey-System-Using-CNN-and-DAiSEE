# DAiSEE Student Engagement Detection

A complete, production-structured deep learning pipeline built with **PyTorch** to detect student engagement and emotional states (Boredom, Engagement, Confusion, Frustration) using the **DAiSEE dataset**.

## 🛠️ Prerequisites \& Environment Setup

### Step 1: Install Python

Make sure you have **Python 3.11** installed on your system.

### Step 2: Create and Activate a Virtual Environment

Open your **Terminal** (macOS/Linux) or **PowerShell** (Windows) inside the project root directory, then run:

* **Create virtual environment:**
```bash
python -m venv venv
```

* **Activate virtual environment:**

  * **Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
    *Note: If Windows blocks the script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.*

  * **macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```


### Step 3: Install Dependencies

Install all required libraries specified in the requirements file:

```bash
pip install -r requirements.txt
```

\---

## 📥 Dataset Preparation

1. **Download the Dataset:**

   * Download the original DAiSEE dataset : https://www.kaggle.com/datasets/olgaparfenova/daisee
   * &#x20;For easy access and official sources, check the `download\_link.txt` file in this repository. 
   * Place the dataset structure matching: 

  ```text
  📂 DataSet/
  ├── 📂 Labels/
  └── 📂 Data/               
      ├── 📂 Train/         
      ├── 📂 Validation/     
      └── 📂 Test/
  ```

2. **Extract Video Frames:**

   * Run the frame extraction script to parse raw videos into evenly spaced image frames (`.jpg`) optimized for model training:

```bash
     python extract\_frames.py
     ```

\---

## 🚀 Model Training \& Evaluation

Once the dataset has been fully extracted and prepared, you can start the training process by running:

```bash
python train\_pytorch.py
```

### Key Training Features:

* **Custom CNN Architecture:** A multi-block convolutional neural network optimized for spatial feature extraction from video frames.
* **Data Augmentation:** Integrated transforms (Resize, Random Horizontal Flip, Color Jitter, Normalization).
* **Optimizer \& Scheduler:** Adam optimizer with `ReduceLROnPlateau` scheduler to dynamically lower learning rates when validation performance plateaus.
* **Evaluation Metric:** Multi-label Macro F1-score evaluation.
* **Checkpointing:** Automatically saves epoch weights and updates the best performing model weights (`best\_model.pth`) inside `checkpoints\_pytorch/`.

\---

## 📊 Results

The model evaluates performance across validation and test splits using Macro F1-score to handle multi-label emotional classifications effectively. Final evaluations are printed upon training completion.

