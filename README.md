ANN_BINARYVLE
# **ANN-Based Modeling of Vapor–Liquid Equilibrium (VLE) for Ethanol–Water System**

This project implements an **Artificial Neural Network (ANN)** to model the **Vapor–Liquid Equilibrium (VLE)** behavior of the **ethanol–water binary mixture**, including prediction of the **azeotropic composition**.
The model is trained on synthetically generated VLE data and successfully learns the **non-ideal thermodynamic behavior** of the system, outperforming traditional Raoult’s-law predictions.

---

## **📌 Project Overview**

The ethanol–water system exhibits strong non-idealities, including an azeotrope that classical ideal models fail to predict.
This project demonstrates how **machine learning**, specifically a neural network, can learn such complexities directly from data.

Key capabilities:

* Generates a dense synthetic dataset for ethanol–water VLE

* Trains an ANN model on (x₁, T, P) → y₁ mapping
* Predicts vapor mole fractions with high accuracy
* Detects azeotropic composition by solving **y₁ = x₁**
* Compares ANN predictions to Raoult’s Law
* Includes evaluation via parity plots

---

## **📁 Repository Structure**

```
📦 ANN_BinaryVLE
├── ANN_BinaryVLE.py            # Main code (training, prediction, azeotrope detection)
├── ANN_BinaryVLE_Report.pdf    # Full project report
├── ANN_BINARYVLE_CODE.pdf      # Code reference PDF
└── README.md                   # (This file)
```

---

## **🔬 Methodology**

### **1. Dataset Generation**

A synthetic dataset of ~450 points is generated to resemble experimental VLE for ethanol–water.
Dense sampling is performed near **x₁ ≈ 0.90** to capture azeotropic behavior.
Inputs and outputs:

* **Inputs:** liquid mole fraction (x₁), temperature (T), pressure (P)
* **Output:** vapor mole fraction (y₁)

Details in report:


---

### **2. ANN Model Architecture**

Architecture used (TensorFlow/Keras):

* **Input layer:** 3 features
* **Hidden layers:** 64 neurons × 2 (ReLU activation)
* **Output layer:** 1 neuron (sigmoid ensures 0 ≤ y₁ ≤ 1)


Loss function: **MSE**
Optimizer: **Adam**

---

### **3. Training**

* 80–20 train-test split
* 200 epochs
* Batch size: 32
* Min-Max scaling applied to inputs & outputs


Validation loss closely tracks training loss, showing good generalization.

---

### **4. Model Evaluation**

* Parity plot of predicted vs. actual y₁
* Accurate regression across full composition range
* Successfully detects azeotrope via root-finding (fsolve):
  **Predicted azeotrope:** x₁ ≈ 0.89
  (matches well with known ethanol–water azeotrope ≈ 0.90)


The model outperforms Raoult’s Law, which cannot predict azeotropic behavior.


---

## **📈 Results Summary**

* ANN accurately captures nonlinear VLE relationships
* Predicted azeotropic composition and temperature match literature values
* Demonstrates strength of **data-driven thermodynamic modeling**
* Provides a foundation for extending to multi-component systems


---

## **🚀 How to Run**

1. Install required dependencies:

   ```bash
   pip install numpy matplotlib tensorflow scikit-learn scipy
   ```

2. Run the main script:

   ```bash
   python ANN_BinaryVLE.py
   ```

Outputs:

* Parity plot visualization
* Azeotrope prediction
* Printed training/evaluation logs

---

## **📚 References**

* *ANN_BinaryVLE_Report.pdf* — Comprehensive methodology and analysis
* *ANN_BINARYVLE_CODE.pdf* — Extracted and formatted version of the Python code
* Main training script: ANN_BinaryVLE.py


