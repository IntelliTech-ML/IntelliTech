README: |
  # 🚀 IntelliNeuro PerceptronX
  ## 🤖 Custom Perceptron Machine Learning Library

  **Author:** Ajay Soni

  ---
  ## 📚 Introduction

  **IntelliNeuro’s PerceptronX** is a fully custom-built Perceptron machine learning library created from scratch,
  utilizing only **NumPy** and **pandas** for efficient numerical computations and data handling.

  This library provides:
  - 🔹 Linear regression
  - 🔹 Binary classification with sigmoid activation
  - 🔹 Foundational multi-class classification using softmax activation

  It is designed with educational clarity and modularity in mind,
  perfect for learners and practitioners eager to grasp the inner workings of perceptrons
  through clean, step-by-step gradient descent optimization.

  ---
  ## ✨ Key Features

  - **🎯 Versatile Learning Tasks**
    - Linear regression for continuous value prediction
    - Binary classification with sigmoid activation
    - Multi-class classification via softmax (demo-level support)

  - **⚙️ Robust Preprocessing Support**
    - Built-in scaling options: `none`, `minmax`, and `standard`
    - Validation to ensure correct scaling usage

  - **⚡ Efficient Training Pipeline**
    - Gradient descent optimization with up to 2,500,000 iterations
    - Early stopping based on tolerance threshold
    - Customizable learning rate and validation split
    - Detailed verbose training logs for monitoring

  - **📊 Comprehensive Prediction & Evaluation**
    - Seamless predictions for all supported tasks
    - Wide range of evaluation metrics:
      - Regression: MSE, RMSE, RMSLE
      - Binary classification: Accuracy, Precision, Recall, F1
      - Multi-class classification: Accuracy, Weighted Precision, Recall, F1

  - **👌 User-Friendly & Informative**
    - Clear, color-coded terminal outputs (via colorama)
    - Helpful warnings and error messages to guide usage

  ---
  ## 🚀 Quickstart Guide

  1. **Install and Import the library:**
	


     ```python
     pip install IntelliNeuro==0.1.0
     from PerceptronX import Perceptron
     ```

  2. **Initialize the model:**

     ```python
     model = Perceptron(
         learning_rate=0.001,
         validation_split=0.2,
         scaling='none',  # Options: 'none', 'minmax', 'standard'
         is_scaled=False,
         tolerance=1e-6
     )
     ```

  3. **Train your model:**

     ```python
     model.fit(X_train, y_train)
     ```

  4. **Make predictions:**

     ```python
     predictions = model.predict(X_test)
     ```

  5. **Evaluate model performance:**

     ```python
     score = model.score(X_test, y_test, metrics='accuracy')
     print(f"Model Accuracy: {score}")
     ```

  ---
  ## 🔍 How It Works

  - **⚙️ Gradient Descent:**
    - Iteratively updates weights and bias to minimize loss
    - Different loss functions per task:
      - Linear regression: Mean Squared Error (MSE)
      - Binary classification: Log loss with sigmoid activation
      - Multi-class classification: Cross-entropy with softmax

  - **🔄 Scaling Techniques:**
    - Manual min-max and standard scaling implementations
    - Auto-validation to prevent misuse

  - **🧪 Validation:**
    - Splits data based on `validation_split` parameter
    - Prints validation metrics post-training for model monitoring

  - **🔔 Activation Functions:**
    - Sigmoid for binary classification tasks
    - Softmax for multi-class classification

  ---
  ## ⚠️ Important Notes

  - **Multi-class classification is experimental** and intended for learning purposes only.
  - Ensure your data is scaled appropriately or specify scaling parameters.
  - Predictions or scoring before training will raise errors.
  - Color-coded outputs help distinguish warnings, info, and errors clearly.

  ---
  ## 🛠️ Installation

  Requires Python 3.7+ and the following packages:

  - numpy
  - pandas
  - scikit-learn (for evaluation metrics)
  - colorama (for colored terminal output)

  Install dependencies with:

pip install numpy pandas scikit-learn colorama

---
## 💡 Best Practices

- Always preprocess your features correctly or enable built-in scaling.
- Use `validation_split` to track model performance during training.
- Treat multi-class functionality as a conceptual demonstration.
- Tune `learning_rate` and `tolerance` to balance accuracy and training speed.
- Watch console warnings carefully to avoid common pitfalls.

---
## 🤝 Support & Contributions

Author: Ajay Soni  
Repository: https://github.com/ml-beginner-learner/IntelliNeuro  
Email: programmingwithcode@gmail.com

Contributions, issues, and feedback are welcome!  
Please open an issue or pull request to help improve this library.

---
## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with proper attribution.
