# 🏥 Predict Medical Insurance Charges 💰

This project predicts individual medical insurance charges based on demographic and lifestyle information using machine learning models. The dataset is sourced from Kaggle and contains features such as age, sex, BMI, number of children, smoking status, and region. The goal of this project is to develop a model that accurately predicts insurance charges for individuals.

## 🚀 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SergKhachikyan/Regression_with_an_Insurance_Dataset.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Regression_with_an_Insurance_Dataset
    ```

3. **Install the dependencies:**
    Make sure Python 3.x is installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch the project:**
    Open the Jupyter Notebook interface to explore and run the notebooks:
    ```bash
    jupyter notebook
    ```

## 🔧 Technologies

This project uses the following technologies:
- **Python** 🐍: Programming language.
- **Pandas** 📊: Data manipulation and analysis.
- **NumPy** 🔢: Numerical computations.
- **Scikit-learn** 🔬: Machine learning library used for model building.
- **Matplotlib & Seaborn** 📈: Data visualization libraries.
- **Jupyter Notebook** 📓: Interactive environment to run and explore the code.

## 📝 How to Use

1. **Prepare the dataset:**
    - Download the dataset from Kaggle: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
    - Place your dataset inside the `data/` folder.

2. **Train the model:**
    - Open Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the relevant notebook (e.g., `insurance_model.ipynb`) and run the cells sequentially to preprocess the data, build the model, and start training.

3. **Make predictions:**
    After training the model, you can use the inference script to predict insurance charges based on new personal data:
    ```bash
    python src/inference.py --input_data path/to/your/input.csv
    ```

## 💡 Features

- **Predict Insurance Charges** 💰: Predict the medical insurance costs based on personal features like age, BMI, smoking status, and more.
- **Data Preprocessing** 🔄: Clean and preprocess raw data before feeding it to the machine learning model.
- **Model Evaluation** 📊: Evaluate the model’s performance using metrics like mean squared error (MSE) and R-squared.
- **Visualization** 🌈: Visualize training performance, model predictions, and feature importance.

## 🧠 Model Architecture

- **Input Layer**: Takes in features such as age, sex, BMI, children, smoker status, and region.
- **Regression Model**: A regression model (like Linear Regression or Random Forest) predicts the insurance cost.
- **Output Layer**: Outputs the predicted insurance charge for the individual.

## 🏆 Model Performance

- **Loss Function**: Mean Squared Error (MSE), suitable for regression tasks.
- **Metrics**: Model performance evaluated by R-squared (R²) and Mean Squared Error (MSE).

## 📊 Visualizations

- **Training Curves**: Visualize loss during training epochs (if applicable).
- **Model Predictions**: Compare predicted insurance charges with actual values.
- **Feature Importance**: Visualize the impact of features like BMI, smoking, and age on predictions.

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the project, open issues, or submit pull requests.
