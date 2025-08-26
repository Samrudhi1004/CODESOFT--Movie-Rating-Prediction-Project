# 🎬 Movie Rating Prediction

This project focuses on building a machine learning model to predict movie ratings based on various features such as genre, director, and actors. The goal is to analyze historical movie data and develop a model that can accurately estimate movie ratings.

## 📁 Project Structure
Movie_Rating_Prediction/
|-- data/
|   |-- IMDb Movies India.csv
|
|-- scripts/
|   |-- predict_rating.py
|
|-- README.md

-   **`data/`**: Contains the dataset used for the project.
-   **`scripts/`**: Houses the main Python script for data preprocessing and model training.
-   **`README.md`**: This file, providing an overview and instructions.

## 🚀 Getting Started

### Prerequisites

You need to have Python and the following libraries installed:
-   `pandas`
-   `scikit-learn`
-   `numpy`

You can install them using pip:
bash
pip install pandas scikit-learn numpy

How to Run the Code ?
Place the dataset: Ensure the IMDb Movies India.csv file is inside the data/ directory.

Execute the script: Run the predict_rating.py script from the terminal.

Bash

python scripts/predict_rating.py

📊 Results :

The script will output the performance metrics of the trained model.

Mean Absolute Error (MAE): Measures the average difference between the predicted and actual ratings. A lower value indicates a better model.

R-squared (R2) Score: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A score closer to 1 indicates a better fit.

Example Output:

Mean Absolute Error (MAE): 0.9364674637959612
R-squared (R2) score: 0.22337748868481644
The results show that the model's predictions are, on average, within ~0.94 points of the actual rating. The R-squared score of ~0.22 indicates that the model explains approximately 22% of the variance in the movie ratings. This suggests that while the model provides a reasonable baseline, further improvements through advanced feature engineering or different machine learning models could enhance its accuracy.
