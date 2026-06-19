## README: TikTok User Verification Status Predictor

## Overview

This project develops a **Logistic Regression model** to predict the `verified_status` of users on a simulated TikTok dataset. The notebook details a complete machine learning workflow, focusing on challenges inherent in real-world data, such as **severe class imbalance** (94% unverified, 6% verified). We cover data simulation, feature engineering, preprocessing (including addressing imbalance), model training, and comprehensive evaluation. The goal is to build a predictive model and derive insights into factors influencing user verification.

## Project Structure

The notebook is organized into the following sections:

1.  **Data Simulation:** Creation of a synthetic TikTok dataset that mimics user and video characteristics, including a highly imbalanced target variable.
2.  **Feature Engineering:** Extraction of relevant features, specifically the length of video transcription text.
3.  **Preprocessing:** Steps to prepare the data for the Logistic Regression model, including one-hot encoding categorical variables, splitting data into training and testing sets, handling class imbalance using `RandomOverSampler`, and scaling numerical features.
4.  **Model Training:** Implementation and training of a Scikit-learn `LogisticRegression` model on the balanced training data.
5.  **Evaluation:** Assessment of the model's performance using appropriate classification metrics for imbalanced datasets, such as the Confusion Matrix, Classification Report (Precision, Recall, F1-score for each class), and overall Accuracy. Interpretation of model coefficients to understand feature importance.

## Dataset Description

The simulated dataset contains 10,000 rows and the following features:

*   **`video_id`**: Unique identifier for each video (dropped after simulation).
*   **`video_duration_sec`**: Duration of the video in seconds (numerical).
*   **`video_transcription_text`**: Text transcription of the video (used for feature engineering, then dropped).
*   **`claim_status`**: Categorical variable indicating the claim status of the content.
*   **`author_ban_status`**: Categorical variable indicating the author's ban status.
*   **`views`, `likes`, `shares`, `comments`**: Engagement metrics (numerical).
*   **`text_length`**: Engineered feature representing the length of `video_transcription_text` (numerical).
*   **`verified_status`**: Binary target variable (0 = unverified, 1 = verified) with a 94/6% class imbalance.

## Key Findings

*   **Class Imbalance Handling:** `RandomOverSampler` effectively balanced the training data, improving the model's ability to detect the minority class (verified users).
*   **Model Performance:**
    *   **Overall Accuracy:** Approximately 50.30%.
    *   **Recall (Verified=1):** 52.50%. This shows the model correctly identified over half of the actual verified users, which is a significant improvement given the severe imbalance.
    *   **Precision (Verified=1):** 6.30%. This low precision indicates that a high number of users predicted as 'verified' were actually unverified (false positives). This is a common trade-off when using oversampling to boost recall in highly imbalanced datasets.
    *   **F1-Score (Verified=1):** 11.25%. A balanced metric, reflecting the overall modest performance for the minority class.
*   **Feature Importance (from Coefficients):**
    *   `author_ban_status_under review` showed the strongest negative influence on verification likelihood.
    *   `claim_status_partially claimed` also had a notable negative impact.
    *   `text_length` and `comments` had positive (though smaller) influences, suggesting longer content or more comments might slightly increase verification probability.

## Technologies Used

*   **Python**
*   **pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib & Seaborn:** For data visualization (though limited in this specific output).
*   **Scikit-learn:** For machine learning tasks (preprocessing, model training, evaluation).
*   **imbalanced-learn (`imblearn`):** Specifically `RandomOverSampler`, for handling class imbalance.

## Getting Started

To run this notebook:

1.  **Open in Google Colab:** Upload the `.ipynb` file to Google Colab.
2.  **Run Cells:** Execute all cells sequentially. The notebook is self-contained and will generate its own synthetic data.

No external data downloads or specific environment setups are required beyond the default Colab environment, which includes all necessary libraries.
