This project aims to develop a sentiment analysis system using logistic regression to classify the sentiment of tweets. The dataset used for this project was sourced from Kaggle, consisting of 1.6 million tweets labeled with sentiments. Sentiment analysis is crucial for understanding public opinion, monitoring brand reputation, and making informed business decisions.

DATASET
The dataset used in this project was sourced from Kaggle and contains 1.6 million tweets. Each tweet is labeled with either positive or negative sentiment. The dataset includes the following fields:

Sentiment: The sentiment of the tweet (0 = negative, 4 = positive).
Tweet ID: The unique identifier for each tweet.
Date: The date and time the tweet was posted.
Query: The query (if any) that was used to obtain the tweet.
User: The user who posted the tweet.
Text: The text content of the tweet.

DATA PREPROCESSING
Data preprocessing is a crucial step in preparing the dataset for model training. The following steps were taken to preprocess the data:

Text Cleaning: Removed URLs, special characters, punctuation, numbers, and stopwords from the tweets.
Tokenization: Split the text into individual words (tokens).
Stemming: Reduced words to their base or root form using the Porter Stemmer.
Vectorization: Converted the text data into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) method.

LOGISTIC REGRESSION MODEL
Logistic regression is a simple yet powerful classification algorithm that is widely used for binary classification tasks. In this project, logistic regression was applied to classify tweet sentiments.
Training and Test Split: The dataset was split into training (80%) and test (20%) sets.
Model Training: The logistic regression model was trained on the training set using the scikit-learn library in Python.
Model Evaluation: The model was evaluated on the test set using accuracy as performance metrics.


DEEP NEURAL NETWORK MODEL

The DNN model was built using the Keras library with TensorFlow as the backend. The architecture includes:

Input Layer: Accepts the TF-IDF vectors of tweets.
Hidden Layers: Multiple dense (fully connected) layers with ReLU activation functions.
Output Layer: A single neuron with a sigmoid activation function for binary classification.

Training and Test Split: The dataset was split into training (80%) and test (20%) sets.
Model Compilation: The model was compiled using the Adam optimizer and binary cross-entropy loss function.
Model Training: The model was trained on the training set with early stopping to prevent overfitting.
Model Evaluation: The model was evaluated on the test set using accuracy as performance metrics.


There was an accurcay improvement of over 15% in DNN model.
