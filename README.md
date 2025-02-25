
# Stock Price Prediction using Machine Learning and Neural Networks  

## Project Overview  
This project aims to predict stock prices using various machine learning algorithms and a custom-built neural network. The study focuses on Amazon (AMZN) stock data, leveraging models such as Gaussian Naive Bayes, Logistic Regression, and Fully Connected Neural Networks (FCNN).  

## Features  
- Stock price prediction using multiple machine learning models  
- Feature engineering with SMA, EMA, and RSI indicators  
- Data preprocessing including normalization and encoding  
- Evaluation of models using Mean Squared Error, R-squared, F1-score, and confusion matrix  
- Custom-built Fully Connected Neural Network for improved predictive performance  

## Technologies Used  
- Python  
- Pandas and NumPy  
- Scikit-learn  
- Yahoo Finance API  
- Matplotlib  
- TensorFlow or Keras for Neural Networks  

## Project Structure  
StockPricePredictor-ML  
- data - Stock price data from Yahoo Finance  
- models - Machine learning and neural network models  
- notebooks - Jupyter Notebooks for experiments  
- src - Core Python scripts  
- README.md - Project documentation  
- requirements.txt - Python dependencies  
    

## Methodology  
1. Data Collection - Extract historical stock prices from Yahoo Finance  
2. Preprocessing - Handle missing values, normalize data, and engineer features  
3. Model Training - Implement and train models including  
   - Gaussian Naive Bayes  
   - Logistic Regression  
   - Fully Connected Neural Network  
4. Evaluation - Compare model performance using Mean Squared Error, R-squared, and F1-score  

## Results and Findings  
- Gaussian Naive Bayes achieved an F1-score of approximately 0.48  
- Categorical Naive Bayes improved performance with an F1-score of 0.65  
- Logistic Regression achieved an F1-score of 0.60  
- The Fully Connected Neural Network showed the best performance with  
  - Mean Squared Error of 7.72 for training and 2.95 for testing  
  - R-squared value of 0.87 for training and 0.94 for testing  
  - The neural network was the best-performing model in capturing stock price patterns  

## Setup and Installation  
1. Clone the repository  
   git clone https://github.com/Akhil-Jonnalagadda/StockPricePredictor-ML.git 
   cd StockPricePredictor-ML  
2. Install dependencies  
   pip install -r requirements.txt  
3. Run the model  
   python src/train_model.py  

## Future Improvements  
- Enhance the Fully Connected Neural Network with LSTM for time-series forecasting  
- Use ensemble learning to combine multiple models  
- Integrate real-time stock market data for live predictions  

## Acknowledgments  
Special thanks to **Professor Mohammad Alam** for his invaluable guidance and mentorship in the field of machine learning and artificial intelligence. His expertise has been instrumental in shaping the foundation of this project.  

This project was developed as part of an Artificial Intelligence course.  
