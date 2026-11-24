# Customer Purchase Prediction

**Mathematics Mini Project** — Using classification algorithms to predict customer buying behaviour.

## Aim
To use classification algorithms and mathematical concepts to predict whether a customer is likely to purchase a product.

## Dataset
The project uses a small synthetic dataset (`data/customer_data.csv`) with customer attributes such as age, income, spending score, previous purchases, website visits and discount usage. The target column `Purchased` is 0 (No) or 1 (Yes).

## Mathematics Used
- **Logistic Regression:** sigmoid function, \(P(y=1)=1/(1+e^{-z})\)
- **K-Nearest Neighbors:** Euclidean distance
- **Decision Tree:** feature-based decision rules
- **Naive Bayes:** Bayes' theorem and conditional probability
- **Accuracy:** \((TP+TN)/(TP+TN+FP+FN)\)

## Classifiers
The program compares Logistic Regression, K-Nearest Neighbors, Decision Tree and Gaussian Naive Bayes using test-set accuracy, then displays a bar graph.

## Run the Project
1. Install Python.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python main.py
   ```

## Files
```text
customer-purchase-prediction/
├── data/
│   └── customer_data.csv
├── main.py
├── requirements.txt
└── README.md
```

## Note
This is a small educational project using synthetic data. Its predictions are illustrative and not intended for real business decisions.
