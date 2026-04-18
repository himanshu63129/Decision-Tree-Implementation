DECISION-TREE-CLASSIFICATION
COMPANY: CODTECH IT SOLUTION 
NAME:HIMANSHU RAJ 
INTERN ID:CTIS6470 
DOMAIN:MACHINE LEARNING 
DURATION: 8 WEEKS 
MENTOR NEELA SANTOSH

# Iris Flower Classification using Decision Trees

## Project Overview
This project provides a comprehensive implementation and analysis of a Decision Tree Classifier applied to the famous Iris dataset. Decision Trees are a popular supervised learning method used for classification and regression. In this repository, we demonstrate how to build, visualize, and evaluate a Decision Tree model to classify iris flowers into three species based on their morphological measurements. 

The goal of this project is not only to achieve high classification accuracy but also to provide transparency into how the model makes its decisions. Through interactive visualizations and detailed performance metrics, we explore the underlying logic of the Decision Tree algorithm.

## Dataset: The Iris Flower Dataset
The dataset used in this project is the **Iris Flower Dataset**, first introduced by British statistician and biologist Ronald Fisher in his 1936 paper. It is perhaps the best-known database to be found in the pattern recognition literature.

### Features
The dataset consists of 150 records under five attributes: 
1. **Sepal Length** (in cm)
2. **Sepal Width** (in cm)
3. **Petal Length** (in cm)
4. **Petal Width** (in cm)
5. **Species** (Class Label)

### Target Classes
The target attribute consists of three iris species:
- **Setosa**
- **Versicolor**
- **Virginica**

One class is linearly separable from the other two; the latter are not linearly separable from each other, making it an excellent benchmark for classification algorithms.

## Technical Methodology: Decision Trees
A Decision Tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g., whether a petal is longer than 2.45cm), each branch represents the outcome of the test, and each leaf node represents a class label.

### How it Works
The algorithm works by partitioning the data into subsets based on feature values. This process is repeated in a recursive manner called recursive partitioning. The splits are chosen to maximize the "purity" of the resulting nodes. In this project, we utilize the **Gini Impurity** criterion, which measures how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.

### Implementation
We use the `scikit-learn` library, specifically the `DecisionTreeClassifier` class. The process involves:
1. **Data Preprocessing**: Loading data via `load_iris` and splitting it into training (70%) and testing (30%) sets.
2. **Training**: Fitting the model on the training data.
3. **Visualization**: Using `plot_tree` to generate a graphical representation of the decision paths.
4. **Evaluation**: Computing accuracy scores and generating a detailed classification report.

## Project Structure
- `decision_tree_analysis.ipynb`: A detailed Jupyter Notebook containing the full workflow, from data exploration to visualization and feature importance analysis.
- `verify_model.py`: A lightweight Python script designed for quick model verification and accuracy benchmarking.
- `README.md`: Comprehensive documentation (this file).

## Getting Started

### Prerequisites
Ensure you have Python installed. We recommend creating a virtual environment. You will need the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Installation
You can install the necessary dependencies using pip:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running the Analysis
To explore the model in-depth, open the Jupyter Notebook:
```bash
jupyter notebook decision_tree_analysis.ipynb
```
Follow the cells sequentially to see the data distribution, the trained tree, and the confusion matrix.

### Verification
To quickly verify the model performance, run the verification script:
```bash
python verify_model.py
```
This script will output the accuracy and confirm if it meets the 90% threshold.

## Results and Insights
The model consistently achieves an accuracy exceeding **94%** on the test set. Key insights from the analysis include:

- **Feature Importance**: Petal length and petal width are the most significant predictors for iris species classification. Sepal measurements have relatively lower predictive power in this specific model.
- **Setosa Separation**: The Setosa species is easily distinguished from the others using a single split (Petal Length <= 2.45cm), highlighting its distinct characteristics.
- **Model Transparency**: The visualized tree allows us to trace any single prediction from the root to the leaf, making it an "explainable" AI model.

## Future Enhancements
- **Hyperparameter Tuning**: Experimenting with `max_depth` and `min_samples_split` to prevent potential overfitting.
- **Cross-Validation**: Implementing k-fold cross-validation for more robust performance estimation.
- **Alternative Algorithms**: Comparing Decision Trees with Random Forests or Gradient Boosting to see if accuracy can be further improved.

---
*Created as part of the Codtech Task 1 - Machine Learning Internship.*

