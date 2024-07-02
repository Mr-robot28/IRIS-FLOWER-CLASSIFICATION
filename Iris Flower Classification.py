import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
iris_df = pd.read_csv(r'C:\Users\sumedh hajare\Downloads\Iris.csv')
iris_df = iris_df.drop(columns=['Id'])

# Display basic information
print("Dataset Overview:")
print(iris_df.info())
print("\nSummary Statistics:")
print(iris_df.describe())
print("\nClass Distribution:")
print(iris_df['Species'].value_counts())
print("\nMissing Values:")
print(iris_df.isnull().sum())

# Visualizations
def plot_histogram(data, column, title):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=column, kde=True)
    plt.title(title)
    plt.show()

for feature in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    plot_histogram(iris_df, feature, f'Distribution of {feature}')

def scatter_plot(data, x, y, hue, title):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, palette='viridis')
    plt.title(title)
    plt.show()

feature_pairs = [
    ('SepalLengthCm', 'SepalWidthCm'),
    ('PetalLengthCm', 'PetalWidthCm'),
    ('SepalLengthCm', 'PetalLengthCm'),
    ('SepalWidthCm', 'PetalWidthCm')
]

for x, y in feature_pairs:
    scatter_plot(iris_df, x, y, 'Species', f'{x} vs {y}')

# Prepare data for modeling
le = LabelEncoder()
iris_df['Species'] = le.fit_transform(iris_df['Species'])

X = iris_df.drop(columns=['Species'])
y = iris_df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training and evaluation
def train_and_evaluate(model, name):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100
    print(f"{name} Accuracy: {accuracy:.2f}%")

models = [
    (LogisticRegression(random_state=42), "Logistic Regression"),
    (KNeighborsClassifier(), "K-Nearest Neighbors"),
    (DecisionTreeClassifier(random_state=42), "Decision Tree")
]

for model, name in models:
    train_and_evaluate(model, name)