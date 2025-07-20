
import streamlit as st
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("20 Newsgroups Text Classifier")

st.write("This app uses these models for classification: Naïve Bayes, SVM, Random Forest, Decision Tree.")

@st.cache_data
def load_data():
    train_data = joblib.load("train_20news.pkl")
    test_data = joblib.load("test_20news.pkl")
    return train_data, test_data

try:
    trainData, test_set = load_data()

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SGDClassifier(loss='hinge', penalty='l1', alpha=0.0005, l1_ratio=0.17),
        "Random Forest": RandomForestClassifier(max_depth=2, random_state=0),
        "Decision Tree": DecisionTreeClassifier(random_state=0)
    }

    cs = list(classifiers.keys())
    classification_space = st.sidebar.selectbox("Select Classifier", cs)

    if st.sidebar.button("Classify"):
        st.write(f"{classification_space} selected")

        classifier = classifiers[classification_space]

        classificationPipeline = Pipeline([
            ('bow', CountVectorizer()),
            ('vector', TfidfTransformer()),
            ('classifier', classifier)
        ])

        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        dataPrediction = classificationPipeline.predict(test_set.data)
        accuracy = np.mean(dataPrediction == test_set.target)

        st.write(f"Accuracy of {classification_space}: {accuracy:.4f}")

except FileNotFoundError:
    st.error("Dataset files not found. Please make sure 'train_20news.pkl' and 'test_20news.pkl' are in the same directory.")
