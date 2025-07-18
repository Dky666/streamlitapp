import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# Title and introduction
st.title("20 Newsgroups Text Classifier")
st.write("Welcome to the Text Classifier App!")
st.write("This app uses the following models for classification:")
st.write("Naïve Bayes, SVM, Random Forest, Decision Tree")

# Sidebar with updated selectbox
cs = ["Naive Bayes", "SVM", "Random Forest", "Decision Tree"]
classification_space = st.sidebar.selectbox("Select a classifier", cs)

# Button to trigger classification
if st.sidebar.button('Classify'):
    trainData = fetch_20newsgroups(subset='train', shuffle=True)
    test_set = fetch_20newsgroups(subset='test', shuffle=True)

    # Naive Bayes
    if classification_space == "Naive Bayes":
        st.write("Naive Bayes selected")
        pipeline = Pipeline([
            ('bow', CountVectorizer()),
            ('vector', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])

    # SVM
    elif classification_space == "SVM":
        st.write("SVM selected")
        pipeline = Pipeline([
            ('bow', CountVectorizer()),
            ('vector', TfidfTransformer()),
            ('classifier', SGDClassifier(loss='hinge', penalty='l1', alpha=0.0005, l1_ratio=0.17))
        ])

    # Random Forest
    elif classification_space == "Random Forest":
        st.write("Random Forest selected")
        pipeline = Pipeline([
            ('bow', CountVectorizer()),
            ('vector', TfidfTransformer()),
            ('classifier', RandomForestClassifier(max_depth=2, random_state=0))
        ])

    # Decision Tree
    elif classification_space == "Decision Tree":
        st.write("Decision Tree selected")
        pipeline = Pipeline([
            ('bow', CountVectorizer()),
            ('vector', TfidfTransformer()),
            ('classifier', DecisionTreeClassifier())
        ])

    # Train and predict
    pipeline = pipeline.fit(trainData.data, trainData.target)
    predictions = pipeline.predict(test_set.data)

    # Display accuracy
    accuracy = np.mean(predictions == test_set.target)
    st.write(f"Accuracy of {classification_space}: {accuracy:.4f}")
