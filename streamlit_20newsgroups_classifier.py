
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

st.title("20 Newsgroups Text Classifier")

st.write("This app uses these models for classification: Naïve Bayes, SVM, Random Forest, Decision Tree.")

cs = ["Naive Bayes", "SVM", "Random Forest", "Decision Tree"]
classification_space = st.sidebar.selectbox("Select Classifier", cs)

if st.sidebar.button("Classify"):
    trainData = fetch_20newsgroups(subset='train', shuffle=True)
    test_set = fetch_20newsgroups(subset='test', shuffle=True)

    if classification_space == "Naive Bayes":
        st.write("Naive Bayes selected")
        classifier = MultinomialNB()
    elif classification_space == "SVM":
        st.write("SVM selected")
        classifier = SGDClassifier(loss='hinge', penalty='l1', alpha=0.0005, l1_ratio=0.17)
    elif classification_space == "Random Forest":
        st.write("Random Forest selected")
        classifier = RandomForestClassifier(max_depth=2, random_state=0)
    elif classification_space == "Decision Tree":
        st.write("Decision Tree selected")
        classifier = DecisionTreeClassifier(random_state=0)

    classificationPipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('vector', TfidfTransformer()),
        ('classifier', classifier)
    ])

    classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
    dataPrediction = classificationPipeline.predict(test_set.data)
    accuracy = np.mean(dataPrediction == test_set.target)

    st.write(f"Accuracy of {classification_space}: {accuracy:.4f}")
