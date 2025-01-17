#made by Angad Singh

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('as.csv') 

def pre(text):
    words = nltk.word_tokenize(text.lower())
    return ' '.join(words)

df['Processed'] = df['questions'].apply(pre)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['Processed'])

y = df['answers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

classifier = SVC(kernel='linear')

classifier.fit(X_train, y_train)

def chatbot():
    print("Chatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        processed_input = pre(user_input)
        input_vector = vectorizer.transform([processed_input])
        response = classifier.predict(input_vector)[0]
        print(f"Chatbot: {response}")


chatbot()