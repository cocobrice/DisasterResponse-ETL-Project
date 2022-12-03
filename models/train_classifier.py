import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)
    X = df.message.values
    y = df.drop(['message', 'genre'], axis=1).values

    categories = df.drop(['message', 'genre'], axis=1).columns
    
    return X, y, categories
    
    
def tokenize(text):
    ## normalise text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    ##tokenise
    words = word_tokenize(text)
    
    #lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed

def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RidgeClassifier()))
    ])
    
    parameters = {
        'clf__estimator__alpha': [0.1, 0.6, 1]}

    cv = GridSearchCV(pipeline, param_grid=parameters) 
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    target_names = ['0','1']
    for x in range(0,len(category_names)):
        print(str(category_names[x]))
        print(classification_report(Y_test[x], y_pred[x], target_names=target_names))    


def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
       
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
