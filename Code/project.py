import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0,'../..')
from my_evaluation import my_evaluation

class my_model():

    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]
    
    # Preprocess the description column
    def preprocess_text(self,text):
        
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        #stemmer = PorterStemmer()
        #tokens_text_stem = [strip_numeric(stemmer.stem(token)) for token in tokens]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        # Join tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)
        
        return processed_text
    
    def fit(self, X, y):
        # do not exceed 29 mins
        
        X['processed_description'] = X['description'].apply(self.preprocess_text)
        
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        XX = self.preprocessor.fit_transform(X['processed_description'])
        XX = pd.DataFrame(XX.toarray())  
        
        
        
        estimators = [
            ('RFC' ,RandomForestClassifier(n_estimators=200, random_state = 42)),
            ('KNC', KNeighborsClassifier(8)),
            #('RC',  RidgeClassifier()),
        ]
        
        param_dist = {
        'n_estimators': [100, 200, 300, 400],  
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [5, 7, 10, 12],
        'min_samples_split': [400],
        'max_features': ['sqrt']
        
        }
        
        #For hyperparameter tuning
        gb_estimator = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_dist, 
                                          n_iter=10, scoring='f1', cv=5, random_state=42)
    
        # Create StackingClassifier with GradientBoostingClassifier final estimator
        self.clf = StackingClassifier(
            estimators=estimators, 
            final_estimator=gb_estimator) 
        self.clf.fit(XX, y)
            
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X['processed_description'] = X['description'].apply(self.preprocess_text)
        XX = self.preprocessor.transform(X['processed_description'])
        predictions = self.clf.predict(XX)
        return predictions