import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

CATS=['Month','VisitorType','OperatingSystems','Browser','Region','TrafficType','Weekend']
NUMS=['Administrative','Administrative_Duration','Informational','Informational_Duration',
      'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']
EXTRAS=['TotalPages','TotalDuration','AverageTimePerPage','ProductPageShare','ProductTimePerPage']

class FeatureEngineer(TransformerMixin,BaseEstimator):
    def __init__(self,enabled=True): self.enabled=enabled
    def fit(self,X,y=None): return self
    def transform(self,X):
        d=X.copy()
        if self.enabled:
            d['TotalPages']=d[['Administrative','Informational','ProductRelated']].sum(axis=1)
            d['TotalDuration']=d[['Administrative_Duration','Informational_Duration','ProductRelated_Duration']].sum(axis=1)
            d['AverageTimePerPage']=d['TotalDuration']/d['TotalPages'].clip(lower=1)
            d['ProductPageShare']=d['ProductRelated']/d['TotalPages'].clip(lower=1)
            d['ProductTimePerPage']=d['ProductRelated_Duration']/d['ProductRelated'].clip(lower=1)
        return d

def variants(name,oversampled=False):
    # Class weights and oversampling are not compounded in these experiments.
    weight=None if oversampled else 'balanced'
    choices={
      'KNN':[KNeighborsClassifier(n_neighbors=5,weights='distance',n_jobs=-1),
             KNeighborsClassifier(n_neighbors=21,weights='distance',n_jobs=-1)],
      'Logistic Regression':[LogisticRegression(C=c,class_weight=weight,max_iter=2000,random_state=42) for c in [.1,1]],
      'Random Forest':[RandomForestClassifier(n_estimators=200,min_samples_leaf=l,
            class_weight=weight,random_state=42,n_jobs=-1) for l in [4,10]],
      'SVM':[SVC(C=c,kernel='rbf',class_weight=weight,cache_size=512,random_state=42) for c in [1,3]],
      'MLP':[MLPClassifier(hidden_layer_sizes=h,alpha=a,max_iter=300,early_stopping=True,
            validation_fraction=.15,n_iter_no_change=15,random_state=42)
            for h,a in [((64,32),.01),((32,),.1)]],
      'Decision Tree':[DecisionTreeClassifier(max_depth=d,min_samples_leaf=l,
            class_weight=weight,random_state=42) for d,l in [(5,5),(8,20)]]}
    return choices[name]

def configurations(name):
    return [{'engineered':eng,'ratio':ratio,'variant':v}
            for eng,ratio in [(False,None),(True,None),(True,.5),(True,.75)] for v in [0,1]]

def build_pipeline(name,config):
    nums=NUMS+(EXTRAS if config['engineered'] else [])
    n=len(nums)
    # Scale numerical inputs before SMOTENC so durations do not dominate distances.
    pre=ColumnTransformer([
      ('numeric',SkPipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
      ('categorical',SkPipeline([('impute',SimpleImputer(strategy='most_frequent')),
          ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),CATS)])
    sampler='passthrough' if config['ratio'] is None else SMOTENC(
        categorical_features=list(range(n,n+len(CATS))),sampling_strategy=config['ratio'],random_state=42)
    post=ColumnTransformer([('numeric','passthrough',list(range(n))),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(range(n,n+len(CATS))))])
    return Pipeline([('features',FeatureEngineer(config['engineered'])),('preprocess',pre),
        ('sampling',sampler),('encode',post),('model',clone(variants(name,config['ratio'] is not None)[config['variant']]))])

def scores(model,X,name):
    return model.decision_function(X) if name=='SVM' else model.predict_proba(X)[:,1]
