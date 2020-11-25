import numpy as np
import pandas as pd
import pickle
import re
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df=pd.read_excel(r'gali_01.xlsx')
ps=PorterStemmer()
corpus=[]

def clearning():                 #function used for cleaning text
    for i in range(len(df)):
        review=re.sub('[^a-zA-Z]',' ',df['COMMENTS'][i])
        review=review.lower()
        review=review.split()
        review=[ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        corpus.append(review)
clearning()

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer() 
X=cv.fit_transform(corpus).toarray()
y=df['RATINGS']
print(X)
y = y.astype('category')
m = y.dtype
'''
m=pd.get_dummies(y)
'''

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=0)
    
sc=StandardScaler()    

def fitting(X_train, X_test ):
    #print("hello")
    
    #sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)

fitting(X_train, X_test)


'''
#from sklearn.neighbors import KNeighborsClassifier
#classifier=KNeighborsClassifier(n_neighbors=5)

from sklearn import svm
classifier=svm.SVC(kernel='linear',gamma='auto',C=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
'''
'''
r="chutiya"
type(r)
m=float(r)
type(m)
'''
'''
y_predq=classifier.predict(float('chutiya'))
'''
'''
from sklearn.metrics import confusion_matrix 
confusion=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
'''
'''
##############CALCULATING MEMORY!!!########################
#custom_pred_value
nb = GaussianNB()
@profile
def naivefit():
    
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    global custom_pred_value

naivefit()
'''
nb = GaussianNB()
    
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)







#CHECKKING OUR MODEL!!!

# custom array 
custom_pred = np.array(['Discounts nhi hai', 'chutiya product', 'cheap rates', 'maa ki aakh','banchod'])
# preprocessing 
custom_pred_cv = cv.transform(custom_pred).toarray()    # preprocessing
custom_pred_sc = sc.transform(custom_pred_cv)           # preprocessing

# predict
custom_pred_value = nb.predict(custom_pred_sc)

# display
for i, s in enumerate(custom_pred):
    print(s + " : " + str(custom_pred_value[i]))




from sklearn.metrics import classification_report
report=classification_report( y_test, y_pred)

# #saving model to disk
# pickle.dump(nb,open('model.pkl','wb'))
# #loading model to compare the results
# model=pickle.load(open('model.pkl','rb'))


# export -> vectorzer, model


# val = input("enter value ")
# m=[val]               
# #val = valw.split()
# custo_pred_cv = cv.transform(m).toarray()    # preprocessing
# custo_pred_sc = sc.transform(custo_pred_cv)

# pred=nb.predict(custo_pred_sc)
# for i, s in enumerate(m):
#           print(s + " : " + str(pred[i]))

def predict_result(val):
        
    
    m=[val]               #since input is a string and we need a list
    #val = valw.split()
    custo_pred_cv = cv.transform(m).toarray()    # preprocessing
    custo_pred_sc = sc.transform(custo_pred_cv)
    pred=nb.predict(custo_pred_sc)
    return pred[0]
    # for i, s in enumerate(m):
    #     print(s + " : " + str(pred[i]))
print(predict_result('hello'))


# m=cv.fit(val)
# sc.fit(m)
# custom_cv_val=cv.transform(m).toarray()
# custom_sc_val=sc.transfrom(m).toarray()
# custommpred=nb.predict(custom_sc_val)

# prediction_cv=cv.transform(val)
# print(prediction_cv)
# prediction_sc=sc.transform(val)

# sample_pred=nb.predict(val)
# for i ,s in enumerate(val):
#     print(s+ " : " + str(val[i]))



# from sklearn.metrics import classification_report
# report=classification_report( y_test, y_pred)
