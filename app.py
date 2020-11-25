from flask import Flask,render_template,url_for,request
#import pandas as pd 
import pickle
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#import joblib
import numpy as np
#import pandas as pd
#import pickle
from flask import jsonify
#import re
#from sklearn.naive_bayes import GaussianNB
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))



@app.route('/')
def home():
	return render_template('home.html')
                        
                        # @app.route('/predict',methods=['GET','POST'])
                        # def predict():
                        
                        #     if request.method == 'POST':
                        #        m= print(request.form.values())
                        #        message = request.form['corpus']                         
                        #        data = [message]
                        #         #vect = cv.fit_transform(data).toarray()                       
                        #        #vect = cv.transform(data).toarray()
                        #        #my_prediction = nb.predict(vect)
                        #     #return render_template('result.html',prediction = my_prediction)


@app.route('/predict',methods=['POST'])
def predict():
    # # Get the data from the POST request.
    # data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['exp'])]])
    # return jsonify(prediction)
   ## feature=[str(x) for x in request.form.values()] how to  get  input 
    # final_feature=[np.array(feature)]
    # prediction=model.predict(final_feature)
    from naive_byes import predict_result
    prediction=predict_result(request.form['search'])
    return render_template('home.html', prediction_text='sentence is {}'.format(prediction))



if __name__ == '__main__':
	app.run(port=5000, debug=True)
    