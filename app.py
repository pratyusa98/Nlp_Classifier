from flask import Flask,render_template,request
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords



app = Flask(__name__)

#spam
filename = 'spam_prediction.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
#gener
filename = 'gener_mnb.pkl'
classifier_mnb = pickle.load(open(filename, 'rb'))
cv_tfidf = pickle.load(open('tfidf-transform.pkl','rb'))


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/gener')
def gener():
	return render_template('gener.html')

#spam prediction
@app.route('/spampredict',methods=['POST'])
def predict_spam():
	if request.method == 'POST':
		msg = request.form['spam']
		ps = PorterStemmer()
		a = re.sub('[^a-zA-Z]',' ',msg)
		a = a.lower()
		a = a.split()
		a = [ps.stem(word) for word in a if word not in set(stopwords.words('english'))]
		a = ' '.join(a)  
		data = [a]

		vect = cv.transform(data)
		my_prediction = classifier.predict(vect)

		return render_template('index.html',pred = my_prediction,msg = msg)

#gener Prediction
@app.route('/generpredict',methods=['POST'])
def predict_gener():
	if request.method == 'POST':
		story = request.form['story']
		ps = PorterStemmer()
		a = re.sub('[^a-zA-Z]',' ',story)
		a = a.lower()
		a = a.split()
		a = [ps.stem(word) for word in a if word not in set(stopwords.words('english'))]
		a = ' '.join(a)  
		data = [a]

		vect = cv_tfidf.transform(data)
		prediction = classifier_mnb.predict(vect)

		return render_template('gener.html',pred = "This Belongs To {} Gener".format(prediction[0]),msg = story)

if __name__=='__main__':
	app.run(debug=True)