from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open("spam-sms-model.pkl","rb"))
cv=pickle.load(open("cv-transform.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = model.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__=='__main__':
    app.run(debug=True)