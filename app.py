from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

#load pickle
feature = pickle.load(open('feature.pkl', 'rb'))
model = pickle.load(open('model_Bayes.pkl', 'rb'))


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
  
  text = request.form['fname']
  #mengubah huruf pada kalimat 
  txt = text.lower()
  txt = re.sub("[^a-zA-Z]", ' ', txt)

  #ekstrak = feature.tranform(txt)

  if text != '':
    prediksi = model.predict(feature.transform([txt]).toarray())
    #Indikasi result 
    if prediksi ==0 :
      res_pred = "Positif"
    else :
      res_pred = "Negatif"
  else:
    res_pred= '-'

  return render_template('index.html', hasil=res_pred)

if __name__ == "__main__":
  app.run()