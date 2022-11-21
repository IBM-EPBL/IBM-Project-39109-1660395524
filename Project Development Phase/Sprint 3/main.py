import numpy as np
import requests
from flask import Flask, render_template, request
from tensorflow.python.keras.models import load_model

app = Flask(__name__)

model = load_model("D:\Documents\ibm\fetch.tar.gz")

token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":"1gwmabI1uLfI58p348zk2GOm5l-4Gq-30YwWSKgGvsUY", "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


ent = []

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def home1():
    if request.method == "POST":
        data = request.form['data']
        # print(data)
        ent.append(data)
        
        inp = str(request.form['data'])
        inp = inp.split(',')
        for i in range(0, len(inp)):
            inp[i] = float(inp[i])
        inp = np.array(inp).reshape(1,-1)
        temp = list(inp)
        temp = temp[0].tolist()
        # print(temp)
        outpt = []
        n_steps = 10
        i=0
        while(i<1):
            if(len(temp) > 10):
                #some prblm
                inp = np.array(temp[1:])
                print("{} day input {}".format(i, inp))
                inp = inp.reshape(1,-1)
                inp = inp.reshape((1, n_steps, 1))
                yhat = model.predict(inp, verbose=0)
                temp.extend(yhat[0].tolist())
                temp = temp[1:]
                outpt.extend(yhat.tolist())
                i += 1
            else:
                inp = inp.reshape((1,len(temp),1)) #n_steps --> len(temp)
                yhat = model.predict(inp, verbose=0)
                temp.extend(yhat[0].tolist())
                outpt.extend(yhat.tolist())
                i+=1
            

        return render_template("predict_opt.html", entries = outpt)
    else:
        return render_template("predict_opt.html")

@app.route('/market')
def home2():
    return render_template("market.html")


if __name__ == '__main__':
    app.run(debug=True)