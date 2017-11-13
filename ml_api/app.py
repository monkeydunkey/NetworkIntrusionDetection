from flask import Flask, jsonify, request
import os
import socket
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

def runRandomForest(receivedData):
    mdl = joblib.load("RandomForest.pkl")
    colOrder = dict(map(lambda x: (x[1], x[0]), enumerate(mdl.feature_cols)))
    test_x = map(lambda x: x[1], sorted(receivedData.items(), key=lambda x: colOrder[x[0]]))
    pred_y = mdl.predict(np.array(test_x).reshape(1, -1))[0]
    return pred_y

@app.route('/prediction', methods=['POST'])
def get_prediction():
    print 'Prediction Called'
    receivedData = request.json
    mdlFuncs = [('RandomForest', runRandomForest)]
    authResult = {}
    for mdlName, func in mdlFuncs:
        prediction = func(receivedData)
        authResult[mdlName] = prediction

    #data = filter(lambda x: x['userId'] == user_id, sampleData)
    return jsonify(authResult), 201

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>" \
           "use /prediction endpoint for getting the prediction results"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
