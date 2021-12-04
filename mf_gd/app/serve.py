import flask
import os, sys 
from train import train 
from predict import predict

app = flask.Flask(__name__)



@app.route("/", methods=["GET"])
def hello():
    return 'Hello - I am Matrix Factorizer and I am alive!'
     

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    
    # some health check here ...
    health = True

    status = 200 if health else 404
    return flask.Response(response="Ping!\n", status=status, mimetype="application/json")


@app.route("/train", methods=["GET"])
def train_endpoint():
    resp = train()
    status = 200 if resp==0 else 404
    return flask.Response(response="Training completed successfully.", status=status, mimetype="application/json")


@app.route("/predict", methods=["GET"])
def predict_endpoint():
    resp = predict()
    status = 200 if resp==0 else 404
    return flask.Response(response="Predictions completed successfully.", status=status, mimetype="application/json")





if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=3000, debug=True)
    
	