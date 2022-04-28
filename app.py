from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

model = pickle.load(open("model/model_RFRegr_FP1.pkl", "rb"))

app = Flask(__name__, template_folder="templates")


@app.route('/')
def main():
    return render_template('main.html')

# Redirecting the API to predict the result


@app.route("/predict", methods=['POST'])
def predict():
    """
    For Rendering result on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template("main.html", prediction_text="Prediksi Harga Taksi Adalah : {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)
