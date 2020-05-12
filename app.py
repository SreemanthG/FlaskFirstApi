# app.py
from flask import Flask           # import flask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
app = Flask(__name__)             # create an app instance

@app.route("/")                   # at the end point /
def hello():
    #-------------
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    #----------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    #--------------
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(X_train,y_train)

    #-------------
    filename = 'finalized_model'
    pickle.dump(regressor, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))

    #---------------
    new_input=np.array([12])
    new_input1=new_input.reshape(1,-1)
    print(regressor.predict(new_input1))
    predictedoutput= regressor.predict(new_input1)
    #--------------
    result = loaded_model.score(X_test, y_test)
    print("Accuracy ",result)
    #----------
    # plt.scatter(X_test,y_test,color='red')
    # plt.plot(X_train,regressor.predict(X_train),color='blue')
    # plt.title('Salary vs Experience(Test Set)')
    # plt.xlabel('Years of Experience')
    # plt.ylabel('Salary')
    # plt.show()
    freqs = {
    'predictedoutput': predictedoutput[0],
    'accuracy': result,
    }
    return jsonify(freqs)         # which returns "hello world"
if __name__ == "__main__":        # on running python app.py
    app.run()                     # run the flask app