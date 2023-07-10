from flask import Flask , request,jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pd.read_pickle('model2.sav')
df = pd.read_pickle('columns.csv')

@app.route("/",methods = ['POST'])
def prediction():
    data = request.get_json(force=True)
    state_name = data['state']
    season = data['season']
    crop = data['crop']
    area = float(data['area'])
    df2 = pd.DataFrame(0,index = range(1), columns=df.columns)
    df2['State_Name_'+state_name] = 1
    df2['Season_'+season] = 1
    df2['Crop_'+crop] = 1
    df2['Area'] = area
    print(df2.shape)
    Production = model.predict(df2)
    return (str(Production*area))

if __name__ == '__main__':
    app.run(port=5000, debug=True)