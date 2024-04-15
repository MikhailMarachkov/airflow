# <YOUR_IMPORTS>
import json
import pandas as pd
import dill
import pandas as pd
import glob
import os
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '..')


def predict():
    # <YOUR_CODE>
    file_path = f'{path}/data/models/cars_pipe.pkl'
    with open(file_path, 'rb') as f:
        model = dill.load(f)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for file in glob.glob(f'{path}/data/test/*.json'):
        with open(file) as fid:
            form = json.load(fid)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            temp = {'car_id': df.id, 'pred': y}
            df_temp = pd.DataFrame(temp)
            df_pred = pd.concat([df_pred, df_temp], axis=0) 

    df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()

