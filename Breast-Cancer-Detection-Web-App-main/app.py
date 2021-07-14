import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model2 = pickle.load(open('classify.pkl', 'rb'))
model = pickle.load(open('modell.pkl', 'rb'))
@app.route('/')
def home():
  return render_template('indexN.html')

@app.route('/predict',methods=['POST'])
def predict():

  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]
  # feature_value = [np.concatenate((features_value,np.array([0.07253, 0.4426, 1.169, 3.176, 34.37, 0.005273, 0.02329, 0.01405, 0.012440000000000001, 
  # 0.01816, 0.0032990000000000003, 15.05, 24.37, 99.31, 674.7, 0.1456, 0.2961, 0.1246, 0.1096, 0.2582, 0.08893])))]

  # features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
  #      'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
  #      'bland_chromatin', 'normal_nucleoli', 'mitoses','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','']

  features_name=['radius_mean', 'texture_mean', 'perimeter_mean',
  'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
  'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
  'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
  'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
  'fractal_dimension_se', 'radius_worst', 'texture_worst',
  'perimeter_worst', 'area_worst', 'smoothness_worst',
  'compactness_worst', 'concavity_worst', 'concave points_worst',
  'symmetry_worst', 'fractal_dimension_worst']
  df = pd.DataFrame(features_value, columns=features_name)
  # print(feature_value)
  # print(df)
  print (features_value)
  # print (feature_value)
  output = model2.predict(df)

  if output == 1:
      res_val = "Breast cancer"
  else:
      res_val = "no Breast cancer"



  return render_template('indexN.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run(debug=True)

