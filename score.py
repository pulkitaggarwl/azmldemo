def init():
    from sklearn.externals import joblib
    from azureml.core.model import Model
	
    global model
    model_path = Model.get_model_path('mymodel4')
    model = joblib.load(model_path)

def run(input_df):
    import json
    pred = model.predict(input_df)
    return json.dumps(str(pred[0]))

def main():
  from azureml.api.schema.dataTypes import DataTypes
  from azureml.api.schema.sampleDefinition import SampleDefinition
  from azureml.api.realtime.services import generate_schema
  import pandas

  df = pandas.DataFrame(data=[[190, 60, 38]], columns=['height', 'width', 'shoe_size'])

  # Test the functions' output
  init()
  input1 = pandas.DataFrame([[190, 60, 38]])
  print("Result: " + run(input1))
  
  inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

  # Generate the service_schema.json
  #generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')
  print("Schema generated")

if __name__ == "__main__":
    main()