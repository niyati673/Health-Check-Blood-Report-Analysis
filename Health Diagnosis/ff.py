import pickle

with open('model_healthcheckpoints.pkl', 'rb') as file:
    contents = pickle.load(file)
    print(contents)
