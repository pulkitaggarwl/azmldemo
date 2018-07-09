'''
Code sample is based from gender classification challenge for 'Learn Python for Data Science #1' 
by @Sirajology on https://youtu.be/T5pRlIbr6gg
'''

from sklearn.svm import SVC
import pickle

# height, width, shoe size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf = SVC()
clf = clf.fit(X, Y)

print('Predicted value:', clf.predict([[190, 70, 43]]))
print('Accuracy', clf.score(X,Y))

print('Export the model to model.pkl')
f = open('model.pkl', 'wb')
pickle.dump(clf, f)
f.close()

print('Import the model from model.pkl')
f2 = open('model.pkl', 'rb')
clf2 = pickle.load(f2)

X_new = [[154, 54, 35]]
print('New Sample:', X_new)
print('Predicted class:', clf2.predict(X_new))


# Import Run from azureml.core, 
# and get handle of current run for logging and history purposes
from azureml.core.run import Run
run = Run.get_submitted_run()

model_path = "model.pkl"

# Save model as part of the run history
with open(model_path, "wb") as file:
    from sklearn.externals import joblib
    joblib.dump(clf, file)
run.upload_file(model_path,  model_path)
os.remove(model_path)


