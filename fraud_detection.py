from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import losses
from keras import optimizers
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import keras.utils.vis_utils as vis
sns.set(color_codes=True)
np.set_printoptions(suppress=True)




sc = StandardScaler()
# Preprocessing del dataset
transaction = pd.read_csv('creditcard.csv')

before_scaling = transaction['Amount'].describe()
transaction['Amount'] = sc.fit_transform(transaction['Amount'].values.reshape(-1,1))
after_scaling = transaction['Amount'].describe()

print(before_scaling)
print(after_scaling)

sns.distplot(transaction.Amount)


fraud = transaction[transaction['Class'] == 1]
normal = transaction[transaction['Class'] == 0]
fraud_test_set = fraud.iloc[:,1:].values
normal = normal.iloc[:,1:-1].values
xTrain, xTest = train_test_split(normal, test_size=0.2)
xTest1, xTest2 = train_test_split(xTest, test_size = 0.01)
normal_test_set = np.zeros((xTest2.shape[0], xTest2.shape[1]+1))
normal_test_set[:, :-1] = xTest2
test_set = np.append(fraud_test_set, normal_test_set, axis = 0)


#Creazione della rete neurale
def create_model1():
    if not os.path.exists("saved_model/model1"):
        print ("Creazione del modello in corso")
        os.mkdir("saved_model/model1")
        trainDimensions = xTrain.shape[1]
        encodingDim = int(trainDimensions/2)
        data = Input(shape = (trainDimensions,))
        encoder = Dense(int(encodingDim), activation = "relu")(data)
        encoder = Dense(int(encodingDim/2), activation = "tanh")(encoder)
        decoder = Dense(int(encodingDim), activation = "relu")(encoder)
        decoder = Dense(trainDimensions, activation = "tanh")(decoder)
        autoencoder = Model(inputs = data, outputs = decoder)
        # Compilazione e addestramento
        autoencoder.compile(optimizer = "Adam", loss = losses.mean_absolute_error)
        history = autoencoder.fit(x = xTrain, y = xTrain, batch_size = 32, epochs = 100, validation_data = (xTest1, xTest1))
        autoencoder.save("saved_model/model1/frauds_model.h5")
        autoencoder.save_weights("saved_model/model1/frauds_weights.h5")    
        with open("saved_model/model1/history.dmp", "wb") as f:
                pickle.dump(history, f)
            
    else:
        autoencoder = load_model("saved_model/model1/frauds_model.h5")
        autoencoder.load_weights("saved_model/model1/frauds_weights.h5")
        with open("saved_model/model1/history.dmp", "rb") as f:
                history = pickle.load(f)
    
    return autoencoder, history, np.array(history.history['val_loss']).mean()

def create_model2():
    if not os.path.exists("saved_model/model2"):
        print ("Creazione del secondo modello in corso")
        os.mkdir("saved_model/model2")
        trainDimensions = xTrain.shape[1]
        encodingDim = int(trainDimensions/2)
        data = Input(shape = (trainDimensions,))
        encoder = Dense(int(encodingDim), activation = "relu")(data)
        encoder = Dense(int(encodingDim/2), activation = "tanh")(encoder)
        encoder = Dense(int(encodingDim/4), activation = "relu")(encoder)
        decoder = Dense(int(encodingDim/2), activation = "tanh")(encoder)
        decoder = Dense(int(encodingDim), activation = "relu") (decoder)
        decoder = Dense(trainDimensions, activation = "tanh")(decoder)
        autoencoder = Model(inputs = data, outputs = decoder)
        # Compilazione e addestramento
        autoencoder.compile(optimizer = "Adam", loss = losses.mean_absolute_error)
        history = autoencoder.fit(x = xTrain, y = xTrain, batch_size = 32, epochs = 100, validation_data = (xTest1, xTest1))
        autoencoder.save("saved_model/model2/frauds_model.h5")
        autoencoder.save_weights("saved_model/model2/frauds_weights.h5")    
        with open("saved_model/model2/history.dmp", "wb") as f:
                pickle.dump(history, f)
            
    else:
        autoencoder = load_model("saved_model/model2/frauds_model.h5")
        autoencoder.load_weights("saved_model/model2/frauds_weights.h5")
        with open("saved_model/model2/history.dmp", "rb") as f:
                history = pickle.load(f)
    
    return autoencoder, history, np.array(history.history['val_loss']).mean()
    


def create_models():
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    
    return create_model1(), create_model2()

# Valutazione delle frodi e costruzione della matrice di confusione
def model_prediction(model, tol, mean_val_loss):
    true_positive = []
    false_positive = []
    true_negative = []
    false_negative = [] 
    for x in test_set:
        y = x[:-1]
        pred_x = model.predict(np.array([y,]))
        error_fraud = np.abs(y - pred_x).mean()
        # L'autoencoder non è riuscito a costruire bene l'input
        if  np.abs(error_fraud - mean_val_loss).mean() > (mean_val_loss - tol):
            if x[-1] == 1:
                true_positive.append(x)
            else:
                false_positive.append(x)
        else:
            if x[-1] == 0:
                true_negative.append(x)
            else:
                false_negative.append(x)
    tp = true_positive.__len__()
    fp = false_positive.__len__()
    fn = false_negative.__len__()
    tn = true_negative.__len__()
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    recall = tp/(tp + fn)
    precision = tp/(tp+fp)
    cm = np.array([tp, fp, fn, tn]).reshape(2, 2)
    return accuracy, precision, recall, cm

def evaluate_loss(history):    
    import matplotlib.pyplot as plt
    plt.title("Data reconstruction loss")
    plt.plot(history.history['loss'], label = "Training set loss")
    plt.plot(history.history['val_loss'], label = "Validation set loss")
    plt.legend()
    plt.show()
    plt.savefig("loss.png")

def compare_loss(history1, history2):
    import matplotlib.pyplot as plt
    #plt.title("Data reconstruction error")
    plt.plot(history1.history['loss'], label = "Training set loss model 1")
    plt.plot(history1.history['val_loss'], label = "Validation set loss model 1")
    plt.plot(history2.history['loss'], label = "Training set loss model 2")
    plt.plot(history2.history['val_loss'], label = "Validation set loss model 2")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("compareloss.png")

def evaluate_metrics(model, mean_val_loss):
    tols = np.arange(0, mean_val_loss, 0.01)
    accuracies = []
    precisions = []
    recalls = []
    for i in range(tols.shape[0]):    
        accuracy, precision, recall, cm = model_prediction(model, tols[i], mean_val_loss)
        print (accuracy, precision, recall)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
    import matplotlib.pyplot as plt
    #plt.title("Model evaluation metrics")
    plt.xlabel("Tollerance")
    plt.xticks(np.arange(0, mean_val_loss, 0.05))
    plt.ylabel("Value")
    #plt.plot(tols, accuracies, label = "Accuracies")
    plt.plot(tols, precisions, label = "Precisions")
    plt.plot(tols, recalls, label = "Recalls")
    plt.legend()
    #plt.show()
    plt.savefig("metrics.png")
    print(cm)
    
def draw_model(model):
    print(model.summary())
    vis.plot_model(model, to_file='model.png', show_shapes = True)
    

def plot_test_set():
    pie = plt.pie([xTest2.shape[0], fraud.shape[0]], radius = 1, explode = (0, 0.1), autopct = "%1.2f%%")
    plt.legend(pie[0], ["Normal", "Fraud"], bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure)
    plt.title("Classifier test set composition")
    plt.axis('equal')
    plt.show()
    
def find_std_deviation(model):
    errors = []
    not_fraudolent = []
    fraudolent = []
    for element in test_set:
        x = element[:-1]
        pred_x = model.predict(np.array([x,]), batch_size = 1)
        error = np.abs(x - pred_x).mean()
        if (element[-1] == 1):
            fraudolent.append(error)
        else:
            not_fraudolent.append(error)
    errors = np.array(not_fraudolent + fraudolent)
    not_fraudolent = np.array(not_fraudolent)
    fraudolent = np.array(fraudolent)
    plt.scatter(x = range(not_fraudolent.shape[0]), y = not_fraudolent, color = "green", s = 1.5, label = "Not fraudolent")
    plt.scatter(x = range(fraudolent.shape[0]), y = fraudolent, color = "red", s = 1.5, label = "Fraudolent")
    plt.legend()
    plt.show()
    plt.savefig("reconstruction_errors.png")
    return errors, fraudolent, not_fraudolent, errors.std(), fraudolent.std(), not_fraudolent.std()

    
    
model1, history1, mean_error1 = create_model1()
model2, history2, mean_error2 = create_model2()