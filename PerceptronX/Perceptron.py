import numpy as np
import pandas as pd
import warnings
import time
from colorama import Fore
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error,root_mean_squared_error,mean_squared_error,accuracy_score,precision_score,recall_score,f1_score

class Perceptron():
    def __init__(self, learning_rate=0.001, validation_split=0.2, scaling='none', is_scaled=False, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.scaling = scaling
        self.is_scaled = is_scaled
        self.tolerance = tolerance
    
    def fit(self,X,y):

        #Scaled Data Check
        if self.scaling=='none' and self.is_scaled==False:
            for col in X.columns:
                col_data = X[col]
                if  is_standard_scaled(col_data)==False and is_minmax_scaled(col_data)==False:
                    raise RuntimeError(
                        f"Dataset column '{col}' must be either standard scaled or min-max scaled."
                    )

        #Scaling
        if self.scaling=='minmax':
            for i in X.columns:
                X.loc[:,i]=m_scaler_fn(X.loc[:,i])

        elif self.scaling=='standard':
            for i in X.columns:
                X.loc[:,i]=s_scaler_fn(X.loc[:,i])



        #Validation Split
        if self.validation_split>0:
            X_rows=X.shape[0]
            split=int(X_rows*(self.validation_split))
            X_validation_set=X.iloc[:split,:]
            X=X.iloc[split:,:]
        
            y_rows=y.shape[0]
            y_validation_set=y.iloc[:split]
            y=y.iloc[split:]
        


        # Data_type Check
        unique=y.nunique()
        if unique==2:
            self.data_type='binary'
        elif unique<20 and unique>2:
            self.data_type='multi'
            print(Fore.RED+"\033[1mWarning:\033[0mSingle perceptron is not suitable for multi-class classification. This function is for learning/demo purposes only and may give poor accuracy.\n"+Fore.RESET)
        else:
            self.data_type='linear'

        
        #Weight and Bias Innitialization
        classes=y.nunique()
        features=X.shape[1]
        if self.data_type=='multi':
            self.weights=np.random.randn(features,classes)
            self.bias=np.zeros(classes)
        else:
            self.weights=np.zeros(features)
            self.bias=0

        #Converting to numpy
        X=X.values
        y=y.values

        print(Fore.BLUE+"****<<<<<<< Training the PerceptronX >>>>>>>****\n"+Fore.RESET)
        
        #Gradient Descent Optimisation
        initial_loss=1e5
        iteration=0
        itering=0
        i=0
        for _ in range(2500000):
            self.weights,self.bias=gradient_descent_optimisation(X,y,self.weights,self.bias,self.data_type,self.learning_rate)
            loss=cost_fn(X,y,self.weights,self.bias,self.data_type)
            itering=itering+1
            
            if (itering-iteration)==50000 and self.tolerance==1e-9 and i<15:
                print(f'>> Iteration:{itering} and Loss:{loss}')
                iteration=itering
                i=i+1
                time.sleep(0.2)
                
            elif (itering-iteration)==10000 and self.tolerance==1e-8 and i<15:
                print(f'>> Iteration:{itering} and Loss:{loss}')
                iteration=itering
                i=i+1
                time.sleep(0.2)
                
            elif (itering-iteration)==2500 and self.tolerance==1e-7 and i<15:
                print(f'>> Iteration:{itering} and Loss:{loss}')
                iteration=itering
                i=i+1
                time.sleep(0.2)
                
            elif (itering-iteration)==500 and self.tolerance==1e-6 and i<15:
                print(f'>> Iteration:{itering} and Loss:{loss}')
                iteration=itering
                i=i+1
                time.sleep(0.2)
                
            elif (itering-iteration)==100 and self.tolerance==1e-5 and i<15:
                print(f'>> Iteration:{itering} and Loss:{loss}')
                iteration=itering
                i=i+1
                time.sleep(0.2)
                
            
            if initial_loss-loss<self.tolerance:
                break
            initial_loss=loss
        print(f'>> Iteration:{itering} and Loss:{loss}')

        # Validation_score
        if self.validation_split>0:
            if self.data_type=='linear':
                score=validation_test(self.weights,self.bias,X_validation_set,y_validation_set,self.data_type)
                print(Fore.GREEN+f'\nValidation Test: MSE:: {score}'+Fore.RESET)
            else:
                if self.data_type=='binary':
                    score=validation_test(self.weights,self.bias,X_validation_set,y_validation_set,self.data_type)
                    print(Fore.GREEN+f'\nValidation Test: Accuracy Score:: {score}'+Fore.RESET)
                else:
                    score=validation_test(self.weights,self.bias,X_validation_set,y_validation_set,self.data_type)
                    print(Fore.RED+f'\nValidation Test: Accuracy Score:: {score}'+Fore.RESET)

        if not self.data_type=='multi':
            weights_list=self.weights.tolist()
        else:
            weights_list=self.weights
        bias_list=self.bias.tolist()
        print(f'\nBias: {bias_list}')
        print(f'Weights: {weights_list}')
        
        
        return self.weights,self.bias

    
    def predict(self,X):
        #Scaled Data Check
        if self.scaling=='none' and self.is_scaled==False:
            for col in X.columns:
                col_data = X[col]
                if  is_standard_scaled(col_data)==False and is_minmax_scaled(col_data)==False:
                    raise RuntimeError(
                        f"Dataset column '{col}' must be either standard scaled or min-max scaled."
                    )
        
        #Scaling the Data
        if self.scaling=='minmax':
            for i in X.columns:
                X.loc[:,i]=m_scaler_fn(X.loc[:,i])

        elif self.scaling=='standard':
            for i in X.columns:
                X.loc[:,i]=s_scaler_fn(X.loc[:,i])


        
        
        if not hasattr(self, 'weights') or not hasattr(self, 'bias'):
            raise ValueError("Model must be trained using fit() before calling predict()")
        if not self.data_type=='multi':
            X=X.values
            self.predicted =predict(self.weights, self.bias, X, self.data_type)
        else:
            logits = np.dot(X, self.weights) + self.bias
            probs = stable_softmax(logits)
            return np.argmax(probs, axis=1)
            
        return self.predicted



    def score(self,X,y,metrics):

        #Scaled Data Check
        if self.scaling=='none' and self.is_scaled==False:
            for col in X.columns:
                col_data = X[col]
                if  is_standard_scaled(col_data)==False and is_minmax_scaled(col_data)==False:
                    raise RuntimeError(
                        f"Dataset column '{col}' must be either standard scaled or min-max scaled."
                    )
        
        #Scaling the Data
        if self.scaling=='minmax':
            for i in X.columns:
                X.loc[:,i]=m_scaler_fn(X.loc[:,i])

        elif self.scaling=='standard':
            for i in X.columns:
                X.loc[:,i]=s_scaler_fn(X.loc[:,i])

        

        selection=metrics
        X=X.values
        if self.data_type=='linear':
            if selection=='mse':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=mean_squared_error(y_pred,y)

            elif selection=='rmse':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=root_mean_squared_error(y_pred,y)

            elif selection=='rmsle':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=root_mean_squared_log_error(y_pred,y)

        elif self.data_type=='binary':
            if selection=='accuracy':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=accuracy_score(y_pred,y)

            elif selection=='precision':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=precision_score(y, y_pred, average='binary')

            elif selection=='recall':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=recall_score(y, y_pred, average='binary')

            elif selection=='f1':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=f1_score(y, y_pred, average='binary')

        else:
            if selection=='accuracy':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=accuracy_score(y_pred,y)

            elif selection=='precision':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=precision_score(y, y_pred, average='weighted')

            elif selection=='recall':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=recall_score(y, y_pred, average='weighted')

            elif selection=='f1':
                y_pred=predict(self.weights,self.bias,X,self.data_type)
                result=f1_score(y, y_pred, average='weighted')

        return result


            



def gradient_descent_optimisation(X,y,weights,bias,data_type,learning_rate):
    n=X.shape[0]
    if not data_type=='multi':
        if data_type=='linear':
            y_pred=predict_for_gradient_descent(weights,bias,X,data_type)
            gradient_weights=(1/n)*X.T.dot(y_pred-y)
            gradient_bias=(1/n)*np.sum((y_pred-y))
        else:
            y_pred=predict_for_gradient_descent(weights,bias,X,data_type)
            gradient_weights=(1/n)*X.T.dot(y_pred-y)
            gradient_bias=(1/n)*np.sum((y_pred-y))

        
        new_weights=weights-(learning_rate*(gradient_weights))
        new_bias=bias-(learning_rate*(gradient_bias))

        return new_weights,new_bias

    else:
        num_classes = weights.shape[1]
        y_encoded = one_hot_encode(y, num_classes)

        logits = X.dot(weights) + bias
        probs = stable_softmax(logits)

        error = probs - y_encoded
        grad_weights = np.dot(X.T, error) / n
        grad_bias = np.sum(error, axis=0) / n

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        return weights, bias


def predict_for_gradient_descent(weights,bias,X,data_type):
    if data_type=='binary':
        pred=np.dot(X,weights)+bias
        pred=sigmoid_fn(pred)
        return pred
    elif data_type=='linear':
        pred=np.dot(X,weights)+bias
        return pred
        

def predict(weights,bias,X,data_type):
    if data_type=='binary':
        pred=np.dot(X,weights)+bias
        pred=sigmoid_fn(pred)
        predicted=[]
        for i in range(pred.shape[0]):
            if pred[i]>0.5:
                predicted.append(1)
            else:
                predicted.append(0)
        pred=np.array(predicted)
        return pred
    elif data_type=='linear':
        pred=np.dot(X,weights)+bias
        return pred
        
    else:
        X_array=X
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
            
        z = X_array.dot(weights) + bias
        probs = stable_softmax(z)
        predictions = np.argmax(probs, axis=1)
        return predictions
        

def sigmoid_fn(value):
    result=1/(1+np.exp(-value))
    return result

def cost_fn(X,y,weights,bias,data_type):
    n=X.shape[0]
    if data_type=='binary':
        y_pred=predict_for_gradient_descent(weights,bias,X,data_type)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        result=-((1/n)*np.sum(((y)*np.log(y_pred)+(1-y)*np.log(1-y_pred))))

    elif data_type=='linear':
        y_pred=predict_for_gradient_descent(weights,bias,X,data_type)
        result=(1/n)*np.sum(np.power((y-y_pred),2))

    else:
        logits = X.dot(weights) + bias
        probs = stable_softmax(logits)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        y_encoded = one_hot_encode(y, weights.shape[1])
        return -np.mean(np.sum(y_encoded * np.log(probs), axis=1))

    return result

def one_hot_encode(y, num_classes):
    y=y.astype('int')
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def s_scaler_fn(data):
    data=data.values
    output=[]
    for i in data:
        scaled=(i-np.mean((data)))/np.std((data))
        output.append(scaled)
    return output

def m_scaler_fn(data):
    data=data.values
    output=[]
    for i in data:
        scaled=(i-np.min(data))/(np.max(data)-np.min(data))
        output.append(scaled)
    return output

def is_standard_scaled(X, tol_mean=1, tol_std=2):
    arr = np.array(X)
    means = arr.mean()
    stds = arr.std()
    check = (means < tol_mean and stds < tol_std)
    return check

def is_minmax_scaled(X, tol_min=-1.5, tol_max=1.5):
    arr = np.array(X)
    mins = arr.min()
    maxs = arr.max()
    check = (mins > tol_min and maxs < tol_max) or (mins==0 and maxs==1)
    return check

def stable_softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def validation_test(weights,bias,X,y,data_type):
    if data_type=='linear':
        X=X.values
        y_pred=predict(weights,bias,X,data_type)
        score=mean_squared_error(y_pred,y)
        return score

    elif data_type=='binary':
        X=X.values 
        y_pred=predict(weights,bias,X,data_type)
        score=accuracy_score(y_pred,y)
        return score

    else:
        logits = np.dot(X, weights) + bias
        probs = stable_softmax(logits)
        y_pred=np.argmax(probs, axis=1)
        score=accuracy_score(y_pred,y)
        return score
