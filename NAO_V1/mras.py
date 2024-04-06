import torch
import numpy as np
import torch.nn.functional as F
from functools import cmp_to_key
from torch.distributions.multivariate_normal import MultivariateNormal

d = 2
T = 2
def H(arr,get_performance):
    e_x = arr.reshape(T,d)
    predict_value = get_performance(e_x.unsqueeze(0))
    return predict_value

def S(x):
    r = 1
    return torch.exp(r*x).cuda()

class pdf:
    def __init__(self):
        self.mu = torch.zeros(T*d).cuda()
        self.sigma = 10*torch.eye(T*d).cuda()
        self.PDF = MultivariateNormal(self.mu, self.sigma)

    def f(self, x):
        return self.PDF.log_prob(x)

    def update(self, newMu, newSigma):
        self.mu = newMu.cuda()
        self.sigma = newSigma.cuda()
        for i in range(T*d):
          for j in range(T*d):
            if i!=j:
              self.sigma[i][j] = 0
            # self.sigma[i][j] = abs(self.sigma[i][j]) #New added line ...might not require
        self.PDF = MultivariateNormal(self.mu, self.sigma)

def I(x, gamma):
    return torch.where(x>=gamma, x, 0)

def calculate_expection(X, func, k, pdf_function, gamma):
    sum = func(X[0],k,pdf_function,gamma)
    count = torch.tensor(1).cuda()
    for i,x in enumerate(X):
        if i==0:
           continue
        sum += func(x, k, pdf_function, gamma)
        count += 1
    return sum / count

def update_mu(X, gamma, k, pdf_function, get_performance):
    k = 1
    def func_numerator(x,k,pdf_function, gamma):
      return (S(H(x,get_performance))**k) * I(H(x,get_performance),gamma) * x
    def func_denominator(x,k,pdf_function, gamma):
      return ((S(H(x,get_performance))**k) * I(H(x,get_performance),gamma))
    return torch.tensor(calculate_expection(X, func_numerator, k, pdf_function, gamma) / calculate_expection(X, func_denominator, k, pdf_function, gamma)).cuda()

def update_sigma(X, gamma, k, pdf_function, get_performance):
    k = 1
    def func_numerator(x,k,pdf_function,gamma):
      matrix1 = torch.unsqueeze(x-pdf_function.mu,dim=0).cuda()
      matrix2 = torch.reshape(matrix1,(T*d,1)).cuda()
      return (S(H(x,get_performance))**k) * I(H(x,get_performance),gamma) *(torch.matmul(matrix2,matrix1))
    def func_denominator(x,k,pdf_function,gamma):
      return ((S(H(x,get_performance))**k) * I(H(x,get_performance),gamma))
    arr = calculate_expection(X, func_numerator, k, pdf_function, gamma) / calculate_expection(X, func_denominator, k, pdf_function, gamma)
    return torch.tensor(arr).cuda()

def return_random_iids(N, prop_df, get_float=0):
    arr = []
    for i in range(N):
      arr.append(prop_df.PDF.sample().tolist())
    return torch.tensor(arr).cuda()

low = torch.tensor([-2, -2]).cuda()
high = torch.tensor([5, 5]).cuda()
N = 200
quantile = 0.05
K = 50
gamma = torch.tensor(-1000).cuda()
epsilon = torch.tensor(0.001).cuda()
alpha = 1

def compare(X, Y):
    if X[0] < Y[0]:
        return -1
    return 1

def mras(arch,predict_lambda, get_performance):
    global N
    global gamma
    global epsilon
    global alpha
    global quantile
    global d
    global T

    T = len(arch[0])
    d = len(arch[0][0])
    N = len(arch)

    randomIids = torch.tensor([]).cuda()
    for architecture in arch:
       flattened_arch = torch.flatten(architecture)
       randomIids = torch.cat((randomIids,flattened_arch.unsqueeze(0)), dim=0)

    # alpha = predict_lambda
    prop_df = pdf()
    for k in range(1, K + 1):
        if randomIids == None:
            N = 100
            randomIids = return_random_iids(N, prop_df)
        # print("Random IIDs: ")
        # for i,x in enumerate(randomIids):
        #    print("IID ", i, ": ", x)
        # randomIids = randomIids.cuda()
        HValues = [H(i,get_performance) for i in randomIids]
        HValues_X = [[iid,randomIids[i]] for i,iid in enumerate(HValues)]
        sortedHValues = sorted(HValues)
        sortedXValues = sorted(HValues_X, key=cmp_to_key(compare))
        XArray = [temp_arr[1] for temp_arr in sortedXValues]
        quantileIndex = int((1 - quantile) * N)
        XArray = XArray[quantileIndex:]
        currGamma = sortedHValues[quantileIndex]
        if k == 1 or currGamma >= gamma + (epsilon / 2):
            gamma = currGamma
            ind = HValues.index(gamma)
        else:
            gamma = currGamma
            N = int(alpha * N)
        prop_df.update(update_mu(XArray, gamma, k, prop_df, get_performance), update_sigma(XArray, gamma, k, prop_df, get_performance))
        randomIids = None
        
    print("Mean (mras.py): ", prop_df.mu)
    print("Sigma (mras.py): ", prop_df.sigma)
    print("Performance of Mean (mras.py): ", H(prop_df.mu, get_performance))
    new_flattened_encoder_outputs = return_random_iids(len(arch), prop_df)
    new_encoder_outputs = torch.tensor([]).cuda()
    for architecture in new_flattened_encoder_outputs:
       proper_architecture = architecture.reshape(T,d)
       new_encoder_outputs = torch.cat((new_encoder_outputs, proper_architecture.unsqueeze(0)), dim=0)
    return new_encoder_outputs
