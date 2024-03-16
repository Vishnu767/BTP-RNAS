import torch
import numpy as np
import torch.nn.functional as F
from functools import cmp_to_key
from torch.distributions.multivariate_normal import MultivariateNormal

d = 2
def H(arr,forward):
    encoder_outputs, encoder_hidden, arch_emb, predict_value = forward(arr)
    return S(predict_value)

def S(x):
    r = 1
    return torch.exp(r*x)

def H2(arr):
    d = len(arr)
    val = 0
    prod = 1
    for i in range(d):
        val += (arr[i]**2) / 4000
        prod *= np.cos(arr[i] / np.sqrt(i + 1))
    return -1 * (val - prod + 1)

class pdf:
    def __init__(self):
        self.mu = torch.zeros(d)
        self.sigma = 10*torch.eye(d)
        self.PDF = MultivariateNormal(self.mu, self.sigma)

    def f(self, x):
        return self.PDF.log_prob(x)

    def update(self, newMu, newSigma):
        self.mu = newMu
        self.sigma = newSigma
        for i in range(d):
          for j in range(d):
            if i!=j:
              self.sigma[i][j] = 0
            # self.sigma[i][j] = abs(self.sigma[i][j]) #New added line ...might not require
        self.PDF = MultivariateNormal(self.mu, self.sigma)

def I(x, gamma):
    return torch.where(x>=gamma, x, 0)

def calculate_expection(X, func, k, pdf_function, gamma):
    sum= 0
    count =0
    for x in X:
        sum += func(x, k, pdf_function, gamma)
        count +=1
    return sum / count

def update_mu(X, gamma, k, pdf_function, forward):
    k = 1
    def func_numerator(x,k,pdf_function, gamma):
      return (S(H(x,forward))**k) * I(H(x,forward),gamma) * x
    def func_denominator(x,k,pdf_function, gamma):
      return ((S(H(x,forward))**k) * I(H(x,forward),gamma))
    return calculate_expection(X, func_numerator, k, pdf_function, gamma) / calculate_expection(X, func_denominator, k, pdf_function, gamma)

def update_sigma(X, gamma, k, pdf_function, forward):
    k = 1
    def func_numerator(x,k,pdf_function,gamma):
      matrix1 = torch.unsqueeze(x-pdf_function.mu,dim=0)
      matrix2 = torch.reshape(matrix1,(d,1))
      return (S(H(x,forward))**k) * I(H(x,forward),gamma) *(torch.matmul(matrix2,matrix1))
    def func_denominator(x,k,pdf_function,gamma):
      return ((S(H(x,forward))**k) * I(H(x,forward),gamma))
    arr = calculate_expection(X, func_numerator, k, pdf_function, gamma) / calculate_expection(X, func_denominator, k, pdf_function, gamma)
    return arr

def return_random_iids2(low, high, N):
    randomIidsX = torch.tensor(np.random.uniform(low[0], high[0], N))
    randomIidsY = torch.tensor(np.random.uniform(low[1], high[1], N))
    randomIids = [torch.tensor([randomIidsX[i].item(), randomIidsY[i].item()]) for i in range(N)]
    return randomIids

def return_random_iids(N, prop_df):
    arr = []
    for i in range(N):
      arr.append(prop_df.PDF.sample().tolist())
    return torch.tensor(arr)
    # return torch.tensor(np.random.multivariate_normal(pdf_function.mu.numpy(), pdf_function.sigma.numpy(), N))


low = torch.tensor([-2, -2])
high = torch.tensor([5, 5])
N = 100
quantile = 0.08
K = 5
gamma = -1000
epsilon = 0.001
alpha = 1.001

def compare(X, Y):
    if X[0] < Y[0]:
        return -1
    return 1

def mras(arch,predict_lambda, forward):
    global N
    global gamma
    global epsilon
    global alpha
    global quantile
    global d

    print("Number of architectures: ", len(arch))
    print("Length of each architecture: ", len(arch[0]))
    randomIids = arch
    alpha = predict_lambda
    N = len(arch)
    # Set the dimension too
    print("Arch: ", arch)
    d = len(arch[0])
    prop_df = pdf()
    print("Dimension of Mean: ", len(pdf.mu))
    print("Dimension - d: ", d)
    for k in range(1, K + 1):
        if randomIids == None:
            N = 100
            randomIids = return_random_iids(N, prop_df)
        HValues = [H(i,forward) for i in randomIids]
        HValues_X = [[H(i,forward), i] for i in randomIids]
        sortedHValues = sorted(HValues)
        sortedXValues = sorted(HValues_X, key=cmp_to_key(compare))
        XArray = [temp_arr[1] for temp_arr in sortedXValues]
        quantileIndex = int((1 - quantile) * N)
        currGamma = sortedHValues[quantileIndex] #bug
        if k == 1 or currGamma >= gamma + (epsilon / 2):
            gamma = currGamma
            ind = HValues.index(gamma)
        else:
            gamma = currGamma
            N = int(alpha * N)
        prop_df.update(update_mu(XArray, gamma, k, prop_df, forward), update_sigma(XArray, gamma, k, prop_df, forward))
        print(prop_df.mu, prop_df.sigma)
        print(H(prop_df.mu, forward))
        randomIids = None
    
    return return_random_iids(len(arch), prop_df)
