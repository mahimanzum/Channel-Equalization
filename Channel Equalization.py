import numpy as np
from scipy.stats import multivariate_normal
file=open('e_train.txt','r')
data=file.read()
#print(type(int(data[0])))
a = 1.0
b = 0.5

def noise():
    mu, sigma = 0.0,15 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    #print(s[0])
    return s[0]
    #return (np.random.rand()-0.5)*0.2
#print((np.random.rand()-0.5)*1.5)
def funcN(cur, prev):
    return a*cur + b*prev + noise()
def func(cur, prev):
    return a*cur + b* prev
mu = []
cnt = [] 
sigma = []
for i in range(8):
    mu.append([0, 0])
    cnt.append([0, 0])
    sigma.append([[0, 0], [0, 0]])


mu =np.array(mu)
cnt =np.array(cnt)
sigma = np.array(sigma)
#print(sigma.shape)


a = a*1.5
b = b*1.5
def train():
    global mu, cnt
    for i in range(len(data)-2):
        a1 = int(data[i+2])
        b1 = int(data[i+1])
        c1 = int(data[i])
        xk = funcN(a1, b1)
        xk_prev = funcN(b1, c1)
        #print(xk)
        index = 4*a1+2*b1+c1
        mu[index][0] += xk
        mu[index][1] += xk_prev
        cnt[index][0] += 1
        cnt[index][1] += 1
        if(i%1000000 == 0):
            print(i)
        if(i>100000):
            break
    #print(mu)
    mu = mu/cnt
    #print(mu)
    for i in range(len(data)-2):
        a1 = int(data[i+2])
        b1 = int(data[i+1])
        c1 = int(data[i])
        xk = funcN(a1, b1)
        xk_prev = funcN(b1, c1)
        index = 4*a1+2*b1+c1
        sigma[index][0][0] += (xk-mu[index][0])**2
        sigma[index][0][1] += (xk-mu[index][0])*(xk_prev-mu[index][1])
        sigma[index][1][0] += (xk_prev-mu[index][1])*(xk-mu[index][0])
        sigma[index][1][1] += (xk_prev-mu[index][1])**2
        #print(sigma[index][0][0])
        if(i%1000000 == 0):
            print(i)
        if(i>100000):
            break
    #print(sigma)
    for i in range(8):
        sigma[i][0][0] /= cnt[i][0]
        sigma[i][0][1] /= cnt[i][0]
        sigma[i][1][0] /= cnt[i][0]
        sigma[i][1][1] /= cnt[i][0]
        
    
train()
#print(sigma)

def test():
    tot = 0
    file=open('e_test.txt','r')
    data=file.read()
    global mu, cnt
    for i in range(len(data)-2):
        a1 = int(data[i+2])
        b1 = int(data[i+1])
        c1 = int(data[i])
        xk = func(a1, b1)
        xk_prev = func(b1, c1)
        #index = 4*a1+2*b1+c1
        dis = 1000
        idx = -1
        pred = -1
        for j in range(8):
            if((mu[j][0]-xk)**2 + (mu[j][1]-xk_prev)**2 < dis):
                dis = (mu[j][0]-xk)**2 + (mu[j][1]-xk_prev)**2
                idx = j
        if(idx<=3):
            pred = 0
        else:
            pred = 1
        if(pred != a1):
            print("mistake at ", i)
            tot+=1
    print("total mistake = ", tot)
def testv():
    tot = 0
    file=open('test.txt','r')
    data=file.read()
    global mu, cnt
    x = []
    x.append(0)
    for i in range(len(data)-1):
        #a1 = int(data[i+2])
        b1 = int(data[i+1])
        c1 = int(data[i])
        x.append(func(b1, c1))
    dp = []
    parent = []
    for i in range(len(x)):
        a = []
        b = []
        for j in range(8):
            a.append(-50000)
            b.append(0)
        dp.append(a)
        parent.append(b)
    #dp = [[-50000]*8]*(len(x))
    #parent = [[0]*8]*(len(x))
    #print(parent)
    if(data[0] == '0'):
        dp[0][0] = -1.38
        dp[0][1] = -1.38
        dp[0][2] = -1.38
        dp[0][3] = -1.38
    else:        
        dp[0][4] = -1.38
        dp[0][5] = -1.38
        dp[0][6] = -1.38
        dp[0][7] = -1.38

    for i in range(1, len(data)):
        for j in range(8):
            if(j<4):
                from1 = 2*j
                from2 = 2*j+1
            else:
                from1 =(j-4)*2
                from2 =(j-4)*2+1
                
            if(dp[i][j] < dp[i-1][from1]+np.log(0.5)+np.log(multivariate_normal.pdf([x[i], x[i-1]],mu[j], sigma[j]))):
                dp[i][j] = dp[i-1][from1]+np.log(0.5)+np.log(multivariate_normal.pdf([x[i], x[i-1]],mu[j], sigma[j]))
                parent[i][j] = from1
                
            if(dp[i][j] < dp[i-1][from2]+np.log(0.5)+np.log(multivariate_normal.pdf([x[i], x[i-1]],mu[j], sigma[j]))):
                dp[i][j] = dp[i-1][from2]+np.log(0.5)+np.log(multivariate_normal.pdf([x[i], x[i-1]],mu[j], sigma[j]))
                parent[i][j] = from2
        #print(dp[i])
    pred = ""
    dp = np.array(dp)
    #print(dp)
    last = np.argmax(dp[-1])
    if last<4:
        pred = "0" + pred
    else:
        pred = "1" + pred
    for i in range(len(data)-1, 0, -1):
        if parent[i][last]<4:
            pred = "0" + pred
        else:
            pred = "1" + pred
        last = parent[i][last]
    for i in range(min(len(data),len(pred)) ):
        if(data[i] != pred[i]):
            print("mistake at ", i);
            tot+=1
    print("total mistake ", tot)
    print(pred)
    print(len(pred))
    #print(len(dp[0]))
testv()
#test()
