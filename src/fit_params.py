
import numpy as np, pandas as pd, argparse, matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
x_data = df['x'].to_numpy()
y_data = df['y'].to_numpy()
N = len(df)
t = np.linspace(6,60,N)

def model(theta_deg, M, X):
    th = np.deg2rad(theta_deg)
    e = np.exp(M*np.abs(t))
    s = np.sin(0.3*t)
    x = t*np.cos(th) - e*s*np.sin(th) + X
    y = 42 + t*np.sin(th) + e*s*np.cos(th)
    return x, y

theta = 28.119274865779154
M = 0.02138370771522498
X = 54.91110507712368

x_hat, y_hat = model(theta, M, X)
L1 = np.sum(np.abs(x_data-x_hat)+np.abs(y_data-y_hat))

print("theta =", theta)
print("M     =", M)
print("X     =", X)
print("L1 Loss =", L1)

plt.figure()
plt.scatter(x_data,y_data,s=6,label="Data")
plt.plot(x_hat,y_hat,label="Model")
plt.title("Fit Overlay")
plt.legend()
plt.show()
