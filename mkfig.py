import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#fields = ["Cross iteration", "Iteration", "Cost", "Accuracy"]
fields = ["Iterations mean", "Iterations dev", "Cost mean", "Cost dev", "Accuracy mean", "Accuracy dev", "Seconds"]

headers = fields
df = pd.read_csv("IrisAngle.csv")
accu = list(df["Accuracy mean"])

plt.plot(accu)
plt.show()

print(accu)


