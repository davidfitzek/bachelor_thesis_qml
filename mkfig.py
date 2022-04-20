import matplotlib.pyplot as plt
import pandas as pd

fields = ["Iterations mean", "Iterations dev", "Cost mean", "Cost dev", "Accuracy mean", "Accuracy dev", "Seconds"]

headers = fields

#df = pd.read_csv("./CANCER/CANCERAmplitude.csv")
#df = pd.read_csv("./CANCER/CANCERAngle.csv")
#df = pd.read_csv("./Adhoc/Adhoc3Amplitude.csv")
#df = pd.read_csv("./Adhoc/Adhoc3Angle.csv")
#df = pd.read_csv("./Adhoc/Adhoc2Amplitude.csv")
#df = pd.read_csv("./Adhoc/Adhoc2Angle.csv")
#df = pd.read_csv("./Iris/IrisAmplitude.csv")
#df = pd.read_csv("./Iris/IrisAngle.csv")
#df = pd.read_csv("./Forest/ForestAmplitude.csv")
df = pd.read_csv("./Forest/ForestAngle.csv")


accu = list(df["Accuracy mean"])
cost = list(df["Cost mean"])
iter = list(df["Iterations mean"])

layers = list(range(1, len(accu) + 1))

plt.subplot(3, 1, 1)
plt.plot(layers, accu)
plt.ylabel("Accuracy")

plt.subplot(3, 1, 2)
plt.plot(layers, cost)
plt.ylabel("Cost")

plt.subplot(3, 1, 3)
plt.plot(layers, iter)
plt.ylabel("Iterations")
plt.xlabel("Number of layers")

plt.show()





