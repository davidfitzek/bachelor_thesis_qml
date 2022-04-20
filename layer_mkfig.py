import matplotlib.pyplot as plt
import pandas as pd

fields = ["Cross iteration", "Iteration", "Cost", "Accuracy"]


headers = fields

#df = pd.read_csv("./CANCER/1Layer/CANCERAmplitudeLayer1.csv")
#df = pd.read_csv("./CANCER/CANCERAngleLayer1.csv")
#df = pd.read_csv("./Adhoc/Adhoc3AmplitudeLayer5.csv")
#df = pd.read_csv("./Adhoc/Adhoc3AngleLayer7.csv")
#df = pd.read_csv("./Adhoc/Adhoc2AmplitudeLayer1.csv")
#df = pd.read_csv("./Adhoc/Adhoc2AngleLayer7.csv")
#df = pd.read_csv("./Iris/IrisAmplitudeLayer1.csv")
#df = pd.read_csv("./Iris/IrisAngleLayer7.csv")
df = pd.read_csv("./Forest/ForestAmplitudeLayer4.csv")
#df = pd.read_csv("./Forest/ForestAngleLayer6.csv")


accu = list(df["Accuracy"])
cost = list(df["Cost"])
iter = list(df["Iteration"])

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

plt.show()





