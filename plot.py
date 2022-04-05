import matplotlib.pyplot as plt

import IrisAmplitudeCost.csv

plt.subplot(3, 1, 1)
			plt.plot(iterations)
			plt.xlabel("Layer")
			plt.ylabel("Iterations")

			plt.subplot(3, 1, 2)
			plt.plot(cost)
			plt.xlabel("Layer")
			plt.ylabel("Cost")

			plt.subplot(3, 1, 3)
			plt.plot(sec)
			plt.xlabel("Layer")
			plt.ylabel("Seconds to execute")

			plt.show()