# mkfig.py

import statistics as stat

import sys
import json
import matplotlib.pyplot as plt

def getdata(data, key):

	# Figure out number of iterations based on data length
	iters = [_ for _ in range(len(data['cross iter1'][key]))]

	res = [[cross_val_data[key][iter] for _, cross_val_data in data.items()] for iter in iters]

	return res

def getstat(vals):
	mean = [stat.mean(val) for val in vals]
	stdev = [stat.stdev(val, xbar = m) for val, m in zip(vals, mean)]

	return mean, stdev

def mkfig(jsonfile):
	data = {}
	with open(jsonfile, 'r') as infile:
		data = json.load(infile)

	key_labels = {
		'costs': 'cost',
		'acc_train': 'accuracy training',
		'acc_val': 'accuracy validation'
	}

	key_colours = {
		'costs': 'blue',
		'acc_train': 'green',
		'acc_val': 'orange'
	}

	# Figure out number of iterations based on data length
	iters = [_ for _ in range(len(data['cross iter1']['costs']))]

	plt.clf()

	for key, label in key_labels.items():
		vals = getdata(data, key)
		mean, stdev = getstat(vals)
		upper = [m + e for m, e in zip(mean, stdev)]
		lower = [m - e for m, e in zip(mean, stdev)]

		# Plot mean value
		plt.plot(iters, mean, label = label, color = key_colours[key], alpha = 1)

		# Plot upper and lower bound
		plt.plot(iters, lower, color = key_colours[key], alpha = 0.3)
		plt.plot(iters, upper, color = key_colours[key], alpha = 0.3)

		# Plot error region
		plt.fill_between(iters, lower, upper, color = key_colours[key], alpha = 0.1)

	plt.xlabel('iterations')

	plt.legend()
	#plt.show()
	plt.savefig('figs' + jsonfile[4 : -4] + 'pdf')

def main():
	for arg in sys.argv:
		# Arguments are expected to be json files in data/
		if (arg.startswith('data/') and arg.endswith('.json')):
			mkfig(arg)
			print('Created: figs' + arg[4 : -4] + 'pdf')
		else:
			print('Usage: python mkfig.py data/*.json')

if __name__ == '__main__':
	main()
