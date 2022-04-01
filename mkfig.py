# mkfig.py

import statistics as stat

import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def getdata(data, key):

	# Figure out number of iterations based on data length
	iters = [_ for _ in range(len(data['cross iter1'][key]))]

	res = [[cross_val_data[key][iter] for _, cross_val_data in data.items()] for iter in iters]

	return res

def getstat(vals):
	mean = [stat.mean(val) for val in vals]
	stdev = [stat.stdev(val, xbar = m) for val, m in zip(vals, mean)]

	return mean, stdev

def getbounds(vals):
	mean = [stat.mean(val) for val in vals]
	maxv = [max(val) for val in vals]
	minv = [min(val) for val in vals]

	return mean, maxv, minv

def prepdata(data, key):
	vals = getdata(data, key)
	mean, stdev = getstat(vals)
	upper = [m + e for m, e in zip(mean, stdev)]
	lower = [m - e for m, e in zip(mean, stdev)]

	return mean, upper, lower

def plot(fig, iters, mean, upper, lower, label, colour):
	# Plot mean value
	fig.plot(iters, mean, label = label, color = colour, alpha = 1)

	# Plot upper and lower bound
	fig.plot(iters, lower, color = colour, alpha = 0.3)
	fig.plot(iters, upper, color = colour, alpha = 0.3)

	# Plot error region
	fig.fill_between(iters, lower, upper, color = colour, alpha = 0.1)

	fig.tick_params(axis = 'y')

def mkfig(jsonfile):
	data = {}
	with open(jsonfile, 'r') as infile:
		data = json.load(infile)

	key_labels = {
		'costs': 'cost',
		'acc_train': 'accuracy training',
		'acc_val': 'accuracy'
	}

	key_colours = {
		'costs': 'blue',
		'acc_train': 'orange',
		'acc_val': 'green'
	}

	# Figure out number of iterations based on data length
	iters = [_ for _ in range(len(data['cross iter1']['costs']))]

	plt.clf()

	fig, ax1 = plt.subplots()

	ax1.set_xlabel('iterations')
	ax1.set_ylabel(key_labels['costs'])

	cost_mean, cost_upper, cost_lower = prepdata(data, 'costs')

	plot(ax1, iters, cost_mean, cost_upper, cost_lower, key_labels['costs'], key_colours['costs'])

	ax2 = ax1.twinx()
	ax2.set_ylabel(key_labels['acc_val'])

	acc_mean, acc_upper, acc_lower = prepdata(data, 'acc_val')

	plot(ax2, iters, acc_mean, acc_upper, acc_lower, key_labels['acc_val'], key_colours['acc_val'])

	l1 = mpatches.Patch(color = key_colours['costs'], label = key_labels['costs'])
	l2 = mpatches.Patch(color = key_colours['acc_val'], label = key_labels['acc_val'])

	plt.legend(handles = [l1, l2], loc = 'upper right')

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
