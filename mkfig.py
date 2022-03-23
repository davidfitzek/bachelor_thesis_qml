# mkfig.py

import sys
import json
import matplotlib.pyplot as plt

def mkfig(jsonfile):
	data = {}
	with open(jsonfile, 'r') as infile:
		data = json.load(infile)

	# Figure out number of iterations based on data length
	iters = [_ for _ in range(len(data['costs']))]

	plt.clf()

	plt.plot(iters, data['costs'], label = 'cost')
	plt.plot(iters, data['acc_train'], label = 'accuracy training')
	plt.plot(iters, data['acc_val'], label = 'accuracy validation')

	plt.xlabel('iterations')

	plt.legend()
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