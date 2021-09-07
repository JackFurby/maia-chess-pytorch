import gzip
import os
import yaml

LEELA_WEIGHTS_VERSION = 2

def read_weights_file(filename):
	if '.gz' in filename:
		opener = gzip.open
	else:
		opener = open
	with opener(filename, 'rb') as f:
		print(f.readline())
		version = f.readline().decode('utf-8')
		if version != '{}\n'.format(LEELA_WEIGHTS_VERSION):
			raise ValueError("Invalid version {}".format(version.strip()))
		weights = []
		e = 0
		for line in f:
			line = line.decode('ascii').strip()
			if not line:
				continue
			e += 1
			weight = list(map(float, line.split(' ')))
			weights.append(weight)
			if e == 2:
				filters = len(line.split(' '))
				#print("Channels", filters)
		blocks = e - (4 + 14)
		if blocks % 8 != 0:
			raise ValueError("Inconsistent number of weights in the file - e = {}".format(e))
		blocks //= 8
		#print("Blocks", blocks)
	return (filters, blocks, weights)


class Trained_Model(object):
	def __init__(self, path):
		self.path = path
		try:
			with open(os.path.join(path, 'config.yaml')) as f:
				self.config = yaml.safe_load(f.read())
		except FileNotFoundError:
			raise FileNotFoundError(f"No config file found in: {path}")

		self.weights = {int(e.name.split('-')[-1].split('.')[0]) :e.path for e in os.scandir(path) if e.name.endswith('.txt') or e.name.endswith('.pb.gz')}

	def getMostTrained(self):
		return self.weights[max(self.weights.keys())]

if __name__ == "__main__":
	#read_weights_file("./maia-1100.pb.gz")
	read_weights_file("./final_1100-40.pb.gz")
	#read_weights_file("./maia-1200.pb.gz")

	#model = Trained_Model("./")
	#print(model.weights)
	#print(model.getMostTrained())
