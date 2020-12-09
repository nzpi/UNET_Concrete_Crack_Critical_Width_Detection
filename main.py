from setting import environment, constant
from util import path, generator
from nn import nn
import argparse

### python main.py --dip=example --tolabel
### python main.py --dataset=example --dip=example --augmentation=0000
### python main.py --dataset=example --arch=example --dip=example --gpu --test
### python main.py --dataset=example --arch=example --dip=example --gpu --train

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tolabel", help="Preprocess images to create labels", action="store_true", default=False)
	parser.add_argument("--augmentation", help="Dataset augmentation by x number", type=int)
	parser.add_argument("--dataset", help="Name of Dataset to be used", type=str, default=constant.DATASET)
	parser.add_argument("--train", help="Train", action="store_true", default=False)
	parser.add_argument("--convlayer", help="Train", action="store_true", default=False)
	parser.add_argument("--test", help="Predict", action="store_true", default=False)
	parser.add_argument("--arch", help="NN architecture to be used", type=str, default=constant.MODEL)
	parser.add_argument("--dip", help="Method for image processing", type=str, default=constant.IMG_PROCESSING)
	parser.add_argument("--gpu", help="Enable GPU mode", action="store_true", default=False)
	parser.add_argument("--measure", help="Crack dimension property approximation", action="store_true", default=False)
	args = parser.parse_args()

	environment.setup(args)
	exist = lambda x: len(x)>0 and path.exist(path.data(x, mkdir=False))
	
	if (args.tolabel):
		generator.tolabel()

	elif (args.measure):
		generator.measure()


	elif args.dataset is not None and exist(args.dataset):

		if (args.augmentation):
			generator.augmentation(args.augmentation)

		elif (args.train):
			nn.train()
			
		elif (args.test):
			nn.test()
		
		elif (args.convlayer):
			nn.convlayer()

	else:
		print("\n>> Dataset not found\n")

if __name__ == "__main__":
	main()

