.PHONY: run data shuffle eda plot all

run:
	python main.py

data:
	python device.py && python generate_datapoints.py

shuffle:
	python shuffle_peerdata.py

eda:
	python eda_analysis.py

plot:
	python plot_training.py

all: data shuffle run plot eda