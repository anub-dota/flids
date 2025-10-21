.PHONY: run data shuffle

run:
	python main.py

data:
	python device.py && python generate_datapoints.py

shuffle:
	python shuffle_peerdata.py