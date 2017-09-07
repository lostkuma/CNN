import csv
import random

random1000 = [random.uniform(0.0, 1.0) for i in range(1000)]
test_set = list()
training_set = None
header = None

with open("image.genre.listing.csv", newline="") as csvfile:
	reader = list(csv.reader(csvfile, delimiter=","))
	header = reader.pop(0)
	total = len(reader)
	for i in range (1000):
		index = int(random1000[i] * (total - i))
		test_set.append(reader[index])
		reader.pop(index)
	training_set = reader

with open("test.image.genre.listing.csv", "w", newline="") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(header)
	for row in test_set:
		writer.writerow(row)

with open("train.image.genre.listing.csv", "w", newline="") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(header)
	for row in training_set:
		writer.writerow(row)
