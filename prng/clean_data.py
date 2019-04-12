import argparse, os, csv
import numpy as np

def clean_data(files, bits):
  first = 1

  for dataset in files:
    clean_dataset = os.path.basename(dataset).replace('.txt', '') + '_clean.txt'
    with open(clean_dataset, 'w+') as fout:
      writer = csv.writer(fout, delimiter=',')
  
      # load data
      with open(dataset) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
          for i in range(bits):
            if not np.isfinite(np.array([row[i]]).astype(float)):
              break
          writer.writerow(row)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--training', metavar='T', nargs='+', help='the training datasets; .txt or .csv file', required=1)
  parser.add_argument('-b', '--bits', help='the number of bits in the prn', type=int, required=1)

  args = parser.parse_args()

  clean_data(args.training, args.bits)
  
  


# use keras


