from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


if __name__ == "__main__":
  import argparse

  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  import tensorflow as tf
  import tensorflow_datasets as tfds

  parser = argparse.ArgumentParser()
  parser.add_argument("--num_partitions", help="Number of output partitions", type=int, default=10)
  parser.add_argument("--output", help="HDFS directory to save examples in parallelized format", default="data/mnist")

  args = parser.parse_args()
  print("args:", args)

  sc = SparkContext(conf=SparkConf().setAppName("data_setup"))

  mnist, info = tfds.load('mnist', with_info=True)
  print(info.as_json)

  # convert to numpy, then RDDs
  mnist_train = tfds.as_numpy(mnist['train'])
  mnist_test = tfds.as_numpy(mnist['test'])

  train_rdd = sc.parallelize(mnist_train, args.num_partitions).cache()
  test_rdd = sc.parallelize(mnist_test, args.num_partitions).cache()

  # save as CSV (label,comma-separated-features)
  def to_csv(example):
    return str(example['label']) + ',' + ','.join([str(i) for i in example['image'].reshape(784)])

  train_rdd.map(to_csv).saveAsTextFile(args.output + "/csv/train")
  test_rdd.map(to_csv).saveAsTextFile(args.output + "/csv/test")
