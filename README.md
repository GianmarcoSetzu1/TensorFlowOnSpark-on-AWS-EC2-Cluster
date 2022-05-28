# TensorFlowOnSpark-on-AWS-EC2-Cluster
University project for the Big Data course. The project consists of a TensorFlow application adapted to a Hadoop / Spark cluster to be able to run in distributed mode with the use of Yarn. The application trains a neural network implemented with Tensorflow, in distributed mode on a cluster of 5 nodes (AWS EC2 Ubuntu Server 20.04 instances). The dataset used is "mnist", the guides will tell you how to download and use it. Inside the repository there is a guide for configuring the cluster, a guide for reusing the project, a script for transforming the dataset into csv format and the script to launch the application.