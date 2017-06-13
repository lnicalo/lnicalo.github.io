---
title: Moving Spark Jobs from HDFS to Google Cloud Storage
featured: images/blog/2017-05-11-Moving-Spark-Jobs-from-HDFS-to-GCS.jpg
layout: post
date: '2016-05-29'
category: [GCS, Spark, S3]
tags: [GCS, Spark, S3]
author: luis
---

Few months ago, it came to my attention an interesting trend in favor of storing massive datasets on blob storage systems like Amazon S3 or Google Cloud Storage (GCS) rather than on HDFS - the Hadoop distributed file system.

In line with this trend, one of our customers wanted to deploy a storage solution using GCS as well as to use Spark to run SQL queries on the stored dataset.

HDFS has erected as probably the solution of choice for building data warehouses over the last years. Nevertheless, although HDFS is indeed a fully scalable storage solution, it comes with a bunch of sticky inconveniences that makes company consider other approaches as well.

Scalability is at expense of adding more nodes to the cluster. This operation may be relatively easy for trained teams but it can be quite cumbersoming for those that do not have a decicated infrastructure person to operate HDFS. Having the cluster down is an uncceptable operational risk.

Moreover, object storages have been engineered to be reliable. Hence, they are able to offer a bettern price than HDFS as there is no need to store three copies of each block of data.

Using Google Cloud Dataproc (think 'Spark-as-a-Service' and the Amazon's EMR counterpart), it is possible to read and write GCS. Some gotchas are on the way as a consequence of using a object store and not a file system. Writing the last phase of Spark job is terribly slow due to non-atomic renames have to be handled in the application code. Read [Apache Spark and Amazon S3 - Gotchas and best practices](https://www.linkedin.com/pulse/apache-spark-amazon-s3-gotchas-best-practices-subhojit-banerjee){:target="blank"} to have a better picture and how to solve it. Spark 2.0 treats some of these issues.

After a month of work, our customer couldn't be happier by enjoying the automatic scalability and better storage price of GCS. There were other issues in the project like parquet storage but it may be a matter of other post. Stay tuned.
