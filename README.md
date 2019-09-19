# kmeans

k-means clustering algorithm implementation written in Go. The k-means 
clustering algorithm partitions a multi-dimensional data set into k clusters,
where each data point belongs to the cluster with the nearest mean, serving as
a prototype of the cluster.

This code is built upon the work of [https://github.com/muesli/kmeans](https://github.com/muesli/kmeans)
Just removed the features we considered unnecessary (such as charting), and the 
use of external dependencies so that the package is more self-contained.

## Future
We plan to add other distance functions besides the already implemented
Euclidean distance

