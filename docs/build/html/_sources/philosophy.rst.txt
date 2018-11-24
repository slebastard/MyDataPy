Project philosophy & architecture
=================================

Instances
---------

This repository contains several learning methods. Those methods are roughly partioned as supervised on one side, and unsupervised on the other.
Each learning method is implemented in DataPy as a class. The learning method is an object that contains several (sub)methods and attributes.
To use the algorithm on data, an instance is created. Using methods of this instance causes its attributes to change value.
This way, one can have all variables resulting from an analysis within a single instance of a class.

For instance, the K-means method is implemented as class Kmeans.

If you have a dataset X on which you want to run Kmeans, you would do:

    KM = KMeans(X, nclass=3)
    KM.run()

Here, KM is an instance of KMeans for 3 clusters, initially bound to dataset X.
Once KM.run() is executed, KM will contain several attributes relative to the clusters built around dataset X.
However, KM can be untied from X, and can be bound to another dataset Y:

	KM.unbind()
	KM.bind(Y)
	KM.run()