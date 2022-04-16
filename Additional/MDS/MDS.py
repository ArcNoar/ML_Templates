"""
url_guide = https://www.youtube.com/watch?v=GEn-_dAyYME

Example_Source = https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/
"""

# Mertic MDS = PCoA

"""
Здесь рассматривались лишь евклидовы расстояни и манхэттэнские, помимо них существует еще множество методов измерения расстояния
Правильный выбор метода напрямую влияет на результат.
"""
"""
n_components = Number of Dimensions
metric = True\False (Metric Mds and Non-Metric MDS)

dissimilarity = Euclidian or Precomputed (Req : Fit(),fit_transform())

embedding_ = Loc of the points in new space

stress_ = Goodness of fit used in MDS

dissimilarity_matrix_ = Matrix of pairwise distances

n_iter_ = Iter Number

"""

from sklearn.manifold import MDS
from matplotlib import pyplot as plt

#import sklearn.datasets as dt

#import seaborn as sns
import numpy as np

from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


X = np.array([[0,0,0],[0,0,1],[1,1,1],[0,1,0],[0,1,1]])
mds = MDS(random_state=0)
X_transform = mds.fit_transform(X)

print(X_transform)

stress = mds.stress_

print(stress)

dist_manhattan = manhattan_distances(X)

mds = MDS(dissimilarity='precomputed',random_state=0)

#Get the embeddings

X_transform_L1 = mds.fit_transform(dist_manhattan)


#Plot constructing

colors = ['r','g','b','c','m']
size = [64,64,64,64,64]

fig = plt.figure(2,(10,4))
ax = fig.add_subplot(121,projection='3d')
plt.scatter(X[:,0],X[:,1],zs=X[:,2],s=size,c=colors)

plt.title('Original Points')

ax = fig.add_subplot(122)
plt.scatter(X_transform[:,0],X_transform[:,1],s=size,c=colors)
plt.title('Embedding in 2D')

fig.subplots_adjust(wspace=.4,hspace=0.5)
plt.show()


"""
Это не весь материал изложенный в статье, я написал только метрическуюю версию MDS
"""