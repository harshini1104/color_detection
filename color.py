# first of all, let us start with plotting the points of the image in 3-D space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import drive
import cv2

image = cv2.imread('drive/MyDrive/target_photos/2.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
red, green, blue = cv2.split(image)
red = red.flatten()
green = green.flatten()
blue = blue.flatten()

# Final steps in code
figure = plt.figure()
axis = Axes3D(figure)
axis.scatter(red, green, blue)
plt.show()

# Application of K-means clustering 
import cv2
from sklearn.cluster import KMeans

class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, img, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = img
        
    def dominantColors(self):
        image = cv2.imread(self.IMAGE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        self.IMAGE = image
        
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(image)
        
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_
        return self.COLORS.astype(int)

image = 'drive/MyDrive/target_photos/2.png'
# For better differentiation, more clusters
clusters = 11
dc = DominantColors(image, clusters) 
colors = dc.dominantColors()
print(colors)
