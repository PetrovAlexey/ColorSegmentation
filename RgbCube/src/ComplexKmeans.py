from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import logging
import torch
import pptk
import itertools

class ComplexKmeans:
    """K-Means surface clustering.
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.

    """
    def __init__(self, n_clusters=1, n_lines=2, max_iter=2, n_init=2):
        self.n_clusters = n_clusters
        self.n_lines = n_lines
        self.max_plane_iter = max_iter
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = 10
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('spam')
        self.cluster_labeler = []
        print(self.n_clusters)  
    
    def _get_cluster_labels_gpu(self, X_gpu, planes, centroids, outliers=0):               
        d = -np.sum(planes*centroids, axis = 1)
        planes4d = np.c_[planes, d]       
        planes4d_gpu = torch.from_numpy(planes4d).float().to("cuda:0")
        distance = (torch.div(torch.matmul(X_gpu, planes4d_gpu.unsqueeze(2)), torch.norm(planes4d_gpu, p=2, dim=1)))[:,:,0]
        return torch.argmax(-torch.abs(distance), dim=0)
    
    def _get_cluster_labels_line_gpu(self, X_gpu, planes, centroids, outliers=0):
        lines4d_gpu = torch.from_numpy(planes).float().to("cuda:0")
        centroids_gpu = torch.from_numpy(centroids).float().to("cuda:0")
        distance = torch.zeros((len(centroids), X_gpu.shape[0]))
        for i in range(len(centroids)):
            verctors4d_gpu = - X_gpu[:,:-1] + centroids_gpu[i]
            temp = torch.cross(verctors4d_gpu, torch.cat(X_gpu.shape[0]*[lines4d_gpu[i].unsqueeze(0)]), dim=1)
            numerator = torch.norm(temp, p=2, dim=1)
            denominator = torch.norm(lines4d_gpu[i])
            distance[i] = numerator/denominator
        return torch.argmax(-torch.abs(distance), dim=0)
                

    def fit(self, X):
        """Compute k-means clustering.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        Returns
        -------
        self
            Fitted estimator.
        """
        best_centroids = None
        best_plane = None
        best_variance = None
        best_pca = [np.zeros(0) for i in range(self.n_clusters)]

        for i in range(self.n_init):
            k = self.n_clusters # - i + 1
            if k < 1:
                k = 1
            # Initialize centers
            planes = self._init_planes(k)
            centroids = self._init_centroids(k)      
            
            variance = np.zeros(k)
            pca_clusters = [np.zeros(0) for i in range(self.n_clusters)]
            
            X4d = np.c_[X, np.ones(X.shape[0])]
            X_gpu = torch.cat(1*[torch.from_numpy(X4d).float()]).to("cuda:0")

            for iteration in range(0, self.max_plane_iter):
                colours = self._get_cluster_labels_gpu(X_gpu, planes, centroids).cpu()
                for num_cluster in range(0, k):
                    pca = PCA(n_components=3)
                    data_cluster = np.array(X)[colours == num_cluster]
                    if (len(data_cluster) > 10):
                        pca.fit(data_cluster)
                        planes[num_cluster] = pca.components_[-1]
                        centroids[num_cluster] = pca.mean_
                        variance[num_cluster] = pca.explained_variance_[-1]
                        pca_clusters[num_cluster] = pca.components_
                    else:
                        planes[num_cluster] = planes[0]
                        + np.random.rand(3)*1
                        print("Lose")
                        # centroids[num_cluster] = centroids[num_cluster-1]
                        # + np.random.rand(3)*10
                        # print(planes[num_cluster])
                        # print(centroids[num_cluster])
                    logging.debug(f'Iteration: {iteration}')
                    logging.debug(f'Centroid {num_cluster}: {centroids[num_cluster]}')
                    logging.debug(f'Plane {num_cluster}: {planes[num_cluster]}')
                logging.debug(f'Variance: {np.sum(variance)}')

                if (best_variance is None or np.sum(variance) < best_variance):
                    best_variance = np.sum(variance)
                    best_plane = planes.copy()
                    best_centroids = centroids.copy()
                    best_pca = pca_clusters.copy()

        
        result_colours = self._get_cluster_labels_gpu(X_gpu, best_plane, best_centroids).cpu().numpy()
        self.cluster_labeler = [[i] for i in range(len(best_plane))]
        
        array_lines = [np.zeros(0) for i in range(len(best_plane))]
        array_centroids = [np.zeros(0) for i in range(len(best_plane))]

        plane_clusters = set(result_colours) 
        
        if self.n_lines <= 0:
             return [result_colours, best_plane, best_centroids, array_lines, array_centroids]
        
        for cluster in range(0, len(best_plane)):
            if cluster not in plane_clusters:
                continue
            if len(best_pca[cluster]) is 0:
                continue
            Y = X_gpu[result_colours == cluster]
            [colours, new_lines, new_line_centroids] = self._fit_lines(Y, best_plane[cluster])
         
            temp_colours = colours.cpu().numpy()
            self.cluster_labeler[cluster].extend(max(*self.cluster_labeler) + np.array(range(1, len(new_lines+1))))
            
            # update colours for lines in plane 
            merged = list(itertools.chain(*self.cluster_labeler))
            result_colours[result_colours == cluster] = max(merged) +colours.cpu().numpy() + 1
           
            # save best lines
            array_lines[cluster] = new_lines
            array_centroids[cluster] = new_line_centroids

        return [result_colours, best_plane, best_centroids, array_lines, array_centroids]
    
    def _fit_lines(self, X_gpu, plane):
        best_centroids = None
        best_lines = None
        best_variance = None
        
        for i in range(self.n_init):
            k = self.n_lines - i + 1
            if k <= 1:
                k = 2
            lines = self._init_lines_at_planes(k, plane)
            centroids = self._init_centroids(k)
            variance = np.zeros(k)
            ratio_variance = np.zeros(k)
            
            iteration = 0
            iteration_timeout = 0
            while iteration <= self.max_iter:
                iteration += 1
                colours = self._get_cluster_labels_line_gpu(X_gpu, lines, centroids)
                for num_cluster in range(0, k):
                    pca = PCA(n_components=3)
                    data_cluster = X_gpu[colours==num_cluster][:, :-1].cpu()
                    if (len(data_cluster) > 2):
                        pca.fit(data_cluster)
                        lines[num_cluster] = pca.components_[0]
                        centroids[num_cluster] = pca.mean_
                        variance[num_cluster] = pca.explained_variance_[-1] + pca.explained_variance_[-2]
                        ratio_variance[num_cluster] = pca.explained_variance_ratio_[-1] + pca.explained_variance_ratio_[-2]
                        logging.debug(f'Variance ratio: {pca.explained_variance_ratio_}')
                    else:
                        vec = np.random.rand(3)
                        #lines[num_cluster] = vec-vec*plane
                        lines[num_cluster] = self._init_lines_at_planes(1, plane)
                        #centroids[num_cluster] = centroids[num_cluster-1]# + np.random.rand(3)*3
                        print(lines[num_cluster])
                        print(centroids[num_cluster])
                        print("Lose")
                        iteration = 1
                        iteration_timeout += 1
                        if iteration_timeout > 4:
                            iteration = self.max_iter + 1
                    logging.debug(f'Iteration: {iteration}')
                    logging.debug(f'Centroid {num_cluster}: {centroids[num_cluster]}')
                    logging.debug(f'Line {num_cluster}: {lines[num_cluster]}')
                logging.debug(f'Variance: {np.sum(variance)}')
                logging.debug(f'Variance ratio: {np.sum(ratio_variance)}')
                
                if (best_variance is None or np.sum(variance) < best_variance):
                    best_variance = np.sum(variance)
                    best_lines = lines.copy()
                    best_centroids = centroids.copy()
                    
                if (np.sum(variance) < 31):
                    break

        colours = self._get_cluster_labels_line_gpu(X_gpu, best_lines, best_centroids)
        return [colours, best_lines, best_centroids]


    def _init_centroids(self, k: int):
        centroids = []
        for i in range(k):
            centroids.append(np.random.rand(3)*255)
        return np.array(centroids)

    def _init_centroids_2d(self, k: int):
        centroids = []
        for i in range(k):
            centroids.append(np.random.rand(2)*255)
        return np.array(centroids)

    def _init_lines(self, k: int):
        lines = []
        for i in range(k):
            lines.append(np.random.rand(2)*10)
        return np.array(lines)
    
    def _init_lines_at_planes(self, k: int, plane):
        planes = []
        for i in range(k):
            vec = np.random.rand(3)
            planes.append(vec-np.dot(vec, plane))
        return np.array(planes)
        
    def _init_planes(self, k: int):
        planes = []
        for i in range(k):
            planes.append(np.random.rand(3)*10)
        return np.array(planes)
