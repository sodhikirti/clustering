import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib import style
style.use('ggplot')


from sklearn.metrics import silhouette_samples as SS_score
from sklearn.metrics import calinski_harabasz_score as CH_score
from sklearn.metrics import davies_bouldin_score as DB_score

from sklearn.cluster import SpectralClustering as Spectral
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as ACluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation as AP_model
from sklearn.mixture import GaussianMixture as GM_model

from sklearn.cluster import MiniBatchKMeans as MBKmeans
from sklearn.cluster import MeanShift as Meanshift
from sklearn.cluster import Birch 





class clustering_algothms():
    def __init__(self, n_clu = None, random_seed = 123, AP_prefer = None, DB_eps = None, DB_min_s = None, MS_bd = None):

        self.n_clu = n_clu
        self.random_seed = random_seed
        self.preference = AP_prefer # For AffinityPropagation
        self.eps, self.min_samples = DB_eps, DB_min_s # for DBSCAN
        self.MS_bd = MS_bd # for meanshift
    
    @staticmethod    
    def default_param_check():
        params_default = {'KMeans': KMeans().get_params(),
                          'MBKmeans': MBKmeans().get_params(),
                          'GM_model':GM_model().get_params(),
                          'DBSCAN': DBSCAN().get_params(),
                          'ACluster': ACluster().get_params(),# Not able to predict new data
                          'Spectral': Spectral().get_params(),
                          'APCluster': AP_model().get_params(),
                          'Meanshift': Meanshift().get_params(),
                          'Birch': Birch().get_params()}
        return params_default
    
    def algotm(self):
        
        ## n_clu as hyperparameters
        kmeans = KMeans (n_clusters = self.n_clu, max_iter = 100000000,random_state = self.random_seed)
        mbkeans = MBKmeans(n_clusters = self.n_clu,random_state = self.random_seed)
        gmm = GM_model(n_components = self.n_clu, random_state = self.random_seed)      
        
        acluster = ACluster (n_clusters = self.n_clu)
        
        spectral = Spectral (n_clusters = self.n_clu)
        
        birch =  Birch(n_clusters = self.n_clu)# sometimes have problem to pick birch model, error:maximum recursion depth exceeded while calling a Python object
        
        ## do not need n_clu as hyperparameters
        
        ap_model = AP_model (preference = self.preference)# The higher preference, the more cluster; 
        
        dbscan = DBSCAN(eps = self.eps, min_samples = self.min_samples, n_jobs = -1)
        
        meanshift = Meanshift(bandwidth = self.MS_bd, n_jobs = -1, bin_seeding = True, min_bin_freq = 5, cluster_all = False) 
        #the smaller bandwidth, the more clusters , n_jobs: how many cpus being used, -1 means all  

        
    
        ALGTM = {'KM': kmeans, 'MBKM':mbkeans, 'GMM': gmm, 'DBSCAN': dbscan, 'ACluster': acluster, 'Spectral': spectral,\
                 'Birch': birch,'APCluster': ap_model, 'Meanshift': meanshift}
        
        return ALGTM
        
class run_single_model(clustering_algothms):
    
    def __init__(self, model_name, learn_data, n_clu = None, AP_prefer = None, DB_eps = None, DB_min_s = None, MS_bd = None):
        super().__init__(n_clu = n_clu, AP_prefer = AP_prefer, DB_eps = DB_eps, DB_min_s = DB_min_s, MS_bd = MS_bd)
        
        self.model_name = model_name
        self.learn_data = learn_data
        
        self.Model, self.labels, self.clu_centers  = self.run_model()
        

    def run_model(self):
        
        Model = self.algotm()[self.model_name].fit(self.learn_data)
        
        if hasattr(Model, 'labels_'):
            labels = Model.labels_
        else:
            labels = Model.predict(self.learn_data)
            
        if hasattr(Model, 'cluster_centers_'):
            clu_centers = Model.cluster_centers_
        else:
            clu_centers = None
                
        return Model, labels, clu_centers

     
    def model_measure (self):
        measure_score = {self.model_name :{
         'SS_score': SS_score(self.learn_data, self.labels).mean(), # -1<= 0 <= 1, the higher the better
         'DB_score': DB_score(self.learn_data, self.labels), # >0, the lower the better
         'CH_score': CH_score(self.learn_data, self.labels)  # the higher the better
           }}
        return measure_score
    
def Search_N_for_single_model (model_name, learn_data, k1, k2):

    keys = {k for k in range(k1, k2)}

    
    values =[run_single_model(model_name, learn_data, n_clu).model_measure() for n_clu in range(k1, k2)]

    measure_score = dict(zip(keys, values))

    return measure_score

def Search_N_for_models (model_list, learn_data, k1, k2):

    compare = pd.DataFrame()
    for model_name in model_list:
        ms = Search_N_for_single_model (model_name, learn_data, k1,k2)
        compare = pd.concat([compare, pd.DataFrame(ms)])
  

    data_plot = compare
    fig, ax = plt.subplots(figsize=(20,5))

    plt.subplot(131)
    for i in  range(0, len(model_list)):
        model_name = model_list[i]
        colr = cm.Pastel1(i) 

        plt.plot(data_plot.columns, [data_plot.loc[model_name, clu].get('SS_score') for clu in range(k1,k2)],color = colr , linewidth = 5, marker = 'o', label = model_name)

        plt.title('Silhouette_score (The higher the better)',fontsize = 15)
        plt.legend(loc='best')

    plt.subplot(132)
    for i in  range(0, len(model_list)):
        model_name = model_list[i]
        colr = cm.Pastel1(i) 
        plt.plot(data_plot.columns, [data_plot.loc[model_name, clu].get('CH_score') for clu in range(k1,k2)],color = colr , linewidth = 5,
                 marker = 'o', label = model_name)

        plt.title('Calinski_Harabasz_score (The higher the better)',fontsize = 15)
        plt.legend(loc='best')


    plt.subplot(133)
    for i in  range(0, len(model_list)):
        model_name = model_list[i]
        colr = cm.Pastel1(i) 
        plt.plot(data_plot.columns, [data_plot.loc[model_name, clu].get('DB_score') for clu in range(k1,k2)],color = colr , linewidth = 5,
                 marker = 'o', label = model_name)

        plt.title('Davies_Bouldin_score (The lower the better)',fontsize = 13)
        plt.legend(loc='best')
    return compare