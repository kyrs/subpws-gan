"""
__desc__   : code for submodular selection
"""
import torch
import numpy as np
from apricot.utils import _calculate_pairwise_distances
from apricot import SaturatedCoverageSelection, FeatureBasedSelection, MixtureSelection, FacilityLocationSelection, GraphCutSelection
from scipy.stats import entropy
import random

class GraphCutSelectionGan(object):
    def __init__(self, train_idxs, Lambdas, embedding, nClass = 10, metric = "cosine", budgetRatio = 0.9, tradeOffAlpha = 2 ):
        # self.path_lfs_details = path_lfs_details 
        print("budgetRatio :", budgetRatio)
        print("alpha : ", tradeOffAlpha)
        self.nClass = nClass
        assert(train_idxs is not None)
        assert(Lambdas is not None)
        assert(embedding is not None)
        self.budgetRatio = budgetRatio
        self.counter = {"uniform": 0, "cost": 0}
        assert(self.budgetRatio < 1)

        if self.budgetRatio < 0: ## Flag to pass all the data without submodularity
            self.returnAll = True
        else:
            self.returnAll = False

        if self.returnAll:
            print("Note : Code will run without submodularity.")
        else:
            pass 

       
        examples, featureSize = embedding.shape
        assert(len(train_idxs)== examples) ## basic checks to ensure correct embeddings are choosen 
        lambda_tensor = torch.tensor(Lambdas, requires_grad=False).float()
        full_filter_idx = lambda_tensor.sum(1)!= 0 ## non abstrainIndex
        
        self.train_idxs_non_abstrained = train_idxs[full_filter_idx]
        self.lambdas_non_abstrained = Lambdas[full_filter_idx]
        self.embedding_non_abstrained = embedding[full_filter_idx] 
        
        print("Non abstrain data : ",np.sum(list(full_filter_idx)), "Total data : ", len(train_idxs))
        self.data = zip(train_idxs[full_filter_idx], Lambdas[full_filter_idx], embedding[full_filter_idx])
        ## defining model and calculating the pairwise distance
        if not self.returnAll:
            totalElmSelect = np.ceil(self.budgetRatio * len(self.train_idxs_non_abstrained))
            self.totalElmSelect = totalElmSelect
            print(f"total element to select : {totalElmSelect}")
            assert(tradeOffAlpha >= 2) ## for monotonicity alpha should be > 2in apricot
            self.modelCost = GraphCutSelection(totalElmSelect, alpha = tradeOffAlpha , optimizer='lazy', metric = 'precomputed')
            self.modelUniform = GraphCutSelection(totalElmSelect, alpha = tradeOffAlpha , optimizer='lazy', metric = 'precomputed')
            print("submodulalr function used : GraphCut")
            print ("Using the max formulation proposed in paper")
            self.pairwise_dist_non_abstrained = _calculate_pairwise_distances(self.embedding_non_abstrained, metric = metric)
        else:
            pass

    def returnSubset(self, dictProbPerSample):
        ## function to identify the cost for each sample in the optimization process 
        #print(dictProbPerSample)
        costList = []
        oneHotRes = {}
        for idx in range(len(self.train_idxs_non_abstrained)):
            elmVal = self.train_idxs_non_abstrained[idx]
            if elmVal in dictProbPerSample:
                probVal =  dictProbPerSample[elmVal]
                entropyVal = self.__entropy(probVal)
                costList.append(entropyVal)
                argMax = np.argmax(probVal)
                oneHot = np.zeros((self.nClass)) 
                oneHot[argMax] = 1 
                oneHotRes[elmVal] = oneHot
            else:
                raise Exception(f"elm id {elmVal} not found in the probability list")
                costList.append[1e4]
                oneHot = np.zeros((self.nClass))
                maxVal = np.max(self.lambdas_non_abstrained[idx])
                oneHot[maxVal-1] = 1
                oneHotRes[elmVal] = oneHot  
        if not self.returnAll:
            print("submodularity selection ..")
            ## normalizing the entropy
            newCostList = (costList/np.sum(costList)) * len(costList) ## ensuring that cost is normalized and in range of total datset size  

            fitObjCost = self.modelCost.fit(self.pairwise_dist_non_abstrained, sample_cost = np.array(newCostList))
            costValueSum = sum(fitObjCost.gains) ## Note : apricot save unweighted gain value(w.r.t sample_cost) in given variable (sum of gain can be used as the value of final submodualr fn)
            print (f"cost Value : {costValueSum}")
            
            fitObjUniform = self.modelUniform.fit(self.pairwise_dist_non_abstrained)
            uniformValueSum = sum(fitObjUniform.gains)
            print(f"uniform value : {uniformValueSum}")

            ## identify the budget val 
            assert(len(fitObjUniform.gains) == self.totalElmSelect )
            assert(len(fitObjUniform.ranking) == self.totalElmSelect )
            print(f"run info : {self.counter}")
            # assert( uniformValueSum == sum(fitObjUniform.gains))
            if uniformValueSum > costValueSum:
                print("max : Uniform")
                self.counter["uniform"] +=1
                listVal = fitObjUniform.transform(self.train_idxs_non_abstrained)
            else:
                print("max : Cost sensitive")
                self.counter["cost"] +=1
                listVal = fitObjCost.transform(self.train_idxs_non_abstrained)
            print(f" Length of selected samples : {len(listVal)}")
        else:
            if self.budgetRatio == -1 :
                listVal = self.train_idxs_non_abstrained
                assert(len(listVal) == len(self.train_idxs_non_abstrained))
                print(f" Length of selected samples : {len(listVal)}")
            else:
                pass
        returnClassDict = {}
        finalDict = {key : oneHotRes[key] for key in listVal}
        return finalDict

    def __entropy(self, prob):
        ## function to calculate entropy for the given prob value 
        assert(np.abs(sum(prob) - 1)< 1e-5 ) ## ensure that input value is probability and it does not overflow
        entropyVal = entropy(prob)
        return entropyVal
    
    def __loadDetails(self):
        (train_idxs,
        val_idxs,
        Lambdas,
        LF_accuracies,
        LF_propensity,
        LF_labels, embedding) = torch.load(self.path_lfs_details, map_location=lambda storage, loc: storage)

        return train_idxs, Lambdas, embedding

if __name__ == "__main__":
    pass
