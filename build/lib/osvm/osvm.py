
'''
INSTANCE SELECTION-BASED BINARY FIREFLY ALGORITHM
'''

#IMPORT PACKAGES
import numpy as np
from numpy import linalg
import csv
import cvxopt
import cvxopt.solvers

#START OF FIREFLY ALGORITHM

#DEFINE OBJECTIVE FUNCTION
def objective_function(dataset, value):
    idx = np.array([np.linalg.norm(x+y) for (x,y) in dataset-value]).argmin()
    return idx, dataset[idx]


#SPLIT THE DATASET  
def split_dataset(dataset, label):
    class_dataset = []
        
    X=dataset[:, [0,1]]

    Y=dataset[:, [2]]
    Y=np.array(Y).ravel()

    class_dataset = X[Y==label]
        
    return class_dataset


#FUNCTION FOR SELECT DISTINCT FIREFLIES
def select_distinct(dataset):
    null = 0
    #FOR SELECTING NON-ZERO INSTANCES - REMOVING NULL INSTANCES
    def non_zero_dataset(dataset):
        null = 0
        total = 0
        non_zero_dataset = []
        total_instances = 0
        null_list = []
        zero =[[0.0,0.0]]
        #zero = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]

        for instance in dataset:
            if instance.tolist() in zero:
                null += 1
            else:
                total += 1
                null_list.append(instance.tolist())
                non_zero_dataset = np.asfarray(null_list,float)
                total_instances = null + total
        print("Total Population of Fireflies : ", total_instances)
        print("Total Number of Null Fireflies : ", null)
        return non_zero_dataset

    #FOR SELECTING UNIQUE INSTANCES - REMOVING REDUNDANT INSTANCES
    def unique_dataset(dataset):
        instance_index = 0
        data_index = 0
        totCount = 0
        count = 0
        unt = 0
        redundant_arr = []
        uniq_arr = []
        redundant = 0
        uniq_counter = 0
        uniq_list = []
        total_red = 0

        for instance in dataset:
            uniq_counter += 1
            length = len(dataset.tolist())
            instance_index += 1
            data_index = 0

            for read_instance in dataset:
                data_index += 1
                if read_instance.tolist() == instance.tolist():
                    if data_index != instance_index:
                        totCount += 1
                        if count > 1:
                            count = 0
                            if instance.tolist() not in redundant_arr:
                                redundant_arr.append(instance.tolist())
                                redundant += 1
                                data_index = 0
                                count = 0
                                break
                            else:
                                break
                    else:
                        if instance.tolist() not in uniq_list:
                            uniq_list.append(instance.tolist())
                            count += 1
                            uniq_arr = np.asfarray(uniq_list,float)
        total_red = length - (len(uniq_arr) + null) 
        print("Total Number of Redundant Fireflies : ", total_red)
        return uniq_arr

    #SELECTING NON-ZERO FIREFLIES
    non_zero = non_zero_dataset(dataset)
    #SELECTING UNIQUE FIREFLIES
    unique = unique_dataset(non_zero)
    
    return unique


#FUNCTION FOR EACH LABEL IN BFFA
def is_bbfa(dataset, label):

    #SPLIT THE DATASET & SELECT THE LABEL ACCORDING TO INITIALIZE PARAMETER
    label_dataset = split_dataset(dataset, label)
    print("Class Dataset : ", label_dataset)
    #SLECT THE DISTINCT FIREFLIES OF THAT LABEL
    distinct_dataset = select_distinct(label_dataset)
    print("Distinct Dataset : ", distinct_dataset)

    #INITIALIZE THE MAXIMUM NUMBER OF SELECTED FIREFLIES
    #TOTAL NUMBER OF FIREFLIES
    tn = len(distinct_dataset)
    print("Total Number of Distinct Fireflies : ", tn)
    #CALCULATED MAX NUMBER OF SELECTED FIREFLIES
    mns = int(round(int(tn * 75) / 100))
    print("Maximum Number of Selected Fireflies : ", mns)
    #SELECT RELEVANT FIREFLIES
    bts = []
    bef = []
    orig_dataset = list(distinct_dataset)
    mask_dataset = []

    #EVALUATE OBJECTIVE FUNCTION TO DETERMINE INITIAL LIGHTEST FIREFLY
    light = objective_function(distinct_dataset, distinct_dataset.mean(0))
    print("Initial Lightest Firefly : ", light[1])
    index = light[0]
    bef.append(light[1].tolist())
    #print("Index ", index)
    #print("Remove ", orig_dataset[index])
    del orig_dataset[index]
    #print("removed orig_dataset ", orig_dataset)
    mask_dataset = np.asfarray(orig_dataset,float)
    #print("Mask Dataset ", mask_dataset)
    
    i = 1
    for i in range(mns-1):
        #print("I",i)
        for j in mask_dataset:
            new_light = objective_function(mask_dataset,index)
            new_index = new_light[0]
            #print("New Light",new_light[1])

            if new_light[1].tolist() not in bef:
                bef.append(new_light[1].tolist())
                #print("New Index ", new_index)
                #print("Remove ", orig_dataset[new_index])
                del orig_dataset[new_index]
                #print("Removed orig_dataset ", orig_dataset)
                mask_dataset = np.asfarray(orig_dataset,float)
                #print("Mask Dataset ", mask_dataset)
                #print("Best Firefly List ", bef_list)
                
            break

    bts = np.asfarray(bef,float)
    print("Best Training Subset :", bts)

    return bts
    

#COMBINE THE DATASET 
def combine_dataset(dataset1, label1, dataset2, label2):
    bts_list = []
    total_bts = []
    d1len = len(dataset1)
    d2len = len(dataset2)

    i = 0
    for i in range(d1len):
        X1=dataset1[i][0],dataset1[i][1],label1
        bts_list.append(X1)
    j = 0
    for j in range(d2len):
        X2=dataset2[j][0],dataset2[j][1],label2
        bts_list.append(X2)
    
    total_bts = np.asfarray(bts_list, float)

    return total_bts


#CLASS FOR INSTANCE SELECTION-BASED BINARY FIREFLY ALGORITHM
class IS_BFFA(object):
    def __init__(self, dataset):
        self.main(dataset)

    #FUNCTION FOR IS_BFFA
    def main(self, dataset):
        #print(dataset)
        #FOR POSITIVE CLASS
        print("Positive Class")
        positive_class = is_bbfa(dataset, 1)

        #FOR NEGATIVE CLASS
        print("\n\n\nNegative Class")
        negative_class = is_bbfa(dataset, -1)
        
        #RETURN THE TOTAL BTS = COMBINE POSITIVE & NEGATIVE CLASS
        self.best_training_subset = combine_dataset(positive_class, 1, negative_class, -1)
        length = len(self.best_training_subset)
        print("\n\n\nBest Training Subset : ", self.best_training_subset)
        print("Total Best Training Subset : ", length)

        #READ FOR TRAINING
        return self.best_training_subset

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

class OPT_SVM(object):

    def __init__(self, kernel=polynomial_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
                #print(X[i],X[j])
                #print(K[i,j])

        #CREATING VALUES FOR QUADRATIC PROGRAMMING SOLVER
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        #If it is Soft margin
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        #print(solution)

        # Lagrange multipliers-gets alpha here
        a = np.ravel(solution['x'])
        #print(a)

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        #print(sv)
        #print(a)
        ind = np.arange(len(a))[sv]
        #print(ind)
        self.a = a[sv]
        #print(self.a)
        self.sv = X[sv]
        #print(self.sv)
        self.sv_y = y[sv]
        #print(self.sv_y)
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept-bias
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

        #print (self.w)
    
    #Getting prediction
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                    #print(s)
                y_predict[i] = s
            return y_predict + self.b
            #print( y_predict + self.b)



    #Getting sign of the prediction
    def predict(self, X):
        return np.sign(self.project(X))

    #Gets the accuracy score of Testing
    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        
#TESTING
def main():
    with open("Dataset#1.csv") as fp:
            #READ CSV FILE
            reader=csv.reader(fp,delimiter=",")
            data=[line for line in reader]
            
            #REMOVES THE HEADER
            del data[0]
            #SAVE THE FIREFLIES IN ARRAY
            fireflies=np.asfarray(data,float)    

            #THIS VARIABLE IS READY FOR TRAINING                        
            isbffa = IS_BFFA(fireflies)


#CALL THE FUNCTION TO EXECUTE
#main()