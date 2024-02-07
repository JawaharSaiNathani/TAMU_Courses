import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score, KFold

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Data Points', s=10)
    plt.ylim(-1, 0)
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title("Visualize Features")
    plt.savefig('train_features.png')
    # plt.show()

    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    # Scatter Data Points onto the Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Data Points', s=10)

    # Plot Decision Boundary
    boundary_x = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    boundary_y = -(W[0] + W[1] * boundary_x) / W[2]
    plt.plot(boundary_x, boundary_y, color='red', label='Decision Boundary')

    # Customize Plot
    plt.ylim(-1, 0)
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.savefig('train_result_sigmoid.png')
    # plt.show()
	### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 500).reshape(-1, 1),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 500).reshape(-1, 1)
    )

    # Create a meshgrid for decision boundary visualization
    meshgrid_points = np.c_[xx.ravel(), yy.ravel()]
    meshgrid_points_withbias = np.insert(meshgrid_points, 0, 1, axis=1)

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.W = W
    Z = logisticR_classifier_multiclass.predict(np.array(meshgrid_points_withbias))
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundaries
    classes = ['Class 0', 'Class 1', 'Class 2']
    formatter = plt.FuncFormatter(lambda i, *args: classes[int(i)])
    custom_cmap = ListedColormap(['#b4a7d6','#93c47d','#fff2cc'])
    plt.figure(figsize=(10, 5))
    plt.contourf(xx, yy, Z, cmap=custom_cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)

    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title('Multi-Class Logistic Regression Decision Boundary')
    plt.tight_layout()
    plt.savefig('train_result_softmax.png')
    # plt.show()
	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
	
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


    #------------Logistic Regression Sigmoid Case------------

    ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.05, max_iter=1000)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print('LR Tr Acc -', logisticR_classifier.score(train_X, train_y))
    print('LR Va Acc -', logisticR_classifier.score(valid_X, valid_y))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, logisticR_classifier.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_test_data)
    test_y_all, test_idx = prepare_y(test_labels)

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = -1
    print("LR Ts Acc -", logisticR_classifier.score(test_X, test_y))
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=1500,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 12)
    print("MLC Tr Acc -", logisticR_classifier_multiclass.score(train_X, train_y))
    print("MLC Va Acc -", logisticR_classifier_multiclass.score(valid_X, valid_y))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, logisticR_classifier_multiclass.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    print('MLC Ts Acc - ', logisticR_classifier_multiclass.score(test_X_all, test_y_all))
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = 0

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000,  k= 2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print('Sof Conv Tr Acc -', logisticR_classifier_multiclass.score(train_X, train_y))
    print('Sof Conv Va Acc -', logisticR_classifier_multiclass.score(valid_X, valid_y))
    print('Sof Conv Ts Acc -', logisticR_classifier_multiclass.score(test_X, test_y))
    print('Sof Params - ', logisticR_classifier_multiclass.get_params())
    ### END YOUR CODE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = -1

    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=10000)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print('Sig Conv Tr Acc -', logisticR_classifier.score(train_X, train_y))
    print('Sig Conv Va Acc -', logisticR_classifier.score(valid_X, valid_y))
    print('Sig Conv Ts Acc -', logisticR_classifier.score(test_X, test_y))
    print('Sig Params - ', logisticR_classifier.get_params())
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=1)
    logisticR_classifier.fit_miniBGD(train_X[:8], train_y[:8], 1)

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.15, max_iter=1,  k= 2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X[:8], train_y[:8], 1)

    print(logisticR_classifier.get_params())
    w_t = logisticR_classifier_multiclass.get_params().T
    print(w_t[1] - w_t[0])
    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
