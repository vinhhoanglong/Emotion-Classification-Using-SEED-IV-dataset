# define cross valuation and loso cross valuation

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import tensorflow as tf


def crossval(generate_model, n_epochs, X_train, y_train, X_test, y_test, filename = None):
    kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=0)
    bestmodel = None
    bestAcc = 0
    cvscores = []
    fold = 1
    for train, test in kfold.split(X_train, y_train):
        model = generate_model()
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold} ...')
        model.fit(X_train[train], y_train[train],epochs=n_epochs, verbose=1) # validation_split=0.2)
        scores = model.evaluate(X_train[test], y_train[test], verbose=1)
        print("Score for fold %d - %s: %.2f%%" % (fold, model.metrics_names[1], scores[1]*100))
        if(scores[1] > bestAcc):
            bestAcc = scores[1]
            bestmodel = model
        cvscores.append(scores[1] * 100)
        fold += 1
    print('------------------------------------------------------------------------')
    print("Avg accuracies: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    test_loss, test_acc = bestmodel.evaluate(X_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    if filename:
        pickle.dump( bestmodel, open( filename, "wb" ) )