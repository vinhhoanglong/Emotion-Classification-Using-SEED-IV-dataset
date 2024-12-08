from utils import map_coor, load_data
from evaluation import *
from architecture import *
from sklearn.model_selection import train_test_split

import argparse


directories = ["data/eeg_feature_smooth/{}/".format(i+1) for i in range(3)]

coord_dict = map_coor()

array = load_data(directories=directories, coord_dict= coord_dict)

print(array.shape)

_X = array.reshape(np.prod(array.shape[0:3]), *array.shape[3:])
print(_X.shape)



session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}

y = np.array(session1_label * 15 + session2_label * 15 + session3_label * 15)

print(y.shape)
y_loso = np.reshape(y, (3,15,24))
print(y_loso.shape)

X = _X.transpose(0, 5, 1,2,3,4)
print(X.shape)
X = X.reshape(X.shape[0], X.shape[1], np.prod(X.shape[2:]))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X.shape[1:])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, required=True, choices=["dense", "3dconv", "lstm"],
    help="Model architecture to use: 'dense' or '3dconv' or 'lstm'"
)
args = parser.parse_args()

if args.model == "dense":
    model = dense
elif args.model == "3dconv":
    X = _X.transpose(0, 5, 2,3, 4, 1)[:,:,:,:,:,:]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], np.prod(X.shape[4:]))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = threedConv
elif args.model == "lstm":
    X = _X.transpose(0, 5, 2,3, 4, 1)[:,:,:,:,:,1]

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], np.prod(X.shape[4:]))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = convlstm2d


crossval(model, 40, X_train, y_train, X_test, y_test)




