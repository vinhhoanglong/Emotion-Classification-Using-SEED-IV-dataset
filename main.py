from utils import map_coor, load_data
from evaluation import *
from architecture import *
from sklearn.model_selection import train_test_split

directories = ["data/eeg_feature_smooth/{}/".format(i+1) for i in range(3)]

coord_dict = map_coor()

array = load_data(directories=directories, coord_dict= coord_dict)

print(array.shape)

_X = array.reshape(np.prod(array.shape[0:3]), *array.shape[3:])
print(_X.shape)
X_loso = array[:,:,:,1,:,:,]
X_loso = np.transpose(X_loso, (0,1,2,6,3,4,5))
print(X_loso.shape)


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
crossval(dense, 40, X_train, y_train, X_test, y_test)

