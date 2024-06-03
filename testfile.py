
#importujem sve biblioteke koje su potrebne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# data predstavlja kao String koji sadrzi podatke o stanju piksela za pojedine slike brojeva
data = pd.read_csv(r"C:\Users\maril\OneDrive\Desktop\brojevi\train.csv")

# Napravimo numpy array umjesto obicnog array-a, posto se oni ponasaju 50x brze nego obicne python liste
data = np.array(data)
# Shape se koristi da se dohvate dimenzije numPy ili panda array-a, u ovom slucaju, m i n predstavljaju broj redova i kolona nase matrice
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets


data_dev = data[0:1000].T #transponovanje se radi tako da svaka kolona predstavlja jedan primjer, umijesto svaki red, a kao data_dev
#predstavljamo prvih 1000 primjeraka
Y_dev = data_dev[0] #prvi red 
X_dev = data_dev[1:n] #ovdje ce se sadrzati svi primjeri
X_dev = X_dev / 255. 

data_train = data[1000:m].T #podaci nad kojima vrsimo trening po istom principu kao gore
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


#uvodimo parametre tezina (weights) i "bias-a". Potrebno poznavanje matematicke teorije iza ovoga
def init_params(): 
    #random.rand kreira array datog oblika i popunjava ga sa vrijednostima "uniformne distribucije" od 0 - 1, 
    W1 = np.random.rand(10, 784) - 0.5 #10, 784 zato sto imamo 784 ulaza u odnosu na 784 ukupna pixela po slici, a 10 predstavlja sledecih 10 ulaza prvog sloja
    b1 = np.random.rand(10, 1) - 0.5 
    W2 = np.random.rand(10, 10) - 0.5 #sa prbog sloja na drugi prelazimo sa 10 na 10
    b2 = np.random.rand(10, 1) - 0.5
    #od svakog izlaza se oduzima 0.5 jer zelimo da premjestimo vrijednost u zonu od -0.5 do 0.5
    return W1, b1, W2, b2 #vracamo date parametre

#ovo predstavlja funkciju koja vraca x ako je x > 0 i koja vraca 0  ako je x <= 0
def ReLU(Z):
    return np.maximum(Z, 0)

#ovo predstavlja datu exponencijalnu funkciju koja je pre kompleksna da je ja sad ovdje objasnjavam
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    

def forward_prop(W1, b1, W2, b2, X):
    #za prvi sloj...za dati cvor sabiramo sve tezine koji vode ka cvoru sledeceg sloja, u ovom slucaju je to predstavljeno
    #kao mnozenje dvije matrice, tome i dot funkcija, na kraju dodajemo bias za dati cvor
    Z1 = W1.dot(X) + b1 
    #A1 racunamo po ReLU funkciji za prethodni rezultat
    A1 = ReLU(Z1)
    #analogno onome gore
    Z2 = W2.dot(A1) + b2
    #primjenjujemo sledeci algoritam
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#izvod gore navedene funkcije
def ReLU_deriv(Z):
    return Z > 0

#kako bi nastavili dalje, moramo izvrsiti odredjene izmjene, a jedna od tih je predsavljanje preko kolonske matrice
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) #kreira se matrica adekvatne velicine
    one_hot_Y[np.arange(Y.size), Y] = 1 #prolazi kroz matricu
    one_hot_Y = one_hot_Y.T #Prevrtanje matrice, odnosno transponovanje
    return one_hot_Y

#definisanje svih potrebnih racunskih operacija
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y) #primjena gore datog objasnjenja
    dZ2 = A2 - one_hot_Y 
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

#updejtujemo parametre preko datih formula, gdje alfa predstavlja ratu ucenja
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)
#Definisemo tacnost sa kojom radi mreza
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#predstavlja iterativni algoritam za pronalazak mximuama/minmuma dat funkcije
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

#Inicijalizujemo parametre
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
#Konacna metoda za testiranje parametara
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
test_prediction(4, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)
