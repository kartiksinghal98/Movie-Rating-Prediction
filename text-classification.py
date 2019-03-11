import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer



sw=set(stopwords.words('english'))
regex=RegexpTokenizer(r"\w+")
lem=WordNetLemmatizer()
ps=PorterStemmer()



def myTokenizer(sentence):
    sentence=sentence.replace("<br />"," ")
    sentence=sentence.replace("\n","")
    sentence=sentence.lower()
    sentence=regex.tokenize(sentence)
    new=[w for w in sentence if w not in sw]
    final=[lem.lemmatize(w) for w in new]
    final=[ps.stem(w) for w in new]
    #final=" ".join(final)
    return final


X_train=[]
with open("imdb_trainX.txt") as f:
    for x in f:
        x=myTokenizer(x)
        X_train.append(x)

Y_train=[]
with open('imdb_trainY.txt') as f:
    for x in f:
        x=x.replace("\n","")
        x=int(x)
        Y_train.append(x)
Y_train=np.array(Y_train)

X_test=[]
with open("imdb_testX.txt") as f:
    for x in f:
        x=myTokenizer(x)
        X_test.append(x)


Y_test=[]
with open('imdb_testY.txt') as f:
    for x in f:
        x=x.replace("\n","")
        x=int(x)
        Y_test.append(x)
Y_test=np.array(Y_train)



# print(len(X_test))
# print(len(Y_test))
# print(len(X_train))
# print(len(X_train))

def countVec(X_train,Y_train):
    vocab=[]
    N=0
    Nc=dict.fromkeys(np.unique(Y_train),0 )
    WordCount_category=dict.fromkeys(np.unique(Y_train),0 )

    for i in range(len(X_train)):
 
        N+=1
        Nc[Y_train[i]]+=1
        for j in X_train[i]:
          
            WordCount_category[Y_train[i]]+=1
            if (j not in vocab):
                vocab.append(j)
              
    Eachword=dict.fromkeys(np.unique(Y_train),dict.fromkeys(vocab,0 ))
    for i in range(len(X_train)):
        for j in X_train[i]:
            Eachword[X_train[i]][j]+=1
    return vocab,N,Nc,WordCount_category,Eachword
    # vector=np.zeros((len(X_train),len(vocab)))
    # for i in range(len(tokens)):
    #     for j in tokens[i]:        
    #         vector[i,vocab[j]]+=1

vocab,N,Nc,WordCount_category,Eachword=countVec(X_train,Y_train)

# print(N)
# print(Nc)
# print(Eachword[1])

def i_wordprob(word,label,WordCount_category,Eachword,vocab):
    C_num=1
    vocab_len=len(vocab)
    
    if(word in Eachword[label]):
        C_num=Eachword[label][word]+1
        
    C_deno=WordCount_category[label]+vocab_len
    
    return C_num/float(C_deno)

def likelihood(X_test,label,WordCount_category,Eachword,vocab):

    likely=1
    for j in X_test:
        likely*=i_wordprob(j,label,WordCount_category,Eachword,vocab)
    return likely

def MultinomialNB(X_train,Y_train,X_test,WordCount_category,Eachword,vocab,Nc,n):
    label=np.unique(Y_train)
    c_score=[]
    for i in label:
        prior=Nc[i]/n
        like=likelihood(X_test,i,WordCount_category,Eachword,vocab)
        c_score.append(prior*float(like))

    final_score=label[np.argmax(c_score)]
    return final_score


def accuracy(X_train,Y_train,X_test,Y_test,WordCount_category,Eachword,vocab,Nc,N):
    count=0
    for i in range(Y_test.shape[0]):
        prediction=MultinomialNB(X_train,Y_train,X_test[i],WordCount_category,Eachword,vocab,Nc,N)
        print(prediction)
        if(prediction==Y_test[i]):
            count+=1
    return (count/Y_test.shape[0])*100


print(accuracy(X_train,Y_train,X_test,Y_test,WordCount_category,Eachword,vocab,Nc,N))






