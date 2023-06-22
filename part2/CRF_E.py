import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore')

# 定义通用函数
def load_data(data_path):
    data_read_all = list()
    data_sent_with_label = list()
    with open(data_path, mode='r', encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                data_read_all.append(data_sent_with_label.copy())
                data_sent_with_label.clear()
            else:
                data_sent_with_label.append(list(line.strip().split(" ")))
    data_read_all.append(data_sent_with_label.copy())
    data_sent_with_label.clear()
    return data_read_all

# 定义特征函数
def word2features(sent, i):
    word = sent[i][0]
    #构造特征字典，我这里因为整体句子长度比较长，滑动窗口的大小设置的是6 在特征的构建中主要考虑了字的标识,是否是数字和字周围的特征信息
    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
    }
    #该字的前一个字
    if i > 0:
        word1 = sent[i-1][0]
        words = word1 + word
        features.update({
            '-1:word': word1,
            '-1:words': words,
            '-1:word.isdigit()': word1.isdigit(),
        })
    else:
        #添加开头的标识 BOS(begin of sentence)
        features['BOS'] = True
    #该字的前两个字
    if i > 1:
        word2 = sent[i-2][0]
        word1 = sent[i-1][0]
        words = word1 + word2 + word
        features.update({
            '-2:word': word2,
            '-2:words': words,
            '-3:word.isdigit()': word2.isdigit(),
        })
    #该字的前三个字
    if i > 2:
        word3 = sent[i - 3][0]
        word2 = sent[i - 2][0]
        word1 = sent[i - 1][0]
        words = word1 + word2 + word3 + word
        features.update({
            '-3:word': word3,
            '-3:words': words,
            '-3:word.isdigit()': word3.isdigit(),
        })
    #该字的后一个字
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        words = word1 + word
        features.update({
            '+1:word': word1,
            '+1:words': words,
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
    #若改字为句子的结尾添加对应的标识end of sentence
        features['EOS'] = True
    #该字的后两个字
    if i < len(sent)-2:
        word2 = sent[i + 2][0]
        word1 = sent[i + 1][0]
        words = word + word1 + word2
        features.update({
            '+2:word': word2,
            '+2:words': words,
            '+2:word.isdigit()': word2.isdigit(),
        })
    #该字的后三个字
    if i < len(sent)-3:
        word3 = sent[i + 3][0]
        word2 = sent[i + 2][0]
        word1 = sent[i + 1][0]
        words = word + word1 + word2 + word3
        features.update({
            '+3:word': word3,
            '+3:words': words,
            '+3:word.isdigit()': word3.isdigit(),
        })
    return features
#从数据中提取特征
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [ele[-1] for ele in sent]


if __name__=='__main__':
    # 读取数据

    train=load_data('NER/English/train.txt')
    valid=load_data('NER/English/validation.txt')
    print(train)
    print('训练集规模:',len(train))
    print('验证集规模:',len(valid))


    X_train = [sent2features(s) for s in train]
    y_train = [sent2labels(s) for s in train]
    X_dev = [sent2features(s) for s in valid]
    y_dev = [sent2labels(s) for s in valid]
    print(X_train[0])
    print(y_train[0])
    print("------------------------------------分割线------------------------------------")

    # 训练模型
    crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.25,c2=0.018,max_iterations=1000,
                                     all_possible_transitions=True,verbose=True)
    crf_model.fit(X_train, y_train)

    # 验证模型效果
    labels=list(crf_model.classes_)
    # labels.remove("O")                  #对于O标签的预测我们不关心，直接去掉
    y_pred = crf_model.predict(X_dev)
    metrics.flat_f1_score(y_dev, y_pred,
                          average='weighted', labels=labels)
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))

    # 保存模型
    import joblib
    joblib.dump(crf_model, "crf_model2_E_1000.joblib")
    print(y_dev)
    print(y_pred)


