import numpy as np
from load import DataLoad
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

label_E = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]


label2num = dict()
num2label = dict()
num = 0
for item in label_E:
    label2num[item] = num
    num2label[num] = item
    num = num+1

#将频次计数转换成概率分布
def freq2prob(d):
    '''
    输入一个频次字典，输出一个概率字典
    '''
    prob_dist = {}
    sum_freq = sum(d.values())
    for p,freq in d.items():
        if sum_freq != 0:
            prob_dist[p] = freq/sum_freq
    return prob_dist

MIN_COUNT = 1 #最小保留词的词频
class HMM:
    def __init__(self,train_sents,amount,vocabu):
        self.label_num = 9
        self.char_num = amount
        self.epsilon = 1e-50  # 无穷小量，防止归一化时分母为0
        self.a = None    # 状态转换概率矩阵
        self.b = None    # 发射概率矩阵
        self.pi = None   # 初始状态概率矩阵
        self.V = None    # 观测序列
        # self.pos = set()
        self.voc = vocabu # 出现过的词的集合
        self.word2num = dict()
        self.num2word = dict()
        self.setup(train_sents)

    def setup(self,train_sents):
        n = 0
        for v in self.voc:
            self.word2num[v] = n
            self.num2word[n] = v
            n += 1
        # print("老子要看这里")

        self.a = np.zeros((self.label_num,self.label_num))    # 状态转换概率矩阵
        self.b = np.zeros((self.label_num,self.char_num))    # 发射概率矩阵
        self.pi = np.zeros(self.label_num)   # 初始状态概率矩阵
        self.a += self.epsilon
        self.b += self.epsilon
        self.pi += self.epsilon
        print("矩阵初始化完成")

        pi_freq = defaultdict(int)
        transition_freq = {}
        emission_freq = {}
        for l in label_E:
            transition_freq[l] = defaultdict(int)
            emission_freq[l] = defaultdict(int)
        print("定义字典")
        for sent in train_sents:
            pi_freq[sent[0][1]] += 1   #统计句子开头各个状态数量
            # 记录训练集中的状态转移
            states_transition = [(p1[1],p2[1]) for p1,p2 in zip(sent,sent[1:])]
            # print("状态转移记录完成")
            for p1,p2 in states_transition:
                transition_freq[p1][p2] += 1
            #发射概率统计
            for w,p in sent:
                emission_freq[p][w] += 1

        print("3个转移字典完成-------------------------------------------------------")
        print("开始字典转概率")
        pi_freq = freq2prob(pi_freq)
        transition = {}
        for p, freq_dis in transition_freq.items():
            transition[p] = freq2prob(freq_dis)

        emission = {}
        for p, freq_dis in emission_freq.items():
            emission[p] = freq2prob(freq_dis)
        print("结束字典转概率")
        print("开始填三个矩阵")
        for p,q in pi_freq.items():
            self.pi[label2num[p]] = q
        for p,q in transition.items():
            for s,t in q.items():
                self.a[label2num[p],label2num[s]] = t
        for p,q in emission.items():
            for s,t in q.items():
                self.b[label2num[p],self.word2num[s]] = t
                # self.b[label2num[p], ord(s)] = t
        print("设置已完成")

    def viterbi(self,V):
        '''
        输入模型和观测序列
        输出概率最大的状态序列
        :param V:
        :param a:
        :param b:
        :param initial_distribution:
        :return:
        '''
        T = V.shape[0]
        M = self.a.shape[0]

        omega = np.zeros((T, M))
        omega[0, :] = np.log(self.pi * self.b[:, V[0]])

        prev = np.zeros((T - 1, M))

        for t in range(1, T):
            for j in range(M):
                # Same as Forward Probability
                probability = omega[t - 1] + np.log(self.a[:, j]) + np.log(self.b[j, V[t]])

                # This is our most probable state given previous state at time t (1)
                prev[t - 1, j] = np.argmax(probability)

                # This is the probability of the most probable state (2)
                omega[t, j] = np.max(probability)

        # Path Array
        S = np.zeros(T)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[T - 1, :])

        S[0] = last_state

        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

        # Flip the path array since we were backtracking
        S = np.flip(S, axis=0)

        # Convert numeric values to actual hidden states
        result = []
        for s in S:
            result.append(num2label[s])
        return result

if __name__=="__main__":
    load = DataLoad()
    filename_E = 'NER/English/train.txt'
    train_sents = load.load_E(filename_E) # 英文训练集
    # filename2_E = 'NER/English/validation.txt'
    filename2_E = r'C:\Users\84663\Desktop\english_test.txt'
    valid_sents_E = load.load_E(filename2_E)
    vocabu = set()
    for s in train_sents:
        for w, p in s:
            # vocab[w] += 1
            # pos.add(p)
            vocabu.add(w)

    for s in valid_sents_E:
        for w, p in s:
            vocabu.add(w)
    amount = len(vocabu)
    # print(amount)

    hmm = HMM(train_sents,amount,vocabu)

    valid = []
    right_label = []
    for sent in valid_sents_E:
        s = []
        # r = []
        for w,l in sent:
            # s.append(ord(w))
            s.append(hmm.word2num[w])
            right_label.append(l)
        s = np.array(s)
        valid.append(s)    # 预测值，按句子分

    label_predict = []
    for v in valid:
        label_predict.append(hmm.viterbi(v))
    num = len(right_label)
    # valid_sents是读取的要预测的验证集
    # label_predict是预测出来的标签


    txt = []
    length = len(valid_sents_E)
    for i in range(length):
        # print(len(valid_sents[i]))
        s = []
        length2 = len(valid_sents_E[i])
        for j in range(length2):
            text = valid_sents_E[i][j][0]
            label = label_predict[i][j]
            # label = num2label[label_predict[i][j]]
            s.append((text,label))
        txt.append(s)
    # print(txt)
    with open('test12.txt','w',encoding="utf-8") as f:
        for i in range(length-1):
            length2 = len(txt[i])
            for j in range(length2):
                f.write(str(txt[i][j][0])+" "+str(txt[i][j][1])+'\n')
            f.write('\n')
        length2 = len(txt[length - 1])
        for j in range(length2):
            f.write(str(txt[length - 1][j][0]) + " " + str(txt[length - 1][j][1]) + '\n')
        f.close()

