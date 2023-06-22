import numpy as np
from load import DataLoad
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

#33种标签
label = ['O','B-NAME', 'M-NAME', 'E-NAME', 'S-NAME','B-CONT',
         'M-CONT', 'E-CONT', 'S-CONT','B-EDU', 'M-EDU', 'E-EDU',
         'S-EDU','B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
         'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG','B-RACE', 'M-RACE',
         'E-RACE', 'S-RACE','B-PRO', 'M-PRO', 'E-PRO', 'S-PRO',
         'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC']
label_E = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]

label2num = dict()
num2label = dict()
num = 0
for item in label:
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

class HMM:
    def __init__(self,train_sents):
        self.label_num = 33
        self.char_num = 65535
        self.epsilon = 1e-50  # 无穷小量，防止归一化时分母为0
        self.a = None    # 状态转换概率矩阵
        self.b = None    # 发射概率矩阵
        self.pi = None   # 初始状态概率矩阵
        self.V = None    # 观测序列
        self.pos = set()
        self.voc = set()
        self.word2num = dict()
        self.num2word = dict()
        self.setup(train_sents)

    def setup(self,train_sents):
        vocab = defaultdict(int)  # 词-词频字典
        # voc = set()  # 出现过的词的集合
        pos = set()  # 词性集合
        for s in train_sents:
            for w, p in s:
                vocab[w] += 1
                pos.add(p)

        self.a = np.zeros((self.label_num,self.label_num))    # 状态转换概率矩阵
        self.b = np.zeros((self.label_num,self.char_num))    # 发射概率矩阵
        self.pi = np.zeros(self.label_num)   # 初始状态概率矩阵
        self.a += self.epsilon
        self.b += self.epsilon
        self.pi += self.epsilon

        pi_freq = defaultdict(int)
        transition_freq = {}
        emission_freq = {}
        for l in label:
            transition_freq[l] = defaultdict(int)
            emission_freq[l] = defaultdict(int)
        for sent in train_sents:
            pi_freq[sent[0][1]] += 1   # 统计句子开头各个状态数量
            # 记录训练集中的状态转移
            states_transition = [(p1[1],p2[1]) for p1,p2 in zip(sent,sent[1:])]

            for p1,p2 in states_transition:
                transition_freq[p1][p2] += 1
            # 发射概率统计
            for w,p in sent:
                emission_freq[p][w] += 1
                self.pos.add(p)
            for p1 in label:
                for p2 in label:
                    if p2 not in transition_freq[p1]:
                        transition_freq[p1][p2] = 0
            for p1 in label:
                for p2 in self.voc:
                    emission_freq[p1][p2] = 0
        # self.pi = freq2prob(pi_freq)
        pi_freq = freq2prob(pi_freq)
        transition = {}
        for p, freq_dis in transition_freq.items():
            transition[p] = freq2prob(freq_dis)

        emission = {}
        for p, freq_dis in emission_freq.items():
            emission[p] = freq2prob(freq_dis)

        for p,q in pi_freq.items():
            self.pi[label2num[p]] = q
        for p,q in transition.items():
            for s,t in q.items():
                self.a[label2num[p],label2num[s]] = t
        for p,q in emission.items():
            for s,t in q.items():
                # self.b[label2num[p],self.word2num[s]] = t
                self.b[label2num[p], ord(s)] = t
    def show(self):
        print(self.pi)
        print("="*50)
        print(self.a)
        print("=" * 50)
        print(self.b)
        print("=" * 50)

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
        # print("----------------------",V[0])
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
    filename = 'NER/Chinese/train.txt'
    train_sents = load.load(filename)   # 中文训练集
    hmm = HMM(train_sents)
    # hmm.show()
    # filename2 = 'NER/Chinese/validation.txt'
    filename2 = r'C:\Users\84663\Desktop\chinese_test.txt'
    valid_sents = load.load(filename2)
    valid = []
    rigt_label = []
    for sent in valid_sents:
        s = []
        for w,l in sent:
            s.append(ord(w))
            rigt_label.append(l)
        s = np.array(s)
        valid.append(s)

    label_predict = []
    for v in valid:
        label_predict.append(hmm.viterbi(v))
        print(hmm.viterbi(v))
    num = len(rigt_label)
    # valid_sents是读取出来要预测的验证集
    # label_predict是预测出来的标签

    num2 = 0
    myans = []
    for sent in label_predict:
        num2 += len(sent)
        for s in sent:
            myans.append(s)
    print(num)
    print(num2)
    score = 0
    for i in range(len(myans)):
        if(myans[i] == rigt_label[i]):
            score += 1
    print('{}/{} = {}'.format(score,len(myans),score/len(myans)))
    print(score/len(myans),)

    txt = []
    length = len(valid_sents)
    for i in range(length):
        # print(len(valid_sents[i]))
        s = []
        length2 = len(valid_sents[i])
        for j in range(length2):
            text = valid_sents[i][j][0]
            label = label_predict[i][j]
            s.append((text,label))
        txt.append(s)

    with open('test11.txt','w',encoding="utf-8") as f:
        for i in range(length):
            length2 = len(txt[i])
            for j in range(length2):
                f.write(str(txt[i][j][0])+" "+str(txt[i][j][1])+'\n')
            f.write('\n')
        f.close()

