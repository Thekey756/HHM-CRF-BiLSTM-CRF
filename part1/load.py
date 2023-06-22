class DataLoad:
    def __init__(self):
        pass
    def load(self,filepath):
        sentences = []
        with open(filepath, encoding='utf-8') as f:
            # text = []
            # label = []
            sent = []
            for line in f:
                if len(line)>1:
                    line = line.split(' ')
                    text = line[0]
                    label = line[1][0:-1]
                    sent.append((text,label))
                else:
                    # sent.append(text)
                    # sent.append(label)
                    sentences.append(sent)
                    sent = []
                    # text = []
                    # label = []

        return sentences
    def load_E(self,filepath):
        sentences = []
        with open(filepath, encoding='utf-8') as f:
            # text = []
            # label = []
            sent = []
            for line in f:
                if len(line)>1:
                    line = line.split(' ')
                    text = line[0]
                    label = line[1][0:-1]
                    sent.append((text,label))
                else:
                    # sent.append(text)
                    # sent.append(label)
                    sentences.append(sent)
                    sent = []
                    # text = []
                    # label = []
        sentences.append(sent)
        return sentences

    #part3使用的数据读取函数
    def load2(self,filepath):
        sentences = []
        with open(filepath,encoding='utf-8') as f:
            text = []
            label = []
            sent = []
            for line in f:
                print(line)
                if len(line)>1:
                    line = line.split(' ')
                    text.append(line[0])
                    label.append(line[1][0:-1])
                    # line = line.strip().split(' ')
                    # text.append(line[0])
                    # label.append(line[1])
                    # sent.append((text,label))
                else:
                    # sent.append(text)
                    # sent.append(label)
                    sentences.append((text,label))
                    sent = []
                    text = []
                    label = []

        return sentences
    def load2_E(self,filepath):
        sentences = []
        with open(filepath,encoding='utf-8') as f:
            text = []
            label = []
            sent = []
            for line in f:
                print(line)
                if len(line)>1:
                    line = line.split(' ')
                    text.append(line[0])
                    label.append(line[1][0:-1])
                    # line = line.strip().split(' ')
                    # text.append(line[0])
                    # label.append(line[1])
                    # sent.append((text,label))
                else:
                    # sent.append(text)
                    # sent.append(label)
                    sentences.append((text,label))
                    sent = []
                    text = []
                    label = []
        sentences.append((text,label))
        return sentences




if __name__=='__main__':
    path = r'NER\English\validation.txt'
    D=DataLoad()
    sents= D.load_E(path)
    print(sents)
    # for s,t in sents[-1]:
    #     print(s)
    #     print(t)
    print('dadadadadadadada')


