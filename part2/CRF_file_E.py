import joblib
import CRF


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





if __name__=='__main__':
    # valid = load_data('NER/English/validation.txt')
    valid = load_data(r'C:\Users\84663\Desktop\english_test.txt')
    CRFModel = joblib.load("crf_model2_E.joblib")
    X_dev = [CRF.sent2features(s) for s in valid]
    labels = list(CRFModel.classes_)
    # labels.remove("O")  # 对于O标签的预测我们不关心，就直接去掉
    y_pred2 = CRFModel.predict(X_dev)

    txt = []
    length = len(valid)
    for i in range(length):
        # print(len(valid_sents[i]))
        s = []
        length2 = len(valid[i])
        for j in range(length2):
            text = valid[i][j][0]
            label = y_pred2[i][j]
            s.append((text, label))
        txt.append(s)
    # print(txt)
    # with open('part2_E.txt', 'w', encoding="utf-8") as f:
    #     for i in range(length):
    #         length2 = len(txt[i])
    #         for j in range(length2):
    #             f.write(str(txt[i][j][0]) + " " + str(txt[i][j][1]) + '\n')
    #         f.write('\n')
    #     f.close()
    with open('test22.txt','w',encoding="utf-8") as f:
        for i in range(length-1):
            length2 = len(txt[i])
            for j in range(length2):
                f.write(str(txt[i][j][0])+" "+str(txt[i][j][1])+'\n')
            f.write('\n')
        length2 = len(txt[length - 1])
        for j in range(length2):
            f.write(str(txt[length - 1][j][0]) + " " + str(txt[length - 1][j][1]) + '\n')
        f.close()