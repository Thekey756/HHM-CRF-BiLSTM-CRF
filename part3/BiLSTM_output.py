import joblib
import json
import torch
import load
import torch.nn as nn
# from BiLSTM_CRF import BiLSTM_CRF
# import BiLSTM_CRF
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w,len(word_to_ix)) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


label = ['O','B-NAME', 'M-NAME', 'E-NAME', 'S-NAME','B-CONT',
         'M-CONT', 'E-CONT', 'S-CONT','B-EDU', 'M-EDU', 'E-EDU',
         'S-EDU','B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
         'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG','B-RACE', 'M-RACE',
         'E-RACE', 'S-RACE','B-PRO', 'M-PRO', 'E-PRO', 'S-PRO',
         'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC']
label_E = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]

START_TAG = "<START>"
STOP_TAG = "<STOP>"
# EMBEDDING_DIM = 5
# HIDDEN_DIM = 4

if __name__=='__main__':
    with open('word_to_ix.txt', 'r', encoding='utf-8') as f:
        word_to_ix = f.read()
        word_to_ix = json.loads(word_to_ix)

    ix_to_tag = {}
    for num in range(len(label)):
        ix_to_tag[num] = label[num]

    ix_to_tag[33] = START_TAG
    ix_to_tag[34] = STOP_TAG

    # model = joblib.load("BiLSTM_CRF_7.joblib")
    model = torch.load("BiLSTM_CRF_7.joblib")
    print(model)
    loader = load.DataLoad()
    # valid_data = loader.load2('NER/Chinese/validation.txt')
    valid_data = loader.load2(r'C:\Users\84663\Desktop\chinese_test.txt')
    # valid_data = loader.load2_E('./Project2/NER/English/validation.txt')
    # print(valid_data[0:2])
    print('================================================================')
    # valid_data2 = loader.load(r'C:\Users\84663\Desktop\chinese_test.txt')  # 用于制作txt文件
    valid_data2 = loader.load(r'C:\Users\84663\Desktop\chinese_test.txt')
    # print(valid_data2[0:2])

    label_predict = []
    tmp = []
    with torch.no_grad():
        length = len(valid_data)
        for i in range(length):
            precheck_sent = prepare_sequence(valid_data[i][0], word_to_ix)
            # print(model(precheck_sent)[1])
            tags = model(precheck_sent)[1]
            # print(tags)
            tmp = [ix_to_tag[t] for t in tags]
            label_predict.append(tmp)           # 收集预测值
            tmp = []

    txt = []
    length = len(valid_data2)
    for i in range(length):
        s = []
        length2 = len(valid_data2[i])
        for j in range(length2):
            text = valid_data2[i][j][0]
            label = label_predict[i][j]
            s.append((text, label))
        txt.append(s)

    with open('test31.txt', 'w', encoding="utf-8") as f:
        for i in range(length):
            length2 = len(txt[i])
            for j in range(length2):
                f.write(str(txt[i][j][0]) + " " + str(txt[i][j][1]) + '\n')
            f.write('\n')
        f.close()
