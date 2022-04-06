import torch.nn as nn

from src.model.abstract_model import AbstractSpamClassifier, \
    AbstractSpamClassifierWithTaggerIndexMapperAndDynamicGraph
from src.tagger import FullTagger, UniversalTagger


class SpamClassifierSingleLstmCell(AbstractSpamClassifier):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, seq_len):
        super(SpamClassifierSingleLstmCell, self).__init__(
            vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob, seq_len)

        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_dim)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        embeds = self.embedding(x)

        for j in range(self.seq_len):
            cell_input = embeds[:, j].view(batch_size, self.embedding_size)
            hidden = self.lstm_cell(cell_input, hidden)

        lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]
        return out, hidden

    def __repr__(self):
        return "SingleLSTMCell"


class SpamClassifierLstmLayer(AbstractSpamClassifier):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, n_layers, seq_len):
        super(SpamClassifierLstmLayer, self).__init__(
            vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob, seq_len)

        self.n_layers = n_layers
        self.lstm = nn.LSTM(embedding_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]
        return out, hidden

    def __repr__(self):
        return "LSTMLayer"


class SpamClassifierLstmPosPenn(AbstractSpamClassifierWithTaggerIndexMapperAndDynamicGraph):

    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, index_mapper, seq_len):
        super(SpamClassifierLstmPosPenn, self).__init__(vocab_size, output_size, embedding_matrix, embedding_size,
                                                        hidden_dim, device, drop_prob, index_mapper, seq_len)

        self.lstm_cell_vbn = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbz = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbg = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbd = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_md = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nn = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nnps = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nnp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nns = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_jjs = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_jjr = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_jj = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rb = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rbr = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rbs = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_EMPTY = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_cd = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_in = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_pdt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_cc = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_ex = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_pos = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_fw = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_dt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_uh = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_to = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_prp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_prp_dollar = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_dollar = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wp_dollar = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wdt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wrb = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_other = nn.LSTMCell(embedding_size, hidden_dim)

        self.tagger = FullTagger()

    @property
    def lstm_mapping(self):
        return {
            'VBN': self.lstm_cell_vbn,
            'VBZ': self.lstm_cell_vbz,
            'VBG': self.lstm_cell_vbg,
            'VBP': self.lstm_cell_vbp,
            'VBD': self.lstm_cell_vbd,
            'MD': self.lstm_cell_md,
            'NN': self.lstm_cell_nn,
            'NNPS': self.lstm_cell_nnps,
            'NNP': self.lstm_cell_nnp,
            'NNS': self.lstm_cell_nns,
            'JJS': self.lstm_cell_jjs,
            'JJR': self.lstm_cell_jjr,
            'JJ': self.lstm_cell_jj,
            'RB': self.lstm_cell_rb,
            'RBR': self.lstm_cell_rbr,
            'RBS': self.lstm_cell_rbs,
            'EMPTY': self.lstm_cell_EMPTY,
            'CD': self.lstm_cell_cd,
            'IN': self.lstm_cell_in,
            'PDT': self.lstm_cell_pdt,
            'CC': self.lstm_cell_cc,
            'EX': self.lstm_cell_ex,
            'POS': self.lstm_cell_pos,
            'RP': self.lstm_cell_rp,
            'FW': self.lstm_cell_fw,
            'DT': self.lstm_cell_dt,
            'UH': self.lstm_cell_uh,
            'TO': self.lstm_cell_to,
            'PRP': self.lstm_cell_prp,
            'PRP$': self.lstm_cell_prp_dollar,
            '$': self.lstm_cell_dollar,
            'WP': self.lstm_cell_wp,
            'WP$': self.lstm_cell_wp_dollar,
            'WDT': self.lstm_cell_wdt,
            'WRB': self.lstm_cell_wrb,
            'OTHER': self.lstm_cell_other
        }

    def __repr__(self):
        return "PosPenn"


class SpamClassifierLstmPosUniversal(AbstractSpamClassifierWithTaggerIndexMapperAndDynamicGraph):

    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, index_mapper, seq_len):
        super(SpamClassifierLstmPosUniversal, self).__init__(
            vocab_size, output_size, embedding_matrix, embedding_size,
            hidden_dim, device, drop_prob, index_mapper, seq_len)

        self.lstm_cell_empty = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_adj = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_adp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_adv = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_conj = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_det = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_noun = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_num = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_prt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_pron = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_verb = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_other = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_x = nn.LSTMCell(embedding_size, hidden_dim)

        self.tagger = UniversalTagger()

    @property
    def lstm_mapping(self):
        return {
            'EMPTY': self.lstm_cell_empty,
            'ADJ': self.lstm_cell_adj,
            'ADP': self.lstm_cell_adp,
            'ADV': self.lstm_cell_adv,
            'CONJ': self.lstm_cell_conj,
            'DET': self.lstm_cell_det,
            'NOUN': self.lstm_cell_noun,
            'NUM': self.lstm_cell_num,
            'PRT': self.lstm_cell_prt,
            'PRON': self.lstm_cell_pron,
            'VERB': self.lstm_cell_verb,
            '.': self.lstm_cell_other,
            'X': self.lstm_cell_x,
        }

    def __repr__(self):
        return "PosUni"
