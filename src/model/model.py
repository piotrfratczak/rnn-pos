import os
import torch
import stanza
import pathlib
import torch.nn as nn
from pymagnitude import FeaturizerMagnitude

from src.utils.tagger import FullTagger, UniversalTagger
from src.model.abstract_model import AbstractClassifier, AbstractDynamicGraphClassifier


class ClassifierSingleLstmCell(AbstractClassifier):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, seq_len):
        super(ClassifierSingleLstmCell, self).__init__(
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


class ClassifierLstmLayer(AbstractClassifier):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, n_layers, seq_len):
        super(ClassifierLstmLayer, self).__init__(
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


class ClassifierConcatPennLstm(ClassifierLstmLayer):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, n_layers, index_mapper, seq_len):
        super(ClassifierConcatPennLstm, self).__init__(
            vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob, n_layers, seq_len)

        pos_vector_dim = FeaturizerMagnitude(100, namespace='PartsOfSpeech').dim
        self.lstm = nn.LSTM(embedding_size + pos_vector_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.index_mapper = index_mapper
        self.tagger = FullTagger()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        tags = []
        for batch in range(batch_size):
            indices_list = x.tolist()[batch]
            list_of_words = self.index_mapper.indices_to_words(indices_list)
            list_of_tags = self.tagger.map_sentence(list_of_words)
            tags.append(list_of_tags)
        pos_vectors = FeaturizerMagnitude(100, namespace='PartsOfSpeech').query(tags)

        embeds = self.embedding(x)
        lstm_input = torch.cat([embeds, torch.tensor(pos_vectors)], dim=2)

        lstm_out, hidden = self.lstm(lstm_input, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]
        return out, hidden

    def __repr__(self):
        return "LSTMConcatPenn"


class ClassifierConcatDependencyLstm(ClassifierLstmLayer):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, n_layers, seq_len):
        super(ClassifierConcatDependencyLstm, self).__init__(
                vocab_size, output_size, embedding_matrix, embedding_size,
                hidden_dim, device, drop_prob, n_layers, seq_len)

        add_vector_dim = FeaturizerMagnitude(100, namespace='PartsOfSpeech').dim +\
            FeaturizerMagnitude(100, namespace='SyntaxDependencies').dim
        self.lstm = nn.LSTM(embedding_size + add_vector_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        stanza_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'data/stanza')
        self.parser = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True, dir=stanza_dir)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        documents = []
        for batch in range(batch_size):
            indices_list = x.tolist()[batch]
            list_of_words = self.index_mapper.indices_to_words(indices_list)
            sentence = ' '.join(list_of_words)
            doc = stanza.Document([], text=sentence)
            documents.append(doc)
        out_docs = self.parser(documents)

        batch_tags = []
        batch_deps = []
        for doc in out_docs:
            tags = []
            deps = []
            for sent in doc.sentences:
                for word in sent.words:
                    tags.append(word.xpos)
                    deps.append(word.deprel)
            batch_tags.append(tags[:self.seq_len])
            batch_deps.append(deps[:self.seq_len])
        pos_vectors = FeaturizerMagnitude(100, namespace='PartsOfSpeech').query(batch_tags)[:, :self.seq_len, :]
        deps_vectors = FeaturizerMagnitude(100, namespace='SyntaxDependencies').query(batch_deps)[:, :self.seq_len, :]

        embeds = self.embedding(x)
        lstm_input = torch.cat([embeds, torch.tensor(pos_vectors), torch.tensor(deps_vectors)], dim=2)

        lstm_out, hidden = self.lstm(lstm_input, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]
        return out, hidden

    def __repr__(self):
        return "LSTMConcatDependency"


class ClassifierConcatUniversalLstm(ClassifierConcatPennLstm):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, n_layers, index_mapper, seq_len):
        super(ClassifierConcatUniversalLstm, self).__init__(
            vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob, n_layers,
            index_mapper, seq_len)

        self.tagger = UniversalTagger()

    def __repr__(self):
        return "LSTMConcatUniversal"


class ClassifierLstmPenn(AbstractDynamicGraphClassifier):

    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, index_mapper, seq_len):
        super(ClassifierLstmPenn, self).__init__(vocab_size, output_size, embedding_matrix, embedding_size,
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


class ClassifierLstmUniversal(AbstractDynamicGraphClassifier):

    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, index_mapper, seq_len):
        super(ClassifierLstmUniversal, self).__init__(
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
