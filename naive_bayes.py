import numpy as np
from Utils.data_utils import Data
from tqdm import tqdm
import pickle as pk
from collections import Counter
from math import log
import argparse

class Model:
	def __init__(self, data, n = 3, Lambda = 1.0, model_name = 'new_tri'):
		"""Initializes an ngram naive bayes model with Add One Smoothing by default."""
		self.n = n
		self.model_name = model_name
		self.save_path = 'Params/model_{}'.format(self.model_name)
		self.Lambda = Lambda
		self.data = data
		self.prior = dict([(lang,0.0) for lang in self.data.classes])
		self.N = float(len(list(self.data.data_iterator('train'))))
		self.language_specific_ngrams_total = dict()
		self.ngram_count = dict()

	def get_ngrams(self, x):
		"""Returns a list of character ngrams for the given document(string)."""
		ngram_list = []
		for i in range(len(x)):
			for n in range(self.n):
				if i >= n:
					gm = x[i-n:i+1]
					ngram_list.append(gm)
		return ngram_list

	def update_language_ngrams(self, x, y):
		"""Updates count of ngrams for the specified language."""
		ngrams_list = self.get_ngrams(x)
		self.language_specific_ngrams_total[y] = len(ngrams_list)
		self.ngram_count[y] = Counter(ngrams_list)

	def train(self):
		"""Trains the model, by updating the counts for the given dataset."""
		print('training ...')
		for language in tqdm(self.data.classes):
			doc = ''
			n_examples = 0 
			for x,y in self.data.data_iterator('train',y = language):
				doc += x
				n_examples += 1
			self.prior[language] = n_examples/self.N
			self.update_language_ngrams(doc, language)
		self.save_params()

	def get_probability(self, y, ngram_list):
		"""Returns unnormalised probability for a given ngram_list and language."""
		p = 0.0
		for ngram in ngram_list:
			likelihood = (self.ngram_count[y][ngram] + self.Lambda )/ ( self.language_specific_ngrams_total[y] + self.Lambda*len(self.ngram_count[y]) )
			log_likelihood = log( likelihood )
			p += log_likelihood
		p = p+log(self.prior[y])
		return p

	def predict(self, doc, language_set=None):
		"""Returns the predicted label for the docment."""
		if language_set is None:
			language_set = list(self.data.classes)
		doc = doc.strip()
		ngram_list = self.get_ngrams(doc)
		probabilities = []
		for language in language_set:
			p = self.get_probability(language, ngram_list)
			probabilities.append( p )
		return language_set[np.argmax(probabilities)]

	def save_params(self):
		"""Saves the parameters of the model"""
		print('saving ...')
		params = [self.prior, self.language_specific_ngrams_total, self.ngram_count]		
		with open(self.save_path,'wb') as f:
			pk.dump(params, f)

	def load_params(self):
		"""Loads the parameters of the model"""
		print('loading ...')
		with open(self.save_path,'rb') as f:
			self.prior, self.language_specific_ngrams_total, self.ngram_count = pk.load(f)

	def evaluate(self, segment='train', print_labels = False, language_set = None):
		"""Evaluates the model and returns the accuracy score. Keep languege_set None to evaluate on all the 235 languages"""
		assert segment in ['train','test','dev']
		print('evaluating {} set...'.format(segment))
		total, correct = 0.0, 0.0
		if language_set is None:
			language_set = list(self.data.classes)
		language_allowed = dict([(l,True) if l in language_set else (l,False) for l in list(self.data.classes)])
		for i,ex in enumerate(self.data.data_iterator(segment)):
			x,y = ex
			if language_allowed[y]:
				y_predicted = '-'
				y_predicted = self.predict(x.strip(), language_set)
				total += 1
				if print_labels:
					print(i, y_predicted, y)
				if y_predicted == y:
					correct += 1
		return correct/total

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', help = 'n as in n-grams (default set to 3)', default = 3, type = int)
	parser.add_argument('--Lambda', help = 'pseudo count in additive smoothing. (default set to 1)', default = 1.0, type = float)
	parser.add_argument('--model_name', help = 'name of the model to save the parameters (default set to trigram)', default = 'new_trigram', type = str)
	parser.add_argument('--enable_train', help = 'to train the model and evaluate the model, otherwise it only evaluates the pretrained model.', action='store_true')
	parser.add_argument('--enable_language_restriction', help = 'To evaluate on selected 6 selected languages (German, French, English, Dutch, Italian, Spanish) instead all 235 languages', action='store_true')
	args = parser.parse_args()
	
	d = Data()
	m = Model(d, n = args.n, Lambda= args.Lambda, model_name = args.model_name)
	if args.enable_train:
		m.train()
	m.load_params()
	print('Evaluating...')
	language_set = None
	if args.enable_language_restriction:
		language_set = ['fra','eng','nld','spa','ita','deu']
	# print('Accuracy DEV set:',m.evaluate('dev', print_labels = True, language_set = language_set ))
	# print('Accuracy TEST set:',m.evaluate('test', print_labels = True, language_set = language_set ))
	hindi_doc = 'विकिपीडिया सभी विषयों पर प्रामाणिक और उपयोग, परिवर्तन व पुनर्वितरण के लिए स्वतन्त्र ज्ञानकोश बनाने का एक बहुभाषीय प्रकल्प है। यह यथासम्भव निष्पक्ष दृष्टिकोण वाली सूचना प्रसारित करने के लिए कृतसंकल्प है। सर्वप्रथम अंग्रेज़ी विकिपीडिया जनवरी 2001 में आरम्भ किया गया '
	print(m.predict(hindi_doc))




