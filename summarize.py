import nltk
nltk.data.path.append('nltk_data')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pre_process

# Get vector representation
def getVectorSpace(cleanSet):
	vocab = {}
	for data in cleanSet:
		for word in data.split():
			vocab[data] = 0
	return vocab.keys()
	
# Calculate Cosine Similarity
def calculateSimilarity(sentence, doc):
	if doc == []:
		return 0
	vocab = {}
	for word in sentence:
		vocab[word] = 0
	
	docInOneSentence = ''
	for t in doc:
		docInOneSentence += (t + ' ')
		for word in t.split():
			vocab[word]=0	
	
	cv = CountVectorizer(vocabulary=vocab.keys())

	docVector = cv.fit_transform([docInOneSentence])
	sentenceVector = cv.fit_transform([sentence])
	return cosine_similarity(docVector, sentenceVector)[0][0]
	

# Main method
def main(texts):

	sentences = []
	clean = []
	originalSentenceOf = {}

	print('	 Optimising Sentences')

	#Data cleansing
	for part in texts:
		cl = pre_process.stem(part)
		sentences.append(part)
		clean.append(cl)
		originalSentenceOf[cl] = part		
	setClean = set(clean)
			
	#calculate Similarity score each sentence with whole documents		
	scores = {}
	for data in clean:
		temp_doc = setClean - set([data])
		score = calculateSimilarity(data, list(temp_doc))
		scores[data] = score

	# Rank sentences by cosine score
	selected = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

	print('	 Removing Redundant Sentences')
	# Remove sentences with cosine score less than 0.25
	summarySet = [sentence for sentence, key in selected.items() if key > 0.25]

	final_summary = []
	for sentence in summarySet:
		final_summary.append(originalSentenceOf[sentence].lstrip(' '))
	return final_summary

def processor(texts, num_sentences):

	sentences = []
	clean = []
	originalSentenceOf = {}

	#Data cleansing
	for part in texts:
		cl = pre_process.stem(part)
		sentences.append(part)
		clean.append(cl)
		originalSentenceOf[cl] = part		
	setClean = set(clean)
			
	#calculate Similarity score each sentence with whole documents		
	scores = {}
	for data in clean:
		temp_doc = setClean - set([data])
		score = calculateSimilarity(data, list(temp_doc))
		scores[data] = score

	print('	 Re-ranking Sentences')
	# Rank sentences by cosine score
	selected = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

	# Get the top scored sentences (High cosine score)
	summarySet = []
	for i, sentence in enumerate(selected.keys()):
		if i != num_sentences: summarySet.append(sentence)
		else: break

	final_summary = []
	for sentence in summarySet:
		final_summary.append(originalSentenceOf[sentence].lstrip(' '))
		
	return final_summary