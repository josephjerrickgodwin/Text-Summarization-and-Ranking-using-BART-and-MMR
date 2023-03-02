import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nltk
nltk.data.path.append('nltk_data')
from nltk.corpus import stopwords

def main(
    input_text: str, 
) -> str:
    """
    Preprocess the input for Text Summarization

    :param input_text: words to preprocess.
    """
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    print('	 Removing Special Characters')
    characters_to_remove = "`!()_[]=+~{}@?;<>:|\n"
    for character in characters_to_remove:
        input_text = input_text.replace(character, " ")
        input_text = input_text.replace("   ", " ")
        input_text = input_text.replace("  ", " ")
        input_text = input_text.replace("''", "")
    
    return input_text.strip()

def stem(input_text: str,
    stemmer: bool = True, 
    remove_stop_words: bool = True, 
    ) -> str:
    
    """
    :param input_text: words to preprocess.
    :param stemmer: Whether or not to stem words.
    :param remove_stop_words: Whether or not to remove stop words.
    :param lemmitizer: Whether or not to lemmitize words.
    :return: base words and preprocessed words.
    """

    # Re-clean the words
    characters_to_remove = "',"
    for character in characters_to_remove:
        input_text = input_text.replace(character, " ")
        input_text = input_text.replace("  ", " ")
        input_text = input_text.replace("-", " ")
    lst_text = input_text.split()

    # remove Stopwords
    if remove_stop_words == True:
        stop_words = set(stopwords.words('english'))
        lst_text = [word for word in lst_text if word not in stop_words]
    
    # Stem Words
    if stemmer == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
            
    # back to string from list
    stemmed_words = " ".join(lst_text)

    return stemmed_words