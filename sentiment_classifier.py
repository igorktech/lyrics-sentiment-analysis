# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# the real thing:
data_path = "data/sentiment_train.csv"

# modifiers that change the sentiment of a word
NEGATING_WORDS=["no", "not", "neither", "nor"]

# download nltk stopwords module if not present
try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords')
 
# removing the negation words from the stopwords because they are needed for sentiment analysis
stops = set(stopwords.words('english'))
stops.remove("no")
stops.remove("not")
stops.remove("nor")

stemmer = PorterStemmer()

#NRC Valence, Arousal, and Dominance Lexicon
#https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip
valences=pd.read_csv('valence-NRC-VAD-Lexicon.txt', sep="\t", names=["word", "sentiment_score"])

#VADER
#https://medium.com/@piocalderon/vader-sentiment-analysis-explained-f1c4f9101cd9
#https://www.kaggle.com/datasets/nltkdata/vader-lexicon?resource=download
vader_lexicon=pd.read_csv('vader_lexicon.txt', sep="\t", names=["word", "sentiment_score_wrong_scale", "_", "__"])
vader_lexicon["sentiment_score"]=[(score+4)/8 for score in vader_lexicon["sentiment_score_wrong_scale"]]


# get the data from the file paths
data_train = pd.read_csv(data_path)


def preprocess_text(data):
    """
    prepare the input text and store the output in "tokens" and "cleaned_tokens" columns
    
    Parameters
    ----------
    data : pd.DataFrame
        data to modify in place.
    """
    data["tokens"]=[seq.replace("-", " ").replace(",", " ").split() for seq in data.text]
    data["cleaned_tokens"]=[clean_tokens(tokens) for tokens in data.tokens]


def clean_tokens(tokens):
    """
    clean token to remove stopwords, punctuation and build stems of input

    Parameters
    ----------
    tokens : list of strings
        input tokens.

    Returns
    -------
    stems : list of strings
        cleaned tokens.

    """
    tokens_without_stopwords=[]
    for token in tokens:
        if not token in stops and not token in [".", ",", ";", "!", "?"]:
            tokens_without_stopwords.append(token)
    
    # stems = []
    # for word in tokens_without_stopwords:
    #     stem_word = stemmer.stem(word)
    #     stems.append(stem_word)
    # return stems

    return tokens_without_stopwords


def feature_extraction(data):
    """
    calculates the features of data in place

    Parameters
    ----------
    data : pd.Dataframe
        data to calculate the features for.
    """
    data["n_characters"] = [len(data.text[i]) for i in range(len(data.text))]
    
    data["sentiment_scores_valence"]=get_average_sentiment_scores_for_different_lexicons(data["tokens"], valences)
    data["sentiment_scores_vader"]=get_average_sentiment_scores_for_different_lexicons(data["tokens"], vader_lexicon)



def get_average_sentiment_scores_for_different_lexicons(tokens, sent_lexicon):
    """
    calculates the average sentiment score of the tokens according to the provided sentiment lexicon

    Parameters
    ----------
    tokens : list of list of strings
        list of the tokens to calculate the average sentiment for.
    sent_lexicon : pd.Dataframe
        contains a single token (column "word") and a sentiment score (column "sentiment_score") per row.

    Returns
    -------
    average_valences : list of float
        the average sentiment score of the tokens.

    """
    average_sentiment_scores = []
    # one sequence in tokens is equivallent to cell in the dataframe
    for seq in tokens:
        n_words_with_sentiment_score = 0
        sum_sentiment_scores=0
        
        # first token doesn't have predessors
        # important for negating words
        predecessor = ""
        prepredecessor = ""
        
        for index, token in enumerate(seq):
            # get the fitting row in the sent_lexicon
            row = sent_lexicon.loc[sent_lexicon['word'] == token]
            if len(row) > 0:
                n_words_with_sentiment_score += 1
                word_sentiment_score = row.iloc[0].sentiment_score
                # negating words change the valence of the following words
                if predecessor in NEGATING_WORDS or prepredecessor in NEGATING_WORDS:
                    # 1-x calculates the oposite on a scale from 0 to 1
                    word_sentiment_score = 1-word_sentiment_score
                sum_sentiment_scores += word_sentiment_score
                
            prepredecessor = predecessor
            predecessor = token
        
        if n_words_with_sentiment_score > 0:
            average_sentiment_score = sum_sentiment_scores / n_words_with_sentiment_score
        else:
            # if no word was found in the lexicon, the sentence has a neutral sentiment in total
            average_sentiment_score = 0.5
        
        average_sentiment_scores.append(average_sentiment_score)
    return average_sentiment_scores


# preprocess lyrics
preprocess_text(data_train)

# extract the features of lyrics
feature_extraction(data_train)
