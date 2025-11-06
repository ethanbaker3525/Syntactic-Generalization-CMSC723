### THIS FILE CONTAINS HELPER FUNCTIONS FOR EXTRACTING SURPRISAL ###

import numpy as np
import torch
import torch.nn.functional as F
import sys

from minicons import scorer

gpt2_scorer = None
gptj_scorer = None
llama2_scorer = None
llama3_scorer = None
qwen3_scorer = None

def get_tok_surprisal_tuples(sentence, model_name): # abstraction layer for easily inferencing multiple models, if infrencing a model requires special steps, this is the method to modify.

    global gpt2_scorer
    global gptj_scorer
    global llama2_scorer
    global llama3_scorer
    global qwen3_scorer

    if model_name == "gpt2":
        if gpt2_scorer == None:
            gpt2_scorer = scorer.IncrementalLMScorer("gpt2", "cpu")
        return gpt2_scorer.token_score(sentence, surprisal=True, base_two=True)[0]
    elif model_name == "gptj":
        raise NotImplementedError
        if gptj_scorer == None:
            gptj_scorer = scorer.IncrementalLMScorer("EleutherAI/gpt-j-6b")
        return gptj_scorer.token_score(sentence, surprisal=True, base_two=True)[0]
    elif model_name == "llama2":
        raise NotImplementedError
        if llama2_scorer == None:
            llama2_scorer = scorer.IncrementalLMScorer("meta-llama/Llama-2-7b")
        return llama2_scorer.token_score(sentence, surprisal=True, base_two=True)[0]
    elif model_name == "llama3":
        raise NotImplementedError
        if llama3_scorer == None:
            llama3_scorer = scorer.IncrementalLMScorer("meta-llama/Llama-3.1-8B")
        return llama3_scorer.token_score(sentence, surprisal=True, base_two=True)[0]
    elif model_name == "qwen3":
        raise NotImplementedError
        if qwen3_scorer == None:
            qwen3_scorer = scorer.IncrementalLMScorer("Qwen/Qwen3-0.6B", "cpu")
        return qwen3_scorer.token_score(sentence, surprisal=True, base_two=True)[0] 


def clean_token_surprisal_tuples(token_surprisal_tuples): # removes meta characters (â,Ģ,Ļ,Ġ) from token_surprisal_tuples
    clean_token_surprisal_tuples = []
    for token, surprisal in token_surprisal_tuples:
        token = token.replace("Ġ", "")
        token = token.replace("âĢ", "")
        #token = token.replace("Ļ", "’") # this token corresponds to ' in words like couldn't (NOTE: this shouldnt need to happen, as all ’, should be replaced with ' before hand)
        clean_token_surprisal_tuples.append((token, surprisal))
    return clean_token_surprisal_tuples
    
# modified from https://github.com/umd-psycholing/lm-syntactic-generalization
def align_surprisal(token_surprisal_tuples, sentence): # given a token_surprisal_tuples list and a sentence, aligns the token surprisals with the words to form the surprisal at each word
    token_surprisal_tuples = clean_token_surprisal_tuples(token_surprisal_tuples) # remove meta characters from tokens
    words = sentence.split(" ")
    token_index = 0
    word_index = 0
    word_level_surprisal = []  # list of word, surprisal tuples
    while token_index < len(token_surprisal_tuples):
        current_word = words[word_index]
        current_token, current_surprisal = token_surprisal_tuples[token_index]
        # token does not match, alignment must be adjusted
        mismatch = (current_word != current_token)
        while mismatch:
            token_index += 1
            current_token += token_surprisal_tuples[token_index][0]
            current_surprisal += token_surprisal_tuples[token_index][1]
            mismatch = current_token != current_word
        word_level_surprisal.append((current_word, current_surprisal))
        token_index += 1
        word_index += 1
    return word_level_surprisal
    
def get_word_surprisal_tuples(sentence:list, model:str) -> list: # given a list of sentences and a language model, returns a list of word surprisal tuples

    token_surprisal_tuples = get_tok_surprisal_tuples(sentence, model) # token_score() outputs a length 1 list with a list of tuples, each containing the token and the surprisal at that token
    # -> [(tok0, 0.0), (tok1, 1.0), ...]
    # Worth noting that some words are split into tokens oddly, such as [('L', 0.0), ('ily', 13.184611320495605), ('Ġwondered', 12....
    word_surprisal_tuples = align_surprisal(token_surprisal_tuples, sentence) # groups multiple tokens comprising a single word by summing their surprisals
    # -> [(word0, 0.0), (word1, 1.0), ...]
    return word_surprisal_tuples

def get_surprisal_at_word(word_surprisal_tuples:list, word:str) -> float: 
    for word_surprisal_tuple in word_surprisal_tuples:
        if word_surprisal_tuple[0] == word:
            return word_surprisal_tuple[1]
        
def get_sentence_surprisal(word_surprisal_tuples:list) -> float: 
    return  sum(word_surprisal_tuple[1] for word_surprisal_tuple in word_surprisal_tuples)
    
if __name__ == "__main__":
    word_surprisal_tuples = get_word_surprisal_tuples("This is a test sentence, hello!", "gpt2")
    print(word_surprisal_tuples)
    print(get_surprisal_at_word(word_surprisal_tuples, "test"))
    print(get_sentence_surprisal(word_surprisal_tuples))

