
import nltk

## membuka file dan output berupa corpus yang sudah di casefold
def openCorpus(corpus_file_name):
    with open(corpus_file_name, "r") as corpus_file:
        corpus = corpus_file.read()
        return corpus.casefold()

## 
def tagsetList(term_tuple):
    list_tag = {}
    total_tag = 0
    for (word, tag) in term_tuple:
        if tag in list_tag: list_tag[tag] += 1
        else: list_tag[tag] = 1
        total_tag += 1
    return list_tag, total_tag

def bigramTagSetList(term_tuple):
    list_bigram_tag = {}
    idx = 0
    while True:
        if (idx+1 == len(term_tuple) ): break
        dict_key = term_tuple[idx][1] + " " + term_tuple[idx + 1][1]
        if dict_key in list_bigram_tag:
            list_bigram_tag[dict_key] += 1
        else:
            list_bigram_tag[dict_key] = 1
        idx += 1
    return list_bigram_tag

##initialization global variables
corpus = openCorpus('corpus_temp.dat')
term_tuple = [nltk.tag.str2tuple(t) for t in corpus.split()]
tagset_list, total_tag = tagsetList(term_tuple)
bigram_tagset_list = bigramTagSetList(term_tuple)

def pWordFromTag(word, tag):
    return countWordWithTag(word, tag) / tagset_list[tag]
    
def pTagFromPrevTag(prev_tag, tag):
    if ((prev_tag + " " + tag) in bigram_tagset_list): 
        return bigram_tagset_list[prev_tag + " " + tag] / tagset_list[prev_tag]
    else: return 0

def countWordWithTag(word, tag):
    count = 0
    for row in term_tuple:
        if (row[0] == word and row[1] == tag):
            count += 1
    return count

# mengembalikan tag pada word input secara unigram
def unigramTag(word):
    prob_tag_list = {}
    term_dict = dict((x, y) for x, y in term_tuple)
    
    if word not in term_dict.keys(): return 'X'
    for (tag, tag_count) in tagset_list.items(): 
        prob_tag_list[tag] = tag_count/total_tag * pWordFromTag(word, tag)
        
    sorted_prob_list = sorted(prob_tag_list.items(), key=lambda kv: kv[1], reverse=True)
    print('Peluang %s (unigram tagger): ' % (word))
    print(sorted_prob_list)
    print()
    return sorted_prob_list[0][0]


# String, String -> String
#  input adalah term yang akan ditag & term sebelumnya, 
#  output adalah tag pada term yang akan ditag secara bigram
def bigramTagger(prev_word, word):
    prob_tag_list = {}
    term_dict = dict((x, y) for x, y in term_tuple)
    
    ## for all tag, the highest prob tag is selected
    if prev_word not in term_dict.keys(): return unigramTag(word)
    for (tag, value) in tagset_list.items():
        prev_tag = term_dict[prev_word]
        prob_tag_list[tag] = pWordFromTag(word, tag) * pTagFromPrevTag(prev_tag, tag)
        
    
    sorted_prob_list = sorted(prob_tag_list.items(), key=lambda kv: kv[1], reverse=True)
    print('Peluang %s (bigram tagger): ' % (word))
    print(sorted_prob_list)
    return sorted_prob_list[0][0]

def bigramSentencesTagger(sentences):
    idx = 0
    word = sentences.casefold().split()
    tagged_sentences = ""
    while True:
        if (idx+1 > len(word) ): break
        elif(idx == 0):
            tagged_sentences += word[idx] + "/" + unigramTag(word[idx])
        else:
            tagged_sentences += word[idx] + "/" + bigramTagger(word[idx-1], word[idx])
        tagged_sentences += " "
        idx += 1
    return tagged_sentences
            

#main
tagged_sentences = bigramSentencesTagger("Aku dan hawking memiliki kursi roda yang menakjubkan dan dia berjalan di jalan yang benar . dia bermain facebook")
print(tagged_sentences)