import nltk


## membuka file dan output berupa corpus yang sudah di casefold
def openCorpus(corpus_file_name):
    with open(corpus_file_name, "r") as corpus_file:
        corpus = corpus_file.read()
        return corpus.casefold()


## List of Tuple -> Dict of Int, Int
# Input adalah term beserta dengan tag-nya, output adalah daftar tag
# dengan jumlah kemunculannya serta total dari tag yang ada pada corpus
def tagsetList(term_tuple):
    list_tag = {}
    total_tag = 0
    for (word, tag) in term_tuple:
        if tag in list_tag: list_tag[tag] += 1
        else: list_tag[tag] = 1
        total_tag += 1
    return list_tag, total_tag


## List of Tuple -> Dict Of Int
# Input adalah term beserta dengan tag-nya, output adalah daftar tag
# dalam format bigram beserta dengan jumlah kemunculannya 
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



class BigramTagger:

    def __init__(self, corpus_file_name):
        ##initialization global variables
        self.corpus = openCorpus(corpus_file_name)
        self.term_tuple = [nltk.tag.str2tuple(t) for t in self.corpus.split()]
        self.tagset_list, self.total_tag = tagsetList(self.term_tuple)
        self.bigram_tagset_list = bigramTagSetList(self.term_tuple)
        self.tagger_prob_list = dict()


    def pWordFromTag(self, word, tag):
        return self.countWordWithTag(word, tag) / self.tagset_list[tag]
        
    def pTagFromPrevTag(self, prev_tag, tag):
        if ((prev_tag + " " + tag) in self.bigram_tagset_list): 
            return self.bigram_tagset_list[prev_tag + " " + tag] / self.tagset_list[prev_tag]
        else: return 0


    def countWordWithTag(self, word, tag):
        count = 0
        for row in self.term_tuple:
            if (row[0] == word and row[1] == tag):
                count += 1
        return count

    # mengembalikan tag pada word input secara unigram
    def unigramTag(self, word):
        prob_tag_list = {}
        term_dict = dict((x, y) for x, y in self.term_tuple)
        
        if word not in term_dict.keys(): return 'X'
        for (tag, tag_count) in self.tagset_list.items(): 
            prob_tag_list[tag] = tag_count / self.total_tag * self.pWordFromTag(word, tag)
            
        sorted_prob_list = sorted(prob_tag_list.items(), key=lambda kv: kv[1], reverse=True)
        self.tagger_prob_list[word] = sorted_prob_list

        # self.tagger_prob_list.append(f"Peluang {word} (unigram tagger): \n {sorted_prob_list}")
        return sorted_prob_list[0][0]


    ## String, String -> String
    # input adalah term yang akan ditag & term sebelumnya, 
    # output adalah tag pada term yang akan ditag secara bigram
    def bigramTagger(self, prev_word, word):
        prob_tag_list = {}
        term_dict = dict((x, y) for x, y in self.term_tuple)
        
        ## for all tag, the highest prob tag is selected
        if prev_word not in term_dict.keys(): return self.unigramTag(word)
        for (tag, value) in self.tagset_list.items():
            prev_tag = term_dict[prev_word]
            ## prevent zero values if word are unavailable in corpus
            prob_tag_list[tag] = (0.000001 + self.pWordFromTag(word, tag)) * self.pTagFromPrevTag(prev_tag, tag)
            
        
        sorted_prob_list = sorted(prob_tag_list.items(), key=lambda kv: kv[1], reverse=True)
        self.tagger_prob_list[word] = sorted_prob_list
        # self.tagger_prob_list.append(f"Peluang {word} (bigram tagger): \n {sorted_prob_list}")
        return sorted_prob_list[0][0]

    ## String -> String
    # input adalah sebuah kalimat. Output adalah kalimat yang sudah di beri tag bigram
    def bigramSentencesTagger(self, sentences):
        idx = 0
        word = sentences.casefold().split()
        tagged_sentences = ""
        self.tagger_prob_list = dict()
        while True:
            if (idx+1 > len(word) ): break
            elif(idx == 0):
                tagged_sentences += word[idx] + "/" + self.unigramTag(word[idx])
            else:
                tagged_sentences += word[idx] + "/" + self.bigramTagger(word[idx-1], word[idx])
            tagged_sentences += " "
            idx += 1
        return tagged_sentences
