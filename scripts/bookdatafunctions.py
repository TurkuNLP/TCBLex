#Imports
import pandas as pd
import os
import TCBC_tools.Structure as Structure
import TCBC_tools.FeatureExtraction as fe

def getStatisticsForDatabaseOnlyPos(sub_corpora, word_age_appearances=None, lemma_age_appearances=None):
    #If wfsa or lfsa included
    flag = (word_age_appearances or lemma_age_appearances) == None
    returnable = {}
    if flag:
        word_age_appearances = {}
        lemma_age_appearances = {}

    #First sort out the subcorpora
    for s in sub_corpora:
        if flag:
            sub_corp = s
        else:
            sub_corp = sub_corpora[s]
        combined_data = pd.concat(sub_corp.values()).reset_index()
        filtered_data = combined_data[['text','lemma','upos']]
        filtered_data = filtered_data.drop_duplicates(['text','lemma','upos'], ignore_index=True)
        #Add word-pos frequencies
        v_words_pos = fe.getColumnFrequencies(sub_corp, ['text','upos'])
        word_pos_freqs = fe.combineFrequencies(v_words_pos)
        filtered_data['Word-POS F'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos']].to_numpy(dtype='str')]
        #Add word frequencies
        v_words = fe.getColumnFrequencies(sub_corp, ['text'])
        word_freqs = fe.combineFrequencies(v_words)
        filtered_data['Word F'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word CD
        word_CD = fe.getCD(v_words)
        filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word D
        word_D = fe.getDispersion(v_words, word_freqs)
        filtered_data['Word D'] = [word_D[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word U
        word_U = fe.getU(v_words, word_freqs, word_D)
        filtered_data['Word U'] = [word_U[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word SFI
        word_SFI = fe.getSFI(word_U)
        filtered_data['Word SFI'] = [word_SFI[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word zipf-values
        l = fe.getL(fe.getTokenAmounts(sub_corp))
        word_zipfs = fe.getZipfValues(l, word_freqs)
        filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]


        #Add lemma frequencies
        v_lemmas = fe.getColumnFrequencies(sub_corp, ['lemma'])
        lemma_freqs = fe.combineFrequencies(v_lemmas)
        filtered_data['Lemma F'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma CD
        lemma_CD = fe.getCD(v_lemmas)
        filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma D
        lemma_D = fe.getDispersion(v_lemmas, lemma_freqs)
        filtered_data['Lemma D'] = [lemma_D[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma U
        lemma_U = fe.getU(v_lemmas, lemma_freqs, lemma_D)
        filtered_data['Lemma U'] = [lemma_U[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma SFI
        lemma_SFI = fe.getSFI(lemma_U)
        filtered_data['Lemma SFI'] = [lemma_SFI[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma zipf-values
        lemma_zipfs = fe.getZipfValues(l, lemma_freqs)
        filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add taivutusperhe size
        tv_sizes = fe.getTaivutusperheSize(sub_corp)
        filtered_data['Lemma MPS'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add syllables per word
        syllable_amount = fe.getSyllableAmountsForWords(filtered_data['text'])
        filtered_data['Word Syllables'] = [syllable_amount[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add length of word
        filtered_data['Word Length'] = filtered_data['text'].str.len()
        #Add length of lemma
        filtered_data['Lemma Length']= filtered_data['lemma'].str.len()
        key = s
        #Slow but steady way of adding words and first appearance ages...
        if flag:
            key = Structure.findAgeFromID(list(sub_corp.keys())[0])
            for w in word_freqs.index:
                word_age_appearances.setdefault(w[0],key)
            for le in lemma_freqs.index:
                lemma_age_appearances.setdefault(le[0],key)
        #Add first appearance
        filtered_data['Word FSA'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        filtered_data['Lemma FSA'] = [lemma_age_appearances[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add to dictionary
        returnable[key] = filtered_data.sort_values('text')

    if flag:
        return returnable, word_age_appearances, lemma_age_appearances
    else:
        return returnable

def getStatisticsForDatabasePosFeats(sub_corpora, word_age_appearances=None, lemma_age_appearances=None):
    #If wfsa or lfsa included
    flag = (word_age_appearances or lemma_age_appearances) == None
    returnable = {}
    if flag:
        word_age_appearances = {}
        lemma_age_appearances = {}

    #First sort out the subcorpora
    for s in sub_corpora:
        if flag:
            sub_corp = s
        else:
            sub_corp = sub_corpora[s]
        combined_data = pd.concat(sub_corp.values()).reset_index()

        filtered_data = combined_data[['text','lemma']]
        filtered_data["upos+features"] = combined_data[['upos','feats']].agg('+'.join, axis=1)
        filtered_data = filtered_data.drop_duplicates(['text','lemma','upos+features'], ignore_index=True)
        #Add word-pos frequencies
        freqs = {}
        for key in sub_corp:
            book = sub_corp[key]
            book["upos+features"] = book[['upos','feats']].agg('+'.join, axis=1)
            freqs[key] = book[["text","upos+features"]].value_counts()
        v_words_pos = freqs
        word_pos_freqs = fe.combineFrequencies(v_words_pos)
        filtered_data['Word-POS+FEATS F'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos+features']].to_numpy(dtype='str')]
        #Add word frequencies
        v_words = fe.getColumnFrequencies(sub_corp, ['text'])
        word_freqs = fe.combineFrequencies(v_words)
        filtered_data['Word F'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word CD
        word_CD = fe.getCD(v_words)
        filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word D
        word_D = fe.getDispersion(v_words, word_freqs)
        filtered_data['Word D'] = [word_D[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word U
        word_U = fe.getU(v_words, word_freqs, word_D)
        filtered_data['Word U'] = [word_U[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word SFI
        word_SFI = fe.getSFI(word_U)
        filtered_data['Word SFI'] = [word_SFI[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word zipf-values
        l = fe.getL(fe.getTokenAmounts(sub_corp))
        word_zipfs = fe.getZipfValues(l, word_freqs)
        filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]


        #Add lemma frequencies
        v_lemmas = fe.getColumnFrequencies(sub_corp, ['lemma'])
        lemma_freqs = fe.combineFrequencies(v_lemmas)
        filtered_data['Lemma F'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma CD
        lemma_CD = fe.getCD(v_lemmas)
        filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma D
        lemma_D = fe.getDispersion(v_lemmas, lemma_freqs)
        filtered_data['Lemma D'] = [lemma_D[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma U
        lemma_U = fe.getU(v_lemmas, lemma_freqs, lemma_D)
        filtered_data['Lemma U'] = [lemma_U[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma SFI
        lemma_SFI = fe.getSFI(lemma_U)
        filtered_data['Lemma SFI'] = [lemma_SFI[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma zipf-values
        lemma_zipfs = fe.getZipfValues(l, lemma_freqs)
        filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add taivutusperhe size
        tv_sizes = fe.getTaivutusperheSize(sub_corp)
        filtered_data['Lemma MPS'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add syllables per word
        syllable_amount = fe.getSyllableAmountsForWords(filtered_data['text'])
        filtered_data['Word Syllables'] = [syllable_amount[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add length of word
        filtered_data['Word Length'] = filtered_data['text'].str.len()
        #Add length of lemma
        filtered_data['Lemma Length']= filtered_data['lemma'].str.len()
        key = s
        #Slow but steady way of adding words and first appearance ages...
        if flag:
            key = Structure.findAgeFromID(list(sub_corp.keys())[0])
            for w in word_freqs.index:
                word_age_appearances.setdefault(w[0],key)
            for le in lemma_freqs.index:
                lemma_age_appearances.setdefault(le[0],key)
        #Add first appearance
        filtered_data['Word FSA'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        filtered_data['Lemma FSA'] = [lemma_age_appearances[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add to dictionary
        returnable[key] = filtered_data.sort_values('text')

    if flag:
        return returnable, word_age_appearances, lemma_age_appearances
    else:
        return returnable


def formatDataForPaperOutputBasic(corpus: dict[str,pd.DataFrame], output_dir: str):
    """
    Function which takes in a corpus and provides four sets of dictionaries as sets of csv-files:
    1. contains data for exact ages as subcorpora
    2. contains data for age groups as subcorpora
    3. contains data for registers as subcorpora
    4. contains data for the whole corpus

    output_dir is the base folder in which all data files will be put into
    """
    ages = sorted(Structure.getAvailableAges(corpus))

    #If output_dir does not yet exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ready_dfs_ages = {}
    ready_dfs_groups = {}
    ready_dfs_whole = {}

    #Subcorpora based on the target age groups
    sub_corpora = []
    #Combine books aged 15 and up into one sub-corpus as there are very few entries in 16,17,18
    over_15 = []
    for i in ages:
        if i<15:
            sub_corpora.append(Structure.getDistinctSubCorp(corpus, i))
        else:
            over_15.append(Structure.getDistinctSubCorp(corpus, i))
    #Sort the aged 15 and over sub-corpora from lowest age to highest
    over_15.sort(key=lambda x:int(Structure.findAgeFromID(list(x.keys())[0])))
    #Combine 15+ aged books into one sub-corpus
    sub_corpora.append(Structure.combineSubCorpDicts(over_15))
    #Sort the sub-corpora from lowest age to highest
    sub_corpora.sort(key=lambda x:int(Structure.findAgeFromID(list(x.keys())[0])))
    #Keep track of when words and lemmas first appear in terms of intended reading age
    ready_dfs_ages, word_age_appearances, lemma_age_appearances = getStatisticsForDatabaseOnlyPos(sub_corpora)

    writePaperOutputCsv(ready_dfs_ages, 'ages_csv', output_dir)
    print("Ages outputted!")
    #Define age group sub-corpora

    
    #Generate correct keys/ids
    group_1 = [5,6,7,8]
    group_2 = [9,10,11,12]
    group_3 = ages[ages.index(13):]
    #Distinct subcorpora
    sub_corp_1= Structure.combineSubCorpDicts([Structure.getDistinctSubCorp(corpus, x) for x in group_1])
    sub_corp_2= Structure.combineSubCorpDicts([Structure.getDistinctSubCorp(corpus, x) for x in group_2])
    sub_corp_3= Structure.combineSubCorpDicts([Structure.getDistinctSubCorp(corpus, x) for x in group_3])
    sub_corps = dict(zip(['7-8','9-12','13+'],[sub_corp_1, sub_corp_2, sub_corp_3]))

    ready_dfs_groups = getStatisticsForDatabaseOnlyPos(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_groups, 'groups_csv', output_dir)
    print("Groups done!")

    print("Start registers")
    #Work with registers
    reg1 = {key:corpus[key] for key in corpus if key[-1]=='1'}
    reg2 = {key:corpus[key] for key in corpus if key[-1]=='2'}
    reg3 = {key:corpus[key] for key in corpus if key[-1]=='3'}

    ready_dfs_registers = {}

    sub_corps = dict(zip(['Fiction','Nonfiction','Textbook'],[reg1, reg2, reg3]))

    ready_dfs_registers = getStatisticsForDatabaseOnlyPos(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_registers, 'genres_csv', output_dir)
    print("Registers done!")

    temp_whole = {"Whole":corpus}
    ready_dfs_whole = getStatisticsForDatabaseOnlyPos(temp_whole, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_whole, 'whole_csv', output_dir)
    
    print("All done!!")
    #return ready_dfs_ages, ready_dfs_groups, ready_dfs_registers, ready_dfs_whole

def formatDataForPaperOutputWithFeats(corpus: dict[str,pd.DataFrame], output_dir: str):
    """
    Function which takes in a corpus and provides four sets of dictionaries as sets of csv-files:
    1. contains data for exact ages as subcorpora
    2. contains data for age groups as subcorpora
    3. contains data for registers as subcorpora
    4. contains data for the whole corpus
    """
    ages = sorted(Structure.getAvailableAges(corpus))

    ready_dfs_ages = {}
    ready_dfs_groups = {}
    ready_dfs_whole = {}

    #Subcorpora based on the target age groups
    sub_corpora = []
    #Combine books aged 15 and up into one sub-corpus as there are very few entries in 16,17,18
    over_15 = []
    for i in ages:
        if i<15:
            sub_corpora.append(Structure.getDistinctSubCorp(corpus, i))
        else:
            over_15.append(Structure.getDistinctSubCorp(corpus, i))
    #Sort the aged 15 and over sub-corpora from lowest age to highest
    over_15.sort(key=lambda x:int(Structure.findAgeFromID(list(x.keys())[0])))
    #Combine 15+ aged books into one sub-corpus
    sub_corpora.append(Structure.combineSubCorpDicts(over_15))
    #Sort the sub-corpora from lowest age to highest
    sub_corpora.sort(key=lambda x:int(Structure.findAgeFromID(list(x.keys())[0])))
    #Keep track of when words and lemmas first appear in terms of intended reading age
    ready_dfs_ages, word_age_appearances, lemma_age_appearances = getStatisticsForDatabasePosFeats(sub_corpora)

    writePaperOutputCsv(ready_dfs_ages, 'ages_with_features_csv', output_dir)
    print("Ages outputted!")
    #Define age group sub-corpora

    
    #Generate correct keys/ids
    group_1 = [5,6,7,8]
    group_2 = [9,10,11,12]
    group_3 = ages[ages.index(13):]
    #Distinct subcorpora
    sub_corp_1= Structure.combineSubCorpDicts([Structure.getDistinctSubCorp(corpus, x) for x in group_1])
    sub_corp_2= Structure.combineSubCorpDicts([Structure.getDistinctSubCorp(corpus, x) for x in group_2])
    sub_corp_3= Structure.combineSubCorpDicts([Structure.getDistinctSubCorp(corpus, x) for x in group_3])
    sub_corps = dict(zip(['7-8','9-12','13+'],[sub_corp_1, sub_corp_2, sub_corp_3]))

    ready_dfs_groups = getStatisticsForDatabasePosFeats(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_groups, 'groups_with_features_csv', output_dir)
    print("Groups done!")

    print("Start registers")
    #Work with registers
    reg1 = {key:corpus[key] for key in corpus if key[-1]=='1'}
    reg2 = {key:corpus[key] for key in corpus if key[-1]=='2'}
    reg3 = {key:corpus[key] for key in corpus if key[-1]=='3'}

    ready_dfs_registers = {}

    sub_corps = dict(zip(['Fiction','Nonfiction','Textbook'],[reg1, reg2, reg3]))

    ready_dfs_registers = getStatisticsForDatabasePosFeats(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_registers, 'genres_with_features_csv', output_dir)
    print("Registers done!")

    temp_whole = {"Whole":corpus}
    ready_dfs_whole = getStatisticsForDatabasePosFeats(temp_whole, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_whole, 'whole_with_features_csv', output_dir)
    
    print("All done!!")
    #return ready_dfs_ages, ready_dfs_groups, ready_dfs_registers, ready_dfs_whole

def writePaperOutputCsv(ready_dfs: dict[str:pd.DataFrame], name: str, parent_folder:str):
    """
    Simple function for writing csv-files based on a list of dictionaries
    Name is the name of the folder containing the csv-files
    """
    path = parent_folder+"/"+name
    if not os.path.exists(path):
        os.mkdir(path)
    for df in ready_dfs:
        name = path+"/"+str(df)+".csv"
        ready_dfs[df].to_csv(name, index=False, sep=';')
        print(df+" done!")
