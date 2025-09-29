#Imports
import json
import os
import pandas as pd


temp_df = pd.DataFrame(pd.read_csv("ISBN_MAPS/ISBN2AGE.csv", delimiter=';', header=None))
isbn2age_series = pd.Series(temp_df[temp_df.columns[1]].to_numpy('str').tolist(), index=temp_df[temp_df.columns[0]].to_numpy('str').tolist())
#JSON to CoNLLU

def transformConlluLine(token: dict):
    id = str(token.get('id', '_'))
    text = str(token.get('text', '_'))
    lemma = str(token.get('lemma', '_'))
    upos = str(token.get('upos', '_'))
    xpos = str(token.get('xpos', '_'))
    feats = str(token.get('feats', '_'))
    head = str(token.get('head', '_'))
    deprel = str(token.get('deprel', '_'))
    return '\t'.join([id, text, lemma, upos, xpos, feats, head, deprel])+"\t_\t_\n"

def main():
    """
    Main function which looks for .json files produced by Trankit and transforms them to CoNLLU files
    """
    #JSON to CoNLLU

    for file in os.listdir("Parsed"):
        new_filename = file[:14] +  str(isbn2age_series.loc[file[:13]]) + file[15:17]
        #Don't do work if file already exists!
        if os.path.exists("Conllus/"+new_filename+".conllu"):
            continue
        with open("Parsed/"+file) as json_file:
            #Convert json to dict
            data = json.load(json_file)
            #pprint(data)
            conllu_doc = ''
            for sentence in data['sentences']:
                conllu_doc += '# sent_id = '+str(sentence['id'])+"\n"
                conllu_doc += '# text = '+sentence['text']+'\n'
                for token in sentence['tokens']:
                    expanded = token.get('expanded')
                    if expanded:
                        conllu_doc += str(token['id'][0])+"-"+str(token['id'][1])+"\t"
                        conllu_doc += token['text']+"\t"
                        conllu_doc += "_\t_\t_\t_\t_\t_\t_\t_\n"
                        for e in token['expanded']:
                            conllu_doc += transformConlluLine(e)
                    else:
                        conllu_doc += transformConlluLine(token)
                conllu_doc += "\n"
            with open("Conllus/"+new_filename+".conllu", "w", encoding="utf-8") as writer:
                writer.write(conllu_doc)

if __name__ == "__main__":
    main()