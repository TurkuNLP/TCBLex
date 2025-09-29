#Imports
from scripts import bookdatafunctions as bdf
import warnings
warnings.filterwarnings('ignore')

#Constants
CONLLU_PATH = "Conllus_v-1-0" #Provide the folder where your CoNLLU files are located
OUTPUT_DIR = 'Data_v1-0' #Provide the folder where the created lexical database should be stored
#Initialize corpus from CoNLLU files
corpus = bdf.cleanWords(bdf.initBooksFromConllus(CONLLU_PATH))    

#Use the monster function (see scripts/bookdatafunctions.py) to get correctly formatted DataFrames and output to Data folder
bdf.formatDataForPaperOutputBasic(corpus, OUTPUT_DIR)
print("Basic set done!")
bdf.formatDataForPaperOutputWithFeats(corpus, OUTPUT_DIR)

print("All done!")