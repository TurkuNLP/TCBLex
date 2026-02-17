#Imports
from scripts import bookdatafunctions as bdf
from TCBC_tools import RandomTrash as rt, Structure as st
import warnings
warnings.filterwarnings('ignore')

#Constants
CONLLU_PATH = "Data_v1-0" #Provide the folder where your CoNLLU files are located
OUTPUT_DIR = 'Conllus_v-1-0' #Provide the folder where the created lexical database should be stored
#Initialize corpus from CoNLLU files
corpus = rt.cleanWords(st.initBooksFromConllus(CONLLU_PATH))    

#Use the monster function (see scripts/bookdatafunctions.py) to get correctly formatted DataFrames and output to Data folder
bdf.formatDataForPaperOutputBasic(corpus, OUTPUT_DIR)
print("Basic set done!")
bdf.formatDataForPaperOutputWithFeats(corpus, OUTPUT_DIR)

print("All done!")