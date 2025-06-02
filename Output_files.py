#Imports
from scripts import bookdatafunctions as bdf
import warnings
warnings.filterwarnings('ignore')

#Constants
CONLLU_PATH = "Conllus" #Provide the folder where your CoNLLU files are located
ISBN2AGE_PATH = "ISBN_MAPS/ISBN2AGE.xlsx" #Provide the direct path of the file containing a mapping from ISBNs to intended reading ages


#Initialize corpus from CoNLLU files
corpus = bdf.mapGroup2Age(bdf.cleanWordBeginnings(bdf.initBooksFromConllus(CONLLU_PATH)), ISBN2AGE_PATH)      

#Use the monster function (see scripts/bookdatafunctions.py) to get correctly formatted DataFrames and output to Data folder
bdf.formatDataForPaperOutput(corpus)

print("All done!")