# CONVERTING DOCUMENTS TO NUMBER VECTOR USING TF-IDF ALGORITHM
With the input is a document corpus, this program will convert them into csv_file and preprocess data to a dictionary including all the words that are in the docs. From this dictionary, the program will process data using TF-IDF algorithm and return a csv_file which saves number vectors of all the sentences in the docs. 

## Installation - Environment
To set up the environment and use the packages, libraries, you need to type the following command:

git clone https://github.com/hanahh080601/hanahh080601.github.io.git
Extract Train_Full.rar zip into a folder named Train_Full
Create a folder named CSV in the project
```
cd NLP_test
pip install -r requirements.txt
python main.py (convert from docs into csv_files)
python test.py (convert from csv_file into number vector)
```

