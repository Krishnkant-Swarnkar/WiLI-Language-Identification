# WiLI-Language-Identification
This repository contains implementation of character Ngram Naive Bayes model for Language Identification.


## Directory Structure
### 4 sub directories:
- Data: it contains WiLI-2018 Benchmark Dataset
- Params: it contains the parameters of the saved models (initially bigram and trigram)
- Results: it contains the results of bigram snd trigram (on both the complete and restricted language set). It also contains images which visualize the results of trigram model.
- Utils: It contains the data_utils.py which is a helper module to read the data

### 2 files:
- naive_bayes.py :  This python script contains the implementation of Ngram Naive Bayes model.
- Readme.md


## How to Run
run (To train and test on all languages):
```
$ python3 naive_bayes.py --n 3 --Lambda 1.0 --model_name trigram --enable_train
```
run (To test pretrained model on all languages):
```
$ python3 naive_bayes.py --n 3 --Lambda 1.0 --model_name trigram
```
run (To train and test on 6 selected languages- French, German, Italian, Spanish, English, Dutch):
```
$ python3 naive_bayes.py --n 3 --Lambda 1.0 --model_name trigram --enable_train --enable_language_restriction
```
run (To test pretrained model on on 6 selected languages- French, German, Italian, Spanish, English, Dutch):
```
$ python3 naive_bayes.py --n 3 --Lambda 1.0 --model_name trigram --enable_language_restriction
```


## Dataset
I have used [Wili-2018 Benchmark Dataset](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjPqqe56KLZAhWJQ48KHc8lAvQQFgg1MAE&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1801.07779&usg=AOvVaw3VCyk-0c5cdTLIHINSf62M) which contains 1000 paragraphs for each of its 235 languages. 


## Performance (Test Set)

- BiGram Naive Bayes Model achieves 0.922357446809 accuracy (on all 235 languages).
- TriGram Naive Bayes Model achieves 0.939795744680851  accuracy (on all 235 languages).

- BiGram Naive Bayes Model achieves 0.9833333333333333 accuracy (on 6 languages: French, German, Italian, Spanish, English, Dutch).
- TriGram Naive Bayes Model achieves 0.9856666666666667 accuracy (on 6 languages: French, German, Italian, Spanish, English, Dutch).
The speed of the classifier increases with the decrease in the number of the languages to be classified.
Both the models use additive smoothing with lambda = 1 (Add-One Smoothing), but optionally, lambda can be changed.


## Intuition 
Since when given the document, the language is unknown, so word based models can not be used because segmentation can not be done without knowing the language of the document (though with many languages word segmentetion with whitespaces works well, but not for with the 235 languages in the dataset). So character n grams seem to be the best choice.


## Some Related work on language detection using character ngrams
- "Google's Compact Language Detector 2" also works on character ngrams (and is optimized for fast speed).
- Priyank Mathur, Arkajyoti Misra, Emrah Budur, “Language Identification from Text Documents”, 2015.
- Some approaches also use word ngrams, but again that works when a single method of word segmentation can be applied for all the languages we are dealing with.


## Visualizations of Trigram Results

### On six languages:
![](Results/Trigram_six_conf_matrix_YG.png?raw=true)

### On all 235 languages:
![](Results/Trigram_all_lang_conf_mat_YG.png?raw=true)














