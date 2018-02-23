MMP
Twitter only allows access to a users most recent 3240 tweets with this method

Czy w bazie danych trzymac?
I created a twitter account mas15@aber.ac.uk, kol
Created twitter app to obtain keys and access tokens 
https://apps.twitter.com/app/14789943

Consumer Key (API Key)	sHEmHwtt3koxdLoa6Ok2vEduH
Consumer Secret (API Secret)	fJZsN0OQW80Vqnw265rT8Jvc7VwADGNS0kB5vMjIRG4d3eywzJ
Access Token	962374758783471616-KZtDMvJkmJigxZWUr3EI8x5iOgguRQB
Access Token Secret	hLtD8RMXyT2kcUDR9oLg7P7MtXSkrlgBWgsZk4u8GJY84

KOD Z http://tweepy.readthedocs.io/en/v3.5.0/getting_started.html


Sentiment anysis:
-Split corpus na test i trainig
we have got 2 data sets : positive, negative, each 150 tweets 
half train, half test

-Define vocabulary - set of all words - using training data only -get all the words from pos and neg tweets, remove duplicates
-Extract features - create word tuples - use a vocabulary
-Train classifier, with NLTK, using trainig data only
-Classify test data, with NLTK, using test data
-Measure accuracy on test data


Wpierw zalozylem prywatne repozytorium na gicie, posciagalem wszystkie potrzebne biblioteki, django, nltk itp

jak to dziala:
tokenizacja: dzielenie na subject, verb, object
potem bag of words: jak czesto slowo sie pojawia w CZYM?
potem patrzymy subjectivity kazdego slowa w lexicon - ale my nie chcemy tego
subkectivity. how much opinial ,. how factual


tweety z http://uk.businessinsider.com/trump-tweets-of-the-year-2017-12?r=US&IR=T/#when-he-chastised-so-called-russian-hacking-1
https://news.sky.com/story/sad-pathetic-a-history-of-donald-trumps-twitter-insults-11123543

919914000959397888,"b'I was recently asked if Crooked Hillary Clinton is going to run in 2020? My answer was, ""I hope so!""'",2017-10-16 13:12:43,pos

KFOLD z http://thelillysblog.com/2017/08/18/machine-learning-k-fold-validation/


-------------
FLASK
export FLASK_APP=hello.py
flask run


Using MaxEntClassifier gave similiar results
Average accuracy was abnout 75% (0.71, 0.82, 0.83)
"crooked Hilary" was giving
pos
0.0
(0.5125457426176581, 0.48745425738356085)
MAGA was usually good