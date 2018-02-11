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