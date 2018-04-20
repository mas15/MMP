Feature: Phrases extraction

  Background: all tests need a phrases extractor
    Given we created a phrases extractor

  Scenario: Extract words
    Given we have vocabulary of: cucumber, apple, banana, tomato
     When we extract phrases from the sentence: I like to eat apples and bananas
     Then we get apple, banana extracted

  Scenario: Extract phrases
    Given we have vocabulary of: cucumber, apple pie, banana, tomato sauce
     When we extract phrases from the sentence: I like to eat apple pie and bananas.
     Then we get apple pie, banana extracted

  Scenario: Extract phrases when nothing matches
    Given we have vocabulary of: cucumber, apple pie, banana, tomato sauce
     When we extract phrases from the sentence: I like to eat pizza and chips.
     Then nothing is extracted

  Scenario: Build a vocabulary
    Given we have texts: Example sentence, Second sentences, Text with tax cuts, Tax cuts and Hillary Clinton
     When we build a vocabulary
     Then the vocabulary consists of hillary, text, sentence, clinton, tax cuts, clinton

  Scenario: Extract phrases when build a vocabulary
    Given we have texts: Example sentence, Second sentences, Text with tax cuts, Tax cuts and Hillary Clinton
     When we build a vocabulary
     When we extract phrases from the sentence: Sentence with tax cuts!
     Then we get sentence, tax cuts extracted
