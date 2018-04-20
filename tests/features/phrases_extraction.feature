Feature: Phrases extraction

  Scenario: Extract words
    Given we have vocabulary of: cucumber, apple, banana, tomato
     When we extract phrases from the sentence: I like to eat apples and bananas
     Then we get apple, banana extracted

  Scenario: Extract phrases
    Given we have vocabulary of: cucumber, apple pie, banana, tomato sauce
     When we extract phrases from the sentence: I like to eat apple pie and bananas
     Then we get apple pie, banana extracted