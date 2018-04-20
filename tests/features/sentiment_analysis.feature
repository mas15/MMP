Feature: Sentiment analysis

  Background: all tests need a sentiment analyser
    Given we created a sentiment analyser

  Scenario: Training
    Given we have texts: Good movie, Good song, Bad tax cuts, Very bad tax cuts, tax cuts are bad
      And we know their sentiment: pos, pos, neg, neg, neg
     When we train a sentiment analyser
     Then we can predict that Good politician is positive
      And we can predict that Bad Mexicans is negative
      And we can predict that New tax cuts is negative
