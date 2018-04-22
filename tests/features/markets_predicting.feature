Feature: Market predicting

  Background: all tests need a currency analyser
    Given we created a currency analyser
      And all the features are: mexico, apples, good, banana, pizza
      And the main features are: mexico, apples, good

  Scenario: Predicting when features match
     When we analyse a tweet: Wall on the border with Mexico.
     Then the result is Down
      And it is based on mexico features

  Scenario: Predicting when features do not match to the main model
     When we analyse a tweet: I like pizza and bananas.
     Then the result is Up
      And it is based on pizza, banana features