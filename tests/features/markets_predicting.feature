Feature: Market predicting

  Background: all tests need a currency analyser
    Given we created a currency analyser
      And all the features are: apples, good, mexico, pizza, snack
      And the main features are: apples, good, mexico

  Scenario: Predicting when features match
     When we analyse a tweet: Wall on the border with Mexico.
     Then the result is Down
      And it is based on mexico features

  Scenario: Predicting when features do not match to the main model
     When we analyse a tweet: I like pizza and snacks.
     Then the result is Up
      And it is based on pizza, snack features