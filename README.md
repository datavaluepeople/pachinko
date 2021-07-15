# pachinko: one-shot optimal decision-making agents and algorithms 

## What is it?

**pachinko** is a package that implements _one-shot optimal decision-making agents_ to solve
environments provided by the [truman package][truman]. The goal is to implement agents that learn
in a single experience, so that they can perform well on unique systems that can't be reliably
simulated and that have a high cost of experimentation.

For more information about the types of environment provided by truman, see the [truman package
README][truman].

## Main features

pachinko implements various algorithms exposed as agents compatible with OpenAI [Gym][gym]
environments:
- the Random agent, which takes a random action each timestep
- the EpsilonGreedy agent, which takes the highest conversion rate action, with a small probability
  of choosing a random action
- the Periodic agent, which splits the time series into a repeating period (eg a separate
  conversion rate for each day of the week) and chooses an action each step based on the upper
  confidence bound of the conversion rate

pachinko is light on testing - the true test of the agents is their performance in environment
suites from the [truman package][truman].

## Installation

To get started, you'll need to have Python 3.7+ installed. Then:

```
pip install pachinko
```

### Editable install from source

You can also clone the pachinko Git repository directly. This is useful when you're working on
adding new agents or modifying pachinko itself. Clone and install in editable mode using:

```
git clone https://github.com/datavaluepeople/pachinko
cd pachinko
pip install -e .
```

## License

[MIT](LICENSE.txt)


[truman]: https://github.com/datavaluepeople/truman
[gym]: https://github.com/openai/gym
