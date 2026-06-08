# Tweet Draft

I tested a small evolutionary AI setup against tabular Q-learning in the same food-seeking world.

Q-learning updated action values from reward feedback.

Evolution used 1,000 neural-network agents:

Evaluation -> Selection -> Mutation

No backprop.
No gradient descent.
No Q-table updates for the evolutionary agents.

Fitness:

- food eaten x 100
- cooperative events x 8
- survival steps x 1
- remaining energy x 0.25

In the direct social-evolution vs Q-learning comparison:

- Q-learning had the better late rolling average: 4,742.5
- Evolution had the best single observed agent: 9,026.7 vs Q-learning's 8,074.8

So this was not "evolution beats Q-learning."

Q-learning was more stable on average.
Evolution occasionally found stronger outliers.

Then I tested reproducibility:

- 100 independent seeds
- 500 generations per seed
- 1,000 agents per generation
- same social-evolution setup
- cultural memory disabled

Results:

- mean best score: 6,837.5
- median best score: 6,574.6
- max observed: 12,303.3
- 5/100 runs exceeded the original 9,026.7

My takeaway:

Evolutionary selection can surface strong behaviors in this simple environment, but the strongest agents are uncommon.

Q-learning remains the stronger average baseline here.

I would not call this broad emergent intelligence.

It is a controlled result showing that selection pressure can produce high-performing behavioral outliers, and the next question is whether memory, communication, cooperation, and cross-generation knowledge transfer can make those behaviors more stable and more common.
