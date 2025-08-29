# first training observations

- training was slower on wall clock time but converged much faster
  - took around 100k timesteps, lasted for barely over a minute and i got over the threshold

- continuous actions spaces seem to be far more sample efficient

- the agent can pick between extremes naturally (e.g. throttling the engine instead of fire-don't fire dichotomy) leading to far better results

- watching the agent play it was far smoother than its discrete counterpart, lowest reward in first 5 eps was 212.

- want to tweak with hyperparams and raise reward threshold and see what happens

