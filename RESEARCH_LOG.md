# BioSearch RL Research Log

I use this file to keep track of how the project changed. The README is the
short version. This file includes the failed ideas and corrections that led to
the final interpretation.

The simulator is synthetic. None of these results should be treated as claims
about moth biology, real odor plumes, or physical robots.

## Initial scope

I wanted one project that combined reinforcement learning, robotics, unreliable
sensors, and a result I could explain in an interview. I kept the first version
small on purpose:

- one agent;
- a 2D world;
- a cheap puff plume instead of CFD;
- six discrete actions;
- local observations only;
- random and moth-inspired baselines before RL;
- deterministic seeds and tests from the start.

I left inverse RL and multiple agents out. They would add more code without
answering the basic question of whether one agent can find the source.

## Phase 1: simulation

### What I wanted to know

Could a cheap plume model create intermittent sensor readings that were useful
for controlled experiments?

### What I built

The source emits groups of point-like puffs. Wind moves them downwind, random
crosswind noise spreads them, and their intensity decays. Two sensors sit in
front of the agent and rotate with its heading. I added sensor noise, random
dropout, and complete left or right failure.

The source is drawn by the renderer, but its position is not included in the
controller or PPO observation.

### What happened

Seeded runs replayed exactly. Tests covered puff motion, sensor failure,
collision handling, episode termination, and recorded histories.

### What this does not prove

The plume is not calibrated to a real chemical or airflow. Obstacles block the
agent but do not change the plume. I can use it to compare controllers, not to
make physical or biological claims.

## Phase 2: baseline controllers

### What I expected

I expected a small state machine to beat random motion in the normal world. I
also expected sensor failure to expose weaknesses in bilateral steering.

### What I built

The moth-inspired controller has three states:

- `SURGE`: move on odor and steer toward the stronger sensor;
- `ZIGZAG`: alternate moving turns after losing odor;
- `LOOP`: search a wider area after a longer odor gap.

It can use odor values and heading relative to wind. It cannot use the source
position.

### Result

On seeds 100 to 149:

| Controller | Normal success | Left sensor disabled |
|---|---:|---:|
| Random | 0% | 0% |
| Moth-inspired | 90% | 0% |

The state machine worked in the normal condition but failed with one dead
sensor. It treated the remaining signal as a left-right comparison and developed
a persistent steering bias.

### What I changed next

I made complete sensor failure a main evaluation condition instead of treating
it as a visual option.

## Phase 3: normal PPO

### Setup

I wrapped the simulator as `BioSearch-v0`. The policy gets 13 local values:
current odor, odor difference, detection memory, wind-relative heading, previous
action, odor moving averages, and obstacle rays.

I trained PPO with a two-layer 128-unit MLP over four parallel environments. The
reward includes success, step cost, collision cost, odor contact, and distance
progress. Distance is only used to calculate reward.

Training requested 100,000 timesteps and completed 100,352. It took 25.6 seconds
on CPU. I kept the best periodic checkpoint rather than the final update.

### Result

On seeds 2000 to 2049:

| Model | Success | Mean steps when successful |
|---|---:|---:|
| Selected checkpoint | 100% | 142.9 |
| Final checkpoint | 94% | 175.0 |

The selected policy then scored 0% with its left sensor disabled.

### What I learned

Perfect normal performance did not mean the policy had learned a reliable search
strategy. The failed-sensor result was more useful than another normal score.

I kept this checkpoint as the specialized control and trained a second policy
instead of replacing it.

## Phase 4: domain-randomized PPO

### Setup

The robust policy used the same PPO architecture. At every reset, the training
environment sampled:

- sensor noise between 0.01 and 0.10;
- dropout between 0 and 0.25;
- both sensors working 60% of the time;
- left or right failure 20% each;
- wind between minus and plus 15 degrees;
- one of two obstacle layouts.

Training requested 200,000 timesteps and completed 200,704. It took 61.8 seconds
on CPU. I compared the 170k and final checkpoints on separate validation seeds.
The 170k model had 75.8% pooled success, compared with 65.8% for the final model,
so I froze the 170k checkpoint for the final test.

### Final paired evaluation

The test used seeds 3000 to 3049, 50 episodes per controller and condition, and
a common 600-step limit.

| Controller | Normal | Noise | Dropout | Left off | Wind +15° | Other obstacles |
|---|---:|---:|---:|---:|---:|---:|
| Random | 2% | 2% | 2% | 2% | 0% | 0% |
| Moth-inspired | 84% | 76% | 82% | 0% | 0% | 0% |
| Normal PPO | 100% | 92% | 98% | 0% | 6% | 82% |
| Robust PPO | 74% | 96% | 68% | 68% | 72% | 92% |

### What I learned

Domain randomization created a tradeoff. It recovered sensor-failure and wind
performance, but normal success fell from 100% to 74%. Robust PPO was not a
universal replacement for Normal PPO.

The “other obstacles” result also needed a correction. Robust PPO had seen that
layout during training. The evaluation seeds were new, but every domain feature
was not new. I kept that correction in the final report.

Each training setup still has only one training seed. The 50 evaluation seeds
measure episode variation, not variation between separately trained policies.

## Phase 5: checking what PPO uses

The robustness table did not tell me whether the policies were following odor,
wind, or a route learned from the fixed source and start positions. I kept the
checkpoints frozen and changed only their observations during evaluation.

### 5.1 Hide odor

I first replaced all current and historical odor inputs with the values that
represent permanent odor absence. The physical plume and sensor readings still
ran normally.

Before evaluation, I fixed seeds 4000 to 4049 and wrote down two useful
thresholds: 10% blind success would count as strong odor dependence, while 50%
would count as substantial odor-independent behavior.

| Policy | Full input | Odor hidden | Paired change | Exact p-value |
|---|---:|---:|---:|---:|
| Normal PPO | 100% | 16% | -84 points | below 0.001 |
| Robust PPO | 68% | 50% | -18 points | 0.064 |

Normal PPO depended heavily on odor. Robust PPO still solved half the episodes
without any odor input. My first explanation was that it had learned to travel
upwind from the familiar downwind start region.

### 5.2 Hide odor and wind separately

I tested full input, odor hidden, wind hidden, and both hidden on seeds 5000 to
5049. I wrote the hypothesis and thresholds to a JSON protocol before running
the models. Exact paired tests were corrected within each policy.

| Policy | Full | Odor hidden | Wind hidden | Both hidden |
|---|---:|---:|---:|---:|
| Normal PPO | 98% | 16% | 52% | 0% |
| Robust PPO | 74% | 36% | 22% | 0% |

Both policies used both cues. Robust PPO lost 52 points when wind was hidden.
Neither policy succeeded without odor or wind.

This test had a problem. Wind is stored as sine and cosine. Replacing both with
zero removes the angle, but zero is not a valid unit vector. I could not tell how
much of the failure came from missing direction and how much came from an unusual
input.

### 5.3 Give the policy a valid but wrong wind direction

I rotated the observed wind vector by 90 degrees. Even seeds received minus 90
degrees and odd seeds received plus 90 degrees. The vector stayed on the unit
circle and the physical wind did not change. I crossed this with odor available
and odor hidden on seeds 6000 to 6049.

| Policy and input | Correct wind | Zero wind | Wind rotated 90° |
|---|---:|---:|---:|
| Normal PPO, odor available | 100% | 50% | 0% |
| Normal PPO, odor hidden | 6% | 0% | 0% |
| Robust PPO, odor available | 70% | 16% | 2% |
| Robust PPO, odor hidden | 36% | 0% | 2% |

The valid wrong angle was even more damaging than no angle. This confirmed that
the policies depend on the direction encoded by the wind values.

Wrong wind can be worse than missing wind because it gives a confident but
misleading direction. I still did not know whether Robust PPO was following a
straight upwind route or performing a wider search.

### 5.4 Reverse the source and start lanes

I moved sources to `y=3` and `y=9`. In the control condition, starts stayed near
the same lane as the source. In the second condition, starts used the opposite
lane. Both conditions had the same number of upper and lower starts, no
obstacles, and the same seed-specific heading and jitter.

Before running seeds 7000 to 7049, I expected both policies to lose at least 30
points with full input. I also expected odor-hidden Robust PPO to fall to 10% or
less in the opposite lane.

| Policy | Input | Aligned lanes | Opposite lanes | Change |
|---|---|---:|---:|---:|
| Normal PPO | Full | 96% | 42% | -54 points |
| Normal PPO | Odor hidden | 22% | 0% | -22 points |
| Robust PPO | Full | 100% | 96% | -4 points |
| Robust PPO | Odor hidden | 18% | 58% | +40 points |

My main prediction was wrong. Normal PPO was sensitive to the new relationship,
but Robust PPO kept 96% full-input success. Its odor-hidden performance increased
by 40 points.

I plotted every Robust-PPO odor-hidden path after seeing that result. This was a
follow-up analysis, not part of the original hypothesis test. In 99 of 100
episodes, the path crossed the arena centerline. The mean crosswind spans were
7.63 and 7.44 units. The paths reached the source's crosswind lane even without
odor.

That plot changed my explanation. Robust PPO was not simply travelling straight
upwind. It had learned a broad wind-oriented sweep. An opposite-lane start made
that sweep cross the source lane more often. With odor available, the policy
could turn those crossings into 96% success.

The lane test is not the same as independent geometry. An opposite start still
predicts the other source lane. The checkpoints were frozen and could not have
learned that new rule, but a continuous independent-coordinate test would be a
cleaner follow-up.

## Current interpretation

This is the explanation I would use now:

1. Normal PPO learned an efficient but narrow solution.
2. Domain randomization reduced normal performance but produced a wider search
   pattern that handled sensor failure and changed wind.
3. Both policies use odor and wind.
4. Robust PPO can search without odor because its wind-oriented sweep sometimes
   intersects the source.
5. Odor still matters because it turns that broad sweep into much more reliable
   localization.

I think the failed lane-reversal prediction is one of the most useful parts of
the project. It forced me to inspect trajectories instead of keeping the first
explanation that fit the success table.

## Main artifacts

- `results/data/milestone4_experiments.csv`
- `results/data/milestone4_summary.csv`
- `results/data/phase5_odor_blind_protocol.json`
- `results/data/phase5_cue_ablation_protocol.json`
- `results/data/phase5_wind_validity_protocol.json`
- `results/data/phase5_geometry_shift_protocol.json`
- `results/figures/milestone4_success_rate.png`
- `results/figures/phase5_wind_validity_success.png`
- `results/figures/phase5_geometry_shift_success.png`
- `results/figures/phase5_geometry_trajectory_audit.png`
- `results/data/irl_protocol.json`
- `results/data/irl_evaluation_episodes.csv`
- `results/data/irl_analysis.json`
- `results/figures/irl_success_rate.png`

## What I planned after Phase 5

I kept the existing PPO result as the stable portfolio version. At that point,
I wrote down these follow-up tests:

1. sample source and start positions independently from continuous ranges;
2. train at least three policies for each training setup;
3. train without distance-progress shaping;
4. test obstacle layouts and wind angles that never appear during training;
5. test whether a reward learned from demonstrations can replace distance
   shaping.

## Phase 6: learning from demonstrations

### Why I reconsidered it

I originally left inverse reinforcement learning out so I could finish the core
project first. After the main PPO experiments worked, I returned to the related
silkmoth and robotics literature.

The closest paper is Hernandez-Reyes et al. (2022), “Learning a Generic
Olfactory Search Strategy From Silk Moths by Deep Inverse Reinforcement
Learning.” The authors recorded male silk moth trajectories in virtual reality,
modeled them as an MDP, learned a reward with IRL, and derived a policy that was
tested in new simulated environments.

Three other papers map to experiments already in the repository.
Hernandez-Reyes et al. (2021) analyzed exploration and exploitation in
silkmoth search. Yamada et al. (2021) tested odor and wind cues when they agree
or conflict, which connects directly to my rotated-wind intervention. Shigaki
et al. (2026) transferred a context-dependent response to unilateral sensory
loss from moth experiments to a robot, which connects to my failed-sensor
evaluation.

That gives this project a clear connection:

- the failed-sensor experiment already studies behavioral compensation;
- the hidden and rotated wind tests already study cue use and cue conflict;
- the moth controller and PPO comparison already contrasts a hand-designed
  biological idea with a learned policy;
- the hand-written PPO reward is the part that IRL could replace.

### The first pilot that I rejected

I tried a small local maximum-entropy IRL pilot before adding anything to the
repository. I used 20 synthetic moth-controller demonstrations, 30 random
rollouts to estimate transitions, and 50 new normal-condition seeds for a smoke
test.

The feature-expectation gap decreased during fitting, but the learned policy
still solved 0 of 50 held-out episodes. A deterministic version repeatedly chose
forward motion. A stochastic version preserved more turning but still collided
far more often than the demonstrator.

This did not show that IRL is unsuitable. It showed that my quick abstraction
was unsuitable. The first version compressed the controller's history-dependent
casting and looping rhythms into a few bins. A second version kept the visible
timing phases but still had weak transition coverage and reward ambiguity.
Aggregate feature matching could assign the wrong sign to correlated cues while
still lowering its optimization error.

I removed the pilot instead of presenting a valid-looking algorithm with a bad
behavioral result. Fast code was not the missing part. The missing part was a
state representation and evaluation protocol that matched the research
question.

### The behavior-cloning gate

I kept the first 100 successful moth-controller episodes starting at seed 8000.
It took 114 attempts and produced 25,298 transitions. Failed attempts remained
in the metadata but did not become expert transitions.

Behavior cloning received the same 13 local observations used by PPO. It did
not receive controller state, source position, source distance, or the
hand-written PPO reward. A class-balanced loss prevented the frequent forward
action from overwhelming the turning actions.

On 20 different successful demonstrations starting at seed 11000, deterministic
action agreement was 64.9%. On validation seeds 9000 to 9029, the cloned policy
reached 83.3% success. That passed the learnability gate. The demonstrations
contained enough information to produce a working closed-loop policy even
though exact action agreement was imperfect.

### The first AIRL run failed differently

I then implemented AIRL with a state-action reward network and a potential
network. The discriminator used the standard shaped form
`g(s,a) + gamma*h(s') - h(s) - log pi(a|s)`. PPO received this learned reward
instead of the environment reward.

Starting AIRL from a random policy failed. Discriminator accuracy quickly rose
above 0.9 while the generator converged toward moving forward. The selected
policy solved only 18% on the diagnostic range and averaged hundreds of
collision events. The reward network could recognize expert transitions, but
the generator did not explore enough expert-like behavior to learn from it.

I did not keep that policy as the reported result.

### Stabilized AIRL protocol

The behavior-cloning result suggested a direct fix: initialize the AIRL
generator from the BC policy instead of a random policy. I also reduced the
reward network from 128 to 64 hidden units, discriminator epochs from eight to
two per round, and discriminator learning rate from `3e-4` to `1e-4`.

The learned reward replaced the hand-written reward, but the simulator still
ended an episode on source contact or the time limit. Physical success on the
validation seeds selected the checkpoint. AIRL therefore learned the stepwise
reward, not the definition of the task boundary.

I ran six rounds of 4,096 PPO steps. Each round used at least 8,192 fresh
generator transitions. Validation selected round one at 93.3% success. Later
rounds ranged from 66.7% to 90%, so continuing adversarial training did not
consistently help.

The earlier failed run had already exposed the first final seed range. I moved
the final comparison to untouched seeds 15000 to 15049 and wrote that split into
the saved protocol.

### Final demonstration-learning result

| Policy | Success | 95% Wilson interval | Mean successful steps | Mean collisions |
|---|---:|---:|---:|---:|
| Moth demonstrator | 88% | 76.2% to 94.4% | 258.0 | 62.0 |
| Behavior cloning | 82% | 69.2% to 90.2% | 254.2 | 86.8 |
| BC-initialized AIRL | 92% | 81.2% to 96.8% | 257.5 | 33.1 |
| Normal PPO | 100% | 92.9% to 100% | 134.0 | 0.0 |
| Robust PPO | 78% | 64.8% to 87.2% | 159.5 | 0.0 |

AIRL succeeded on seven seeds where BC failed. BC succeeded on two seeds where
AIRL failed. The paired change was 10 percentage points, but the exact McNemar
p-value was 0.1797. I treat the result as a promising numerical improvement, not
a conclusive difference.

The collision result is descriptive too. AIRL reduced BC's mean from 86.8 to
33.1 collision events, but one trained policy cannot establish a stable method
effect. Normal PPO was still faster and had no collisions on this range.

### What this phase shows

The implemented result answers a smaller question than biological IRL:

1. Synthetic moth-controller demonstrations are learnable from local inputs.
2. AIRL can infer a reward that supports successful search without source
   distance or the hand-written environment reward.
3. BC initialization matters in this experiment. AIRL from scratch failed.
4. The final success advantage over BC needs more training seeds and episodes.

This does not recover a moth's reward. It recovers a useful reward from my own
synthetic controller. Real moth trajectories, a matching observation model,
and independently trained policies would be required for a biological claim.

### IRL artifacts

- `models/irl/bc_policy.zip`
- `models/irl/airl_policy.zip`
- `models/irl/airl_reward.pt`
- `results/data/irl_protocol.json`
- `results/data/irl_training.csv`
- `results/data/irl_evaluation_episodes.csv`
- `results/data/irl_evaluation_summary.csv`
- `results/data/irl_analysis.json`
- `results/figures/irl_success_rate.png`

### References used for this phase

1. Hernandez-Reyes, C. et al. (2022), “Learning a Generic Olfactory Search
   Strategy From Silk Moths by Deep Inverse Reinforcement Learning,” *IEEE
   Transactions on Medical Robotics and Bionics*.
   [doi:10.1109/TMRB.2021.3129113](https://doi.org/10.1109/TMRB.2021.3129113)
2. Fu, J. et al. (2018), “Learning Robust Rewards with Adversarial Inverse
   Reinforcement Learning,” *International Conference on Learning
   Representations*. [arXiv:1710.11248](https://arxiv.org/abs/1710.11248)
3. Shigaki, S. et al. (2026), “Insect-inspired adaptive behavioral compensation
   strategy against olfactory sensory deficiency for robotic odor source
   localization,” *npj Robotics*.
   [doi:10.1038/s44182-026-00080-5](https://doi.org/10.1038/s44182-026-00080-5)
4. Hernandez-Reyes, C. et al. (2021), “Identification of Exploration and
   Exploitation Balance in the Silkmoth Olfactory Search Behavior by
   Information-Theoretic Modeling,” *Frontiers in Computational Neuroscience*.
   [doi:10.3389/fncom.2021.629380](https://doi.org/10.3389/fncom.2021.629380)
5. Yamada, M. et al. (2021), “Multisensory-motor integration in olfactory
   navigation of silkmoth, Bombyx mori, using virtual reality system,” *eLife*.
   [doi:10.7554/eLife.72001](https://doi.org/10.7554/eLife.72001)

## What I would test next

1. Repeat Normal PPO, Robust PPO, BC, and AIRL with at least three training
   seeds each.
2. Sample source and start coordinates independently from continuous ranges.
3. Train PPO without distance-progress shaping.
4. Test the learned AIRL reward under sensor loss and cue conflict.
5. Replace synthetic demonstrations with validated moth trajectories only after
   confirming their experimental conditions and license.
