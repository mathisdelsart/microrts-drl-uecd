# Speaker script: 20-minute defense

Target: **20:00.** ~2510 spoken words at ~128 words/min; the full run lands at
**~19:30**, leaving a small buffer to slow down (Part 8 ends ~11:58). Slides = the 52
numbered content slides (title, outline, section transitions, image credits and thank-you
are extra). (Struck-through lines are skipped aloud; presenter cues are in (parentheses).)

| Part | Slides | Words | Speech | Cumulative |
|------|--------|-------|--------|-----------|
| Title + Outline | -- | 58 | 0:27 | 0:27 |
| 1. Context & Objective | 5 | 233 | 1:49 | 2:16 |
| 2. Background | 5 | 218 | 1:42 | 3:58 |
| 3. State of the Art | 3 | 153 | 1:11 | 5:09 |
| 4. Contributions | 2 | 58 | 0:27 | 5:36 |
| 5. Evaluation Framework | 4 | 223 | 1:44 | 7:20 |
| 6. Environment Stack | 4 | 187 | 1:27 | 8:47 |
| 7. UECD Architecture | 6 | 260 | 2:01 | 10:48 |
| 8. Training Recipe | 3 | 149 | 1:10 | 11:58 |
| 9. Results & Interpretation | 17 | 775 | 6:02 | 18:00 |
| 10. Discussion & Conclusion | 3 | 196 | 1:32 | 19:32 |
| **Total** | **52** | **2510** | **19:32** | |

---

# Part 0. Introduction  [target 0:20]

## Title  [0:13]

> Hello everyone, and thank you for being here. It is a real pleasure to present
> to you the work I completed for my master's thesis, on deep reinforcement
> learning for competitive agents in MicroRTS.

## Outline  [0:07]

> Here is the complete path I will take for this presentation. I have quite a lot to
> cover, so let's not lose any time.

---

# Part 1. Context & Objective  [target 2:40]

> Let me start first with the context and objectives of this thesis.

## Real-time strategy (RTS) games  [0:40]

> In a real-time strategy game you gather resources, build an economy and command an
> army, all at once and in real time. So you need two skills together:
> macro-management and micro-management. You probably
> know iconic titles like StarCraft II or WarCraft III. But beyond these games, it
> represents a benchmark for real-time sequential decision-making, a step toward
> real-world problems.

## Why RTS is hard for AI  [0:25]

> It is hard for AI for a huge number of reasons, for instance a combinatorial action
> space, real-time decisions, and very sparse rewards. The slide lists more, and
> together they make RTS a uniquely demanding benchmark.

## Why MicroRTS?  [0:25]

> But you could say, why MicroRTS, and not StarCraft II? Because it keeps the full strategic depth, but
> strips away the graphics and the noise of a commercial game. So it is faster and
> affordable on an academic budget, while still having an active international
> competition. That makes it an ideal research testbed.

## Units, combat, and counters  [0:20]

> The game itself has workers for the economy, buildings for production, and three military
> unit types for combat, locked in a rock-paper-scissors of counters, as shown in the
> illustration. In principle this should push the agent toward a mixed army
> composition.

## Research questions  [0:20]

> This frames my two research questions.
>
> The first one studies how the agent's design shapes its performance and robustness.
>
> The second asks whether I can be competitive within an academic compute budget.

---

# Part 2. Background  [target ~2:25]

> Let me very briefly set up the two tools behind the whole thesis:
> reinforcement learning, and the PPO algorithm.

## What is deep reinforcement learning (DRL)?  [0:40]

> Deep reinforcement learning is a loop where an agent acts in an environment and learns
> purely from the reward, by trial and error. "Deep" just means its policy is a neural
> network, and I formalize the loop as a Markov decision process, where the goal is to
> maximize the reward collected over time.

## MicroRTS as a decision problem (MDP)  [0:30]

> MicroRTS fits this MDP formulation naturally.
>
> The state is the whole board plus the clock.
>
> The action is one command per unit.
>
> The transition is the game engine itself.
>
> The reward is sparse.
>
> And gamma sets how far ahead the agent looks.

## The PPO Algorithm  [0:20]

> To learn the policy, I use PPO. But why PPO, and not another algorithm? Because it's the
> standard on-policy algorithm for MicroRTS, and it gives a stable training loop.

## PPO: on-policy, model-free, actor-critic  [0:30]

> PPO has three defining properties:
>
> Model-free, meaning it learns from playing, without modelling the game.
>
> On-policy, meaning it is trained only on fresh data.
>
> Actor-critic, meaning there is an actor that picks actions and a critic that judges them.

## The PPO training loop  [0:10]

> This is the whole PPO loop on MicroRTS. I'll skip the notation; the idea is simply
> play, learn, and repeat.

---

# Part 3. State of the Art  [target ~1:45]

> A lot of research has gone into playing RTS games. This part reviews that landscape.

## Classical RTS AI: three hand-designed paradigms  [0:30]

> The classical approach is hand-designed agents, in three families.
>
> Rule-based, which follow fixed hand-written rules.
>
> Search-based, which plan by simulating the game forward.
>
> And hierarchical, which split macro and micro-management.
>
> They are strong, but their behaviour is hand-designed in advance.

## DRL: from board games to RTS  [0:25]

> Over the last decade, Deep RL has mastered harder and harder games: first Atari
> with DQN, then the board games with AlphaGo and its successors. And in
> 2019 it finally cracked real-time strategy, with AlphaStar in StarCraft II and
> OpenAI Five in Dota 2. But the catch is that both needed an enormous amount of
> compute.

## DRL on MicroRTS: the line we extend  [0:30]

> However, Deep RL on MicroRTS is still rare; most entries are rule- or search-based.
> Two agents stand out: GridNet, in 2021, opened up reinforcement learning on MicroRTS with
> its bridge, and RAISocketAI was the first deep-RL agent to win the 2023 competition.

---

# Part 4. Contributions  [target 0:30]

> Let me now begin with my own contributions.

## Six contributions  [0:20]

> My work has six main contributions, plus a broad literature review and two extra
> experiments. I will not list them now; each will come up naturally in the rest of
> the presentation.

## A paper accepted at IEEE CoG 2026  [0:10]

> Several contributions were condensed into a paper, accepted at the 2026 CoG
> conference, held in Madrid in September.

---

# Part 5. Evaluation Framework  [target ~1:30]

> My first contribution is how I measure performance, because a weak measurement
> means weak conclusions.

## A reproducible CoG-style tournament  [~0:25]

> The 2026 competition pivoted to LLM-based agents, so I rebuilt the classic
> round-robin tournament myself, as shown in the illustration.
>
> It is anchored on the 2023 edition, with 10 games per matchup.

## Twelve maps, one focus  [~0:20]

> This edition ships twelve maps of different size and layout. My main agent focuses
> on a single one, shown here, named basesWorkers 16-by-16 A. The others come back
> later, for the generalization tests and the multi-map experiments.

## Fifteen benchmark agents, four paradigms  [~0:20]

> So I have a tournament, but against which agents? I gathered many open-source
> agents, fifteen opponents in total. Four are built-in baselines, from random up to a
> simple search. The other eleven are real competition winners,
> mixing the classical and deep-RL paradigms I showed earlier.

## Beyond win rate: four game-theoretic metrics  [~0:25]

> For the metrics, I use the win rate, but win rate alone can be misleading, because
> strategies can form cycles, and a single number cannot capture every aspect that makes
> an agent good and robust. So I add four game-theoretic metrics, each a different perspective.
>
> Nash averaging asks how exploitable you are against the optimal mixture of opponents.
>
> alpha-Rank asks who dominates as strategies evolve over time.
>
> Copeland counts how many opponents you beat, regardless of margin.
>
> And regret measures how far you are from the best possible counter.

---

# Part 6. Environment Stack  [target ~1:20]

> My second contribution is the environment stack the agent trains in. I reworked it
> in many ways; here are the changes that matter most.

## An extended Java-Python bridge  [~0:25]

> First, the bridge between the Java game and my Python code. One vectorized client
> steps many games at once on a shared Java Virtual Machine, running self-play and agent-versus-bot
> in the same batch. That is what makes large-scale training
> practical.

## Richer observation: 29 to 73 channels  [~0:20]

> Second, what the agent sees. Every cell on the board is a stack of planes. The
> standard encoding has 29 binary channels, as shown in the illustration, and I extend
> it to 73, adding continuous values and thresholds a one-hot cannot express.

## Smarter masking: avoid collisions  [~0:20]

> Third, what the agent is allowed to do. A mask already removes illegal moves. I
> make it destination-aware, so no two units go to the same cell, and the agent sees
> the pending conflicts. Remember these two: they will be among the strongest gains later.

## Optional training mechanisms  [~0:15]

> The stack also ships optional, flag-gated mechanisms.
>
> For example: zero-padding for multi-map training, a shaped reward composed of added
> dense rewards on top of the basic sparse one, and wrappers like frame stacking and
> symmetry augmentation.

---

# Part 7. The UECD Architecture  [target ~2:35]

> This brings me to the architecture, my third contribution.

## An RTS network needs three kinds of reasoning  [~0:25]

> To play an RTS well, a network must reason three ways at once.
>
> Locally, convolutions handle the tactics around a unit.
>
> Relationally, a Transformer lets units coordinate across the map.
>
> Globally, self-attention weighs the trade-offs between regions.
>
> No standard layer covers all three.

## From GridNet to UECD  [~0:25]

> I build it up from the GridNet baseline, adding the three axes I just showed, one
> component at a time, until I reach UECD.

## The UECD architecture  [~0:20]

> Here is the complete UECD architecture, with 4.7 million parameters.
>
> You can see the U-Net backbone with its CBAM attention in blue and orange, the entity
> Transformer in green, and the bottleneck self-attention in red.
>
> I will explain each one in the next slides.

## Entity Transformer: relational reasoning  [~0:30]

> The entity Transformer turns every unit into a token, concatenating its raw observation,
> the CNN features at its cell, and its normalized position, then lets them attend to one
> another at any distance. The enriched tokens are then scattered back into the map at
> two scales, one finer for tactics and one coarser for strategy. That gives the relational
> axis, and it preserves each unit's identity, which convolutions blur away through pooling.

## CBAM: attention on what matters  [~0:15]

> CBAM is a cheap attention module that gates the backbone. It tells the network what to
> look at and where.
>
> The what comes from channel attention, and the where from spatial attention,
> highlighting, for example, frontlines and exposed units.

## Global reasoning: seeing the whole map  [~0:40]

> Convolutions are blind to the far side of the map, so I add a self-attention layer at
> the bottleneck, where every region attends to every other, cheaply. The depth is
> pyramidal, so most of the capacity sits there, summarizing the whole map.

---

# Part 8. Training Recipe  [target ~1:15]

> My fourth contribution is the training pipeline that drives all of this.

## A modular training pipeline  [~0:25]

> The idea here is modularity, because the changes are everywhere: they touch the
> bridge, the architecture, and PPO itself, as you can see in the illustration.
>
> I keep one fixed PPO baseline, and put every mechanism on top of it, off by default.
>
> I then ablate them in isolation.

## A catalog of mechanisms, six axes  [~0:25]

> ~~Altogether, about twenty such mechanisms, grouped into six families: action sampling, reward and value shaping, opponent curriculum and self-play, auxiliary tasks, generalization, and architecture refinements. I won't go through them all; what matters is that each one is studied in isolation.~~
>
> This slide details the catalog of training mechanisms, grouped into six families. I will
> not have time to detail them all.

## MCW: focus the gradient on close matchups  [~0:25]

> But let me just highlight one, novel and among the strongest: Matchup Competitive
> Weighting. It focuses learning on the games still in the balance, weighting each
> transition by how close its matchup is, highest around fifty-fifty and lowest once
> the game is decided. And it only reweights the policy-gradient term, so the value
> estimates stay untouched, keeping the critic accurate on every game.

---

# Part 9. Results & Interpretation  [target ~7:40]

> Which finally brings me to the results.

## Architecture ablation: +21.5 pp  [~0:25]

> First, the architecture on its own, with no extra features.
>
> This slide is dense, but the takeaway is simple: going from GridNet up to UECD lifts the
> win rate from 65 to 86 percent, purely from the network.
>
> And every component adds a positive gain, even if the variance is sometimes high.

## Feature ablation: signal beats capacity  [~0:25]

> This figure shows all the feature ablations, but the conclusion is simple: the biggest
> gains come from better perception and a sharper training signal, not from a bigger model.
>
> The top four features each add around twenty points, but cost almost no extra parameters.

## Single-map agent: a discount-induced rush collapse  [~0:35]

> (Take this one slowly.)
>
> I scaled that up to 300 million steps, with self-play, and I annealed the shaped reward
> to zero, to let it optimize the true sparse signal.
>
> And instead of getting better, it collapsed. It threw away its good play and converged
> to a degenerate worker-rush.
>
> You can see it on the graph: the episode length drops sharply, and the signals for good
> behaviour fall to zero.

## Rush fragility on held-out opponents  [~0:20]

> You can also see it on the left: the training opponents in green stay perfect, while the
> unseen opponents in red collapse. So it overfits the training distribution instead of
> learning a strategy that transfers.

## Why it collapses: a discount-induced shortcut  [~0:10]

> (Say this one slowly.)
>
> The collapse has a formal explanation. Once the shaped reward is gone, the only thing
> left to optimize is the discounted win, at the very end of the game.
>
> And because the discount is below one, the easiest way to make that bigger is simply to
> end the game sooner, no matter how well the agent plays.
>
> And this shortcut slips through the advantage, which PPO's clipping does not stop.
>
> So this collapse comes directly from discounting a pure sparse reward.

## UECD-Best: two-phase fine-tuning  [~0:30]

> To fix this issue, I go back to a checkpoint from before the collapse, and I keep a
> floor of 10 percent on the shaped reward.
>
> Then I fine-tune in two short phases: first broadening the opponent pool, then
> hardening specifically against RAISocketAI.
>
> The takeaway is that the final agent, called UECD-Best, takes 350 million steps and 9.47
> GPU-days in total.

## UECD-Best in action on basesWorkers16x16A  [~0:25]

> So before the numbers, let me show you the result.
>
> Here is a match against RAISocketAI.
>
> And as you can see, my agent plays only Ranged units, showing how dominant they are.

## Tournament: UECD-Best tops the field  [~0:20]

> In the full tournament, UECD-Best tops the field: first of nineteen agents, with
> almost 97 percent overall win rate. It finishes ten points ahead of RAISocketAI.
>
> But to be clear: this is a strong single-map specialist, not a general MicroRTS
> champion.

## Tournament: the head-to-head matrix  [~0:30]

> This head-to-head matrix is dense, but the takeaway is simple: it beats every single opponent.
>
> As you can see in the table, two cells need context, because the deterministic format
> misleads both ways: it looks over-confident against RAISocketAI, and under-confident
> against WorkerRush. Everywhere else, the win is clear.

## Beyond win rate: Copeland and Nash  [~0:30]

> Still on this single map. Copeland measures pairwise dominance, and I rank first. Nash
> measures worst-case robustness, and I sit at zero, on the non-exploitable frontier.

## Beyond win rate: alpha-Rank  [~0:15]

> alpha-Rank measures evolutionary stability, and again I come first, on this map.

## Beyond win rate: regret  [~0:20]

> And finally, regret measures the robustness gap: I am first on average, and second only
> on the worst case, because of the WorkerRush artifact I just showed.
>
> So across all four metrics I lead. But again, this is one map: a strong single-map
> specialist, not yet a general MicroRTS champion.

## Generalization: two distribution-shift probes  [~0:15]

> I then tested the single-map agent on two maps it never saw: a bigger grid, and a
> different layout.

## Generalization: scale vs layout  [~0:25]

> Scaling up to 32-by-32 is fine, the strategy transfers. But a new layout makes the
> win rate collapse: single-map training overfits the shape of the map far more than
> its size. Still, it is never truly broken: it keeps playing coherent, just suboptimal
> games, and only loses to the strongest opponents.

## The five-map training pool  [~0:15]

> To try to overcome this problem, I ran a second agent, this time on a pool of five
> maps across three sizes.

## Extra experiment #1: multi-map training  [~0:45]

> Training on all five padded maps removes the per-map collapse. You can see it on the
> basesWorkers 8-by-8 map, where the single-map agent has only a 15% win rate, against 66%
> for the multi-map agent.
>
> Overall, the multi-map agent beats the single-map one by 17%. It still ranks only fifth
> and loses to the strongest agents, but that is on just 200 million steps split across five
> maps, so more budget should lift it.

## Extra experiment #2: behaviour-cloning warm-start  [~0:06]

> And a last experiment, on warm-starting. I pre-train the agent by imitating 300 bot
> games, then continue with PPO.
>
> It clearly accelerates the early phase, 88 against 28 percent at 30 million steps, but it
> does not lift the final plateau. So behaviour cloning buys you compute-efficiency, not a
> higher ceiling.

---

# Part 10. Discussion & Conclusion  [target ~1:30]

> Now that the results are in, let me step back and discuss the limitations, the
> future work, and the conclusion.

## Honest limitations  [~0:35]

> The limits come in two kinds. Some are deliberate choices: I scoped everything to
> MicroRTS, with full observability and pure reinforcement learning, and I kept the
> final agent on a single map. The others are real open gaps: the
> play style is monotone, the agent's policy is sensitive to the map layout, and the
> multi-map agent is not yet competition-grade.

## Future work: four directions  [~0:25]

> Each of those gaps points to a direction, and I propose four: combining deep RL
> with language models, decomposing the policy hierarchically, adding a
> neuro-symbolic layer, and generating adversarial maps to train against. Those are
> the short-term directions. The long-term direction would be a general, Ludii-style
> framework for RTS games.

## Conclusion  [~0:25]

> So, what are the answers to the two research questions?
>
> For the design: the gains that matter come from better perception, competitive
> self-play, and a relational, multi-scale architecture.
>
> For the budget: yes, my agent beats the strongest prior competitor on a fixed map,
> for about 9.5 GPU-days.

## Image credits

> Here are the credits for the few images that are not my own.

## Thank you  [0:02]

> Thank you all for your attention.

---
