# Project retrospective

**Ted Roper · 2026-07-29 · Hands-Free *Five Nights at Freddy's***

> **Where it ended:** Night 1 of FNAF was completed **hands-free** — cursor and
> clicks both driven by my hand through a webcam, no keyboard or mouse. It ran on
> `fistvsrest_frozen_mnv3_large` (AD-21) and it was **not clean**: missed clicks,
> re-squeezes, and stretches where I was fighting the cursor rather than playing
> the game. There is no recording and the session wasn't instrumented, so this is
> a qualitative result and I'm reporting it as one. It is also worth saying plainly
> that the checkpoint that beat the game is **not** the registered `champion` —
> that model scores 100% on the offline test set and has still never been
> validated live.

**What I'm most proud of technically** is not a model score — it's that this
project twice refused to let me publish a wrong conclusion, and the mechanism was
something I built on purpose. In Week 4 my new tuning harness beat the deployed
baseline by ~5 points on the first two trials. That would have been the headline
of the report. Because I'd insisted the harness always re-run the *old* recipe as
a control, the drift check fired instead, and the cause turned out to be timm's
head initialization on a 2-class head (`r = 1/sqrt(out_features)` → ±0.707
weights → ±27 logits → CE 2.70 at init), not hyper-parameter search: same recipe,
same seed, init the only change, val 0.9221 → 0.9740. The search's real
contribution was ~1 point, and that's how it went in the report — as a three-rung
ladder (AD-22). A week later the same discipline pointed at *me*. I was certain
the tuned model was overfit; the evidence said the train/serve **crop geometry**
was wrong. Training crops HaGRID's annotated bbox, the live loop crops
MediaPipe's 21-landmark hull, both at the same `pad = 0.15` — and a hull is
**0.768×** an annotated box, worst on `fist` (**0.705**), the one class that fires
clicks. The measurement I'm proudest of is that one, because HaGRID's annotations
carry the landmarks as well as the bbox, so the live crop was recoverable
**offline, over 27k–29k images per gesture, with no webcam and no model** — and it
explained my exact complaint: median p(fist) barely moved (0.927 → 0.984) but the
**10th percentile went 0.464 → 0.932**, and a K-frame confirm dies on the bottom
decile, not the average. Fixing the pad took fist click-rate 81.5% → 94.4% while
false clicks *fell* 0.7% → 0.0% (AD-25).

**The biggest challenge was the train/serve gap, and it beat me twice before I
learned to measure it instead of arguing with it.** The first time was the
architecture: a full-frame classifier reached 53% test because the hand is under
5% of a HaGRID frame, and cropping to the hand took the same frozen backbone to
91.7% — a controlled A/B that rewrote the pipeline into two stages (AD-04). The
second time was the crop-geometry bug above, which had been sitting in the §A.3
contract as "same relative tightness" for four consecutive strategies without
anyone checking whether that was true. What I actually learned from those is
narrower and more useful than "watch out for domain shift": **the gap is almost
never where the intuition points, and it's usually cheaper to measure than to
theorize about.** The most embarrassing instance had nothing to do with ML at all
— late one session every timing went ~2.1× slower at once, uniformly, on an idle
CPU. The laptop was on battery, pinned at its 1.4 GHz base clock, and the loop
dropped from 28.3 FPS to 9–11: *with every latency fix applied, on battery, it was
slower than the unfixed loop on AC.* Power state was the single largest term in my
frame budget. Now `play.py` warns at startup and the benchmark stamps AC/battery
and CPU clock onto every report, because a number I can't compare is a number I
shouldn't have taken.

**With five more weeks**, the first thing I'd do is the fix I deliberately didn't
take: **retrain on landmark-hull crops** so the live box matches training by
construction (1.00×) instead of by a scalar pad that's necessarily a compromise —
the equalizing pad is 0.42 for `fist` and 0.27 for `palm`, and the loop can't know
which it's looking at before classifying. I skipped it because it re-bases every
accuracy number this project has published, and I didn't want to spend the last
week invalidating my own evidence base. Second, I'd **instrument the live run**.
The single weakest thing in this project is that the headline result — Night 1 —
is the least measured thing in it: I have 11 timed obstacle-course trials and zero
logged frames from the actual game. Per-frame confidences and FSM transitions
during real gameplay would turn "rough edges" into a distribution, and would
settle whether the 100%-offline champion or the model that actually beat the game
is the better controller. Third, I'd get **other people's hands** into the data —
every trial in that dashboard is mine, and a controller evaluated by its author is
a controller with an unknown usability floor. Fourth, a self-capture fine-tune at
arm's length, which is the regime HaGRID simply doesn't contain and the one the
game is played in.

**Against my Week-1 takeaway, I got both halves, and the second one taught me
more than I expected.** I wanted to stop treating pretrained models as black
boxes and be able to justify every layer decision — and I can: frozen backbone
first, head-only linear probe on cached features, then two blocks unfrozen with
augmentation, with the freeze schedule, the label smoothing (0.062, which caps
achievable confidence near 0.969 and is why the better model reads *lower* on the
HUD), and the head init all decisions I made deliberately and can defend. That
part went as planned. The part I underestimated was the second goal — bridging an
offline model to a live system — because it turned out that almost nothing that
mattered after Week 3 was a modeling problem. It was crop geometry, thread pools
starving MediaPipe, a 0 ms click pulse a 60 Hz DirectX game can't see, 67 ms of
camera backlog, CPU power states, and a stuck mouse button caused by a `tick()`
that was never called. My Week-1 note also said I wanted to learn to interface ML
into something other than a dashboard or a notebook; I ended up doing both — a
real commercial game driven by simulated input, *and* a web app that hands the
controller to a stranger and scores them. The strongest habit I'm taking out of
this is the cheapest one to state and the hardest to actually do: **before you
believe a new number, make the new code reproduce the old one.**
