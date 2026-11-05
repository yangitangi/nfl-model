# Pre-registration 2: 2026 shadow run, L1 re-test, line-movement lead

Written 2026-11-05 (about 00:40 UTC), before the first prospective game (2026 week 9 TNF, JAX @ BAL, kickoff
2026-11-06 01:15 UTC). No forward outcomes existed when it was written. Frozen model parameters are unchanged from
`rebuild/PREREGISTRATION.md` / `rebuild/session4_frozen.json`.

The data comes from `rebuild/shadow_tracker.py`:
- logs append to `rebuild/shadow_log/`;
- the prediction of record is the last row logged before kickoff;
- git commit times on branch `rebuild/shadow-log-2026` are the audit trail.

## Operating rules (fixed)

- **F1** is the frozen ridge (alpha 1e4, k 0.25, 66 features), fitted once on 2010-2025. It is never refitted
  during 2026.
- **Line at prediction** = the nflverse `spread_line` when the tracker runs. Grading uses the nflverse final
  (closing) `spread_line`.
- **Starter QB** = the nflverse listed starter for the game.
- **A prediction is logged only when both teams' injury reports for that week are published.** The frozen F1
  learned large coefficients for the rare `inj_report_missing` flag from only 9 training team-games: about
  +0.54 pts (home) and −1.05 pts (away) after k. An early-week prediction would therefore carry that artifact.
  This is an operating rule, not a model change, and it was found on the first live run before any outcome
  existed.

  A game that reaches kickoff without a logged prediction counts as "missed" and is reported.
- **Prospective sample** = 2026 week 9 onward. Weeks 1-8 were a retrospective backfill (session 4) and are
  excluded from every test below.
- **Run schedule:** Tuesday and Thursday afternoons and Sunday morning (US Eastern). The schedule is operational
  and can change without affecting the tests.

## D1: switch decision for F1 vs production

At the end of the 2026 regular season, on prospective games (weeks 9-18, about 150 games), **recommend replacing
production XGBoost with F1** if all of the following hold:
1. F1 MAE minus market MAE (closing line) ≤ +0.05;
2. F1 MAE minus production MAE < 0 (point estimate; production's logged prediction on the same games);
3. at least 90% of games have an F1 prediction logged before kickoff.

This is a robustness decision, not an edge claim. The SE of (1) at n ≈ 150 is about 0.02.

## L1b: re-test of "fade A-lite's largest edges" (confirmatory, forward data only)

- **Definition:** identical to L1 in `rebuild/PREREGISTRATION.md`:
  - frozen A-lite parameters;
  - z from the start-of-week state;
  - top quintile |z| ≥ 0.3666252761195482;
  - bet against A-lite's side.

  Graded at the nflverse closing line; pushes excluded. z is the one logged in the prediction of record.
- **Sample:** prospective games from 2026 week 9 through the end of the **2027 regular season**.
- **Single analysis point:** after the 2027 regular season. Interim tallies in `SUMMARY.md` are descriptive and
  must not stop or extend the test.
- **Pass rule:** fade ATS ≥ 52.4% **and** n ≥ 100 **and** the 95% interval excludes 50%. Otherwise "no edge".
- **Power, stated honestly:**
  - About 24% of games fall in the top quintile, roughly 65 bets per season, so the expected n at the analysis
    point is about 100.
  - At n = 100, passing needs an observed rate of about 60% or more.
  - If the true rate is 53.5% (the DEV estimate), the chance of passing is about 10%. If the true rate is 60%, it
    is about 50%.
  - This test can only confirm a *large* effect. Detecting 53.5% with 80% power needs about 1,260 bets, about 19
    seasons.
  - **Correction to the session 4 report:** its recommendation #4 said "about 400 bets, about 2 seasons". That was
    wrong: 400 bets is about 6 seasons, and even 400 bets gives under 50% power at 53.5%.

## L2: line movement, "follow the move" (confirmatory, forward data only)

- **Opening line:** the tracker's first capture of the game made at least 48 h before kickoff. Captures run
  Tuesday, Thursday and Sunday, so this is normally the Tuesday line.
- **Closing line:** the nflverse final `spread_line`.
- **Move:** closing minus opening.
- **Bet:** if |move| ≥ 0.5, take the side the line moved toward (move > 0 means toward the home team), graded
  ATS at the closing line. Pushes are excluded. Games without a ≥ 48 h capture are excluded.
- **Hypothesis:** the closing line under-reacts to the information that moved it, so following the move covers
  more than 52.4%.
- **Sample:** prospective games from 2026 week 10 through the end of the 2027 regular season. Week 9 has no
  Tuesday capture.
- **Single analysis point:** after the 2027 regular season.
- **Pass rule:** ATS ≥ 52.4% **and** n ≥ 100 **and** the 95% interval excludes 50%.
  - A significant result in the *opposite* direction (fading the move wins) is **not** a pass. It is reported only
    as a new hypothesis for a new pre-registration.
- **Power:**
  - Production's snapshot-to-close moves were ≥ 0.5 pts in only 16% of 2026 weeks 1-8 games (the movement
    distribution was checked; no outcomes were looked at). With a 2-5 day lead, more games should qualify; about
    25-35% is expected, i.e. roughly 70-95 bets per season and about 120-150 by the analysis point.
  - Like L1b, this can only confirm a large effect (an observed rate of about 58-60%).
- **Known limitation:** nflverse lines are a consensus feed with irregular update times, not a timestamped
  sportsbook feed. A real line-movement strategy would need the actual price available when the bet is placed.

## What would change this plan

Nothing in the interim data. The only allowed changes are operational:
- run times;
- fixing a crash, documented in a commit;
- stopping the shadow if production is retired.

Any new idea goes into a new pre-registration file with its own forward sample.
