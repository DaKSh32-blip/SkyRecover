# SkyRecover Learning Log

My notes on how SkyRecover works, why it was built this way, and what I'd say about it in an interview. One section per piece, added as the project grows.

---

## Piece 1: Getting the data

### What it does
- `download.py` pulls real flight records from the **US Bureau of Transportation Statistics (BTS)**. Every US airline has to report every domestic flight: scheduled time, actual time, delay, cancellation, and which aircraft flew it. BTS publishes one zip per month.
- `clean.py` turns the messy raw file into one clean table, one row per flight.
- `config.py` holds the shared rules every module uses.

### Key ideas
- **Tail number = the aircraft's license plate.** It's what lets us follow one plane through its whole day, flight after flight. Without it, there's no way to see one delay ripple into the next flight. That's why rows with no tail number are dropped.
- **"Delayed" means 15+ minutes.** That's the official US DOT definition, so my numbers line up with official on-time stats.
- **Minimum turn time = 35 minutes.** The least time a plane needs on the ground to unload, clean, refuel and board. It's an average for narrow-body jets, and a starting assumption I can refine later.
- **Diverted flights are dropped**, because they landed at a different airport and their arrival delay doesn't mean anything.
- **Times come as numbers like `1430`.** `hhmm_to_minutes` turns that into minutes after midnight: 14 x 60 + 30 = 870.

### Questions I should be ready for
**Q: Where does your data come from?**
Real US government data from BTS. Every domestic flight is reported by law. I used Delta's flights from January to March 2024.

**Q: Why not just use a Kaggle dataset?**
BTS is the original source, it updates every month, and it includes tail numbers, which I need to follow each aircraft. Pulling it directly also shows I can build a real data pipeline, not just open a CSV.

---

## Piece 2: Exploring the real data, and the time-zone bug

### What the data says (Delta, January 2024)
| Fact | Number |
|---|---|
| Flights | 74,200 |
| Aircraft | 920 |
| Airports | 142 |
| Departures from Atlanta | 16,469 (about 1 in 5 of all Delta flights) |
| Flights per plane per day | 3.4 on average, up to 10 |
| Cancellation rate | 0.64% |

### A data check that passed
97% of the time, a plane's next flight departs from the airport its previous flight landed at. That proves the tail numbers really let us chain each aircraft's day together. (The other 3% are things like overnight repositioning or swaps, which I treat as "no connection.")

### The bug (my best interview story so far)
**The problem:** BTS records every time in **local time at that airport**. My first version calculated landing time as "departure time + flight time," which mixes two time zones.

```
Leaves Atlanta 10:00 (Eastern) + 4h50 flight = 14:50   <- wrong, that's Atlanta's clock
Actually lands in LA at                         11:50   <- Pacific time
```

So every cross-country plane looked like it had about 3 hours more or less on the ground than it really did.

**How I noticed:** the numbers looked physically impossible. 19% of turns came out under 35 minutes, and some planes seemed to depart before they had even landed.

**The fix:** use BTS's reported local arrival time (`CRSArrTime`), and only use "departure + flight time" to work out **which day** the plane lands (red-eyes land the next morning). Time-zone gaps are always under 12 hours, so rounding to the nearest day always picks the right date. Now both times for a turn are on the **same airport's clock**, so subtracting them gives the real ground time.

**The result:**
| | Before fix | After fix |
|---|---|---|
| Typical ground time | 97 min | 77 min (realistic) |
| Turns under 35 min | 19% | 1.4% |
| Correlation: "late inbound plane" vs. next flight's delay | 0.36 | 0.52 |

**The lesson:** always sanity-check data against the real world. The code ran fine and gave no errors. The only clue was that the numbers didn't make physical sense.

### What "correlation 0.52" means
Correlation runs from 0 (no connection) to 1 (perfect connection). I built a simple measure: *how late the inbound plane arrived, minus its spare ground time.* That alone has a 0.52 correlation with the next flight's departure delay. In plain words, **a big part of Delta's delays are just earlier delays rippling forward.** That's the core idea the whole project is built on.

### Questions I should be ready for
**Q: Tell me about a bug you found.**
The time-zone story above: what looked wrong, how I traced it, how I fixed it, and the before/after numbers.

**Q: How do you know your data is right?**
I checked it against reality. Does the plane's next flight leave from where it landed (97% yes)? Are ground times physically possible (1.4% under 35 min after the fix)? Do the numbers match how airlines actually operate?

**Q: Why does the ripple effect matter?**
One late plane can make every one of its later flights late, plus the passengers and crews connecting to them. If you can predict that, you can step in early: swap in a spare plane or rebook passengers before the problem spreads.

---

## Piece 3: The Ripple Engine

### What it does
`ripple.py` follows every aircraft through its day and works out how one late flight spreads delay to that plane's later flights.

### The core idea (a bus doing a loop)
- **Ground time** = time between a plane landing and its next scheduled departure.
- **Slack** = ground time minus the 35-minute minimum turn. It's the spare time that can soak up a delay.
- If a plane lands late by **more** than its slack, the leftover delay is **pushed** onto the next flight:

```
pushed delay = max(0, how late the plane landed - slack)
```

- The leftover keeps passing down the chain until a big enough slack absorbs it, or the plane's day ends.

Example from the real data (DL1115 PHL to DTW, Jan 27, if it landed 90 min late):
```
DTW->MSY  slack  0  ->  90 min late
MSY->DTW  slack 23  ->  67 min late
DTW->CLT  slack 22  ->  45 min late
CLT->DTW  slack 15  ->  30 min late
4 later flights delayed, 232 minutes total, from one 90-minute delay
```

### How flights get linked
Two flights are "connected" only if all of these are true: same tail number, the second leaves from where the first landed, the ground time is between 0 and 12 hours (longer = overnight break), and the first flight actually flew (not cancelled). 88% of Delta's flights link to an inbound flight this way.

### Design decision: slack can't be negative
Some turns are scheduled tighter than 35 minutes. A negative slack would mean a plane that lands **on time** still makes the next flight late, which is impossible. If the airline scheduled a faster turn, that plane and airport can clearly do it, so the slack is set to 0 instead.

### The validation (my strongest number)
Airlines have to tell BTS **why** each delayed flight was late. One official cause is **"Late Aircraft"**, meaning the plane showed up late from its previous flight. That's exactly what my ripple model estimates, so I can check my model against the airline's own reports.

Delta, January 2024, 11,441 delayed flights compared:
| Check | Result |
|---|---|
| Correlation between my estimate and Delta's reported Late Aircraft minutes | **0.85** |
| Agree on whether a ripple happened at all | **86%** |
| Total minutes, mine vs. Delta's | 206,024 vs. 249,009 |

My model captures most of it but **undercounts by about 17%**. Likely reasons: the real minimum turn time varies by aircraft and airport (35 minutes is an average), and late planes sometimes lose extra time waiting for a gate or crew. Those are things to improve later.

### A known limitation (good to bring up yourself)
The worst "spreader" in January was an 18-hour delay that my model says rippled into 5 more flights. In real life, Delta probably **swapped in a different plane** long before that. My model assumes the plane keeps flying its original schedule. Deciding when to swap planes is exactly what the Recovery Optimizer (a later piece) is for.

### Commands
```
skyrecover prepare                                  # raw zips -> clean flights table
skyrecover ripple                                   # validation + top 10 delay spreaders
skyrecover ripple --flight-id 62498 --delay 90      # "what if this flight lands 90 min late?"
```

### Tests (12 passing)
Tests use tiny hand-made examples where I know the right answer, like a plane with 50 minutes on the ground (slack 15) landing 60 minutes late, so exactly 45 minutes should push to the next flight. They also lock in the time-zone fix from piece 2.

### Questions I should be ready for
**Q: How do you know your ripple model is right?**
I compared it against the delay causes Delta itself reports to the government. My estimate has a 0.85 correlation with Delta's reported "Late Aircraft" minutes, and we agree on whether a ripple happened 86% of the time.

**Q: What are its limitations?**
It undercounts by about 17%, because it uses one average minimum turn time. And it assumes a late plane keeps flying its schedule, when in reality airlines swap aircraft. Both are on my roadmap.

**Q: Why is this useful?**
It shows where one delay will cause the most damage downstream. If you know a plane's next four flights have almost no slack, that's the plane to swap out first.
