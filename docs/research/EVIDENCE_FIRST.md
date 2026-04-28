# Evidence-First Operating Principle

> *Operator guidance, 2026-04-28:*
> *"Before risking money: pull the data first. Dubai Land Department*
> *publishes transaction volumes monthly. CBUAE publishes monetary*
> *statistics. Hong Kong and mainland family-office formation data is*
> *partially public. If the thesis is real, it should already be visible*
> *in at least one of those. If it's not visible, you're trading a vibe."*

This is the standing operating principle for every agent in the
research pipeline: **a tradable thesis must be empirically locatable in
public data.** A plausible-sounding macro story is not enough. Vibes
are not edge.

## What this means in practice

For every candidate hypothesis the pipeline considers, the agent (Idea,
Bull, Bear) must answer four questions BEFORE assigning it weight:

1. **Where would this thesis show up if it's real?**
   Name the specific public dataset(s) where the underlying mechanism
   would leave a trace. Examples:
   - FX flow theses → balance-of-payments data, central-bank reserves
   - Real-estate theses → land-registry transaction volumes + prices
   - Credit theses → bank-lending surveys, sovereign CDS spreads
   - Migration / capital-flight theses → tax filings, family-office
     formations, immigration statistics
   - Sentiment theses → published surveys (Sentix, ZEW), positioning
     reports (CFTC COT, EPFR flows)

2. **Have we actually pulled that data?**
   Not "could we?" — *have we?* If the data hasn't been queried, the
   thesis is unsubstantiated regardless of how compelling the
   narrative reads.

3. **What does the data say?**
   Quote specific values, dates, magnitudes. "Dubai property
   transaction volumes fell 12% YoY in March 2026 with the largest
   declines in luxury segments" is evidence. "It feels like a
   downturn is coming" is not.

4. **If the data does NOT show the thesis, why might it not yet?**
   Sometimes leading indicators arrive before lagging ones. If the
   thesis depends on a still-unobserved tipping point, that's a real
   concern — but state it explicitly and identify the specific
   threshold that, when crossed, would substantiate the thesis.

## How agents should use this

**Idea Agent**: every hypothesis brief must include a "data sources"
section listing the public datasets where evidence would be located,
plus the specific signal expected (direction, magnitude, time horizon).

**Bull Reviewer**: when supporting a candidate, cite the data — quote
the specific values from the report's data sources. If the candidate's
hypothesis brief listed datasets but the strategy code didn't actually
consume them, that's a gap to flag, not a passable rule.

**Bear Reviewer**: this principle is your sharpest tool. For every
claim Bull makes, ask "where would this be visible publicly?" and
"have they checked there?" If a thesis depends on a phenomenon that
should leave a trace in published data and the candidate hasn't shown
the trace, REJECT with this principle as the reason.

**Question Resolver**: smart-questions of the form "which dataset
would substantiate / contradict this claim?" are the highest-value
ones an agent can ask. Route to `code_tool` if the dataset is in our
DB; route to `human` if it's a dataset we'd need to seed first.

## When the data isn't accessible

A thesis whose required dataset is not yet ingested into our pipeline
is not necessarily wrong — it's *not yet researchable*. The correct
response in that case is to:

1. File a data-seeding ticket (separate from the strategy ticket).
2. ESCALATE the candidate — operator decides whether the thesis is
   important enough to seed the new data source first.
3. Do NOT promote the strategy on a "we'd see it if we looked"
   argument. If you'd see it, look.

## Changelog

- 2026-04-28 — initial version, codified from operator guidance after
  the UAE → China capital-flight hypothesis discussion (CL-hhpf).
