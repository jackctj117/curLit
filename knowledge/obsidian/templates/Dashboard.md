---
type: dashboard
tags: [dashboard, dataview]
source: personal
last_reviewed: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

_Personal MOC. These Dataview blocks activate with the Dataview plugin and run
over the frontmatter conventions this vault shares with the generated mirror
(`type`, `tags`, `region`, `tickers`, `commodities`, `confidence`,
`last_reviewed`). Without Dataview they render as inert code blocks._

## My themes by confidence

```dataview
TABLE region, confidence, last_reviewed
FROM #theme
SORT confidence DESC
```

## Companies by region

```dataview
TABLE region, tickers, confidence
FROM #company
SORT region ASC
```

## Recent events

```dataview
TABLE date, region, confidence
FROM #event
SORT date DESC
LIMIT 20
```

## Stale notes (review me)

```dataview
TABLE type, last_reviewed
WHERE last_reviewed
SORT last_reviewed ASC
LIMIT 20
```
