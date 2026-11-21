# StepCue caregiver view: design spec

## Who and when
A family caregiver glancing at a phone or laptop for 2-3 seconds between tasks.
In the demo, judges watching a projector from across a room.

**The one question:** Is my person OK right now, and do I need to do anything?

Secondary questions:
1. What happened recently? (episodes, how long, how quickly walking resumed)
2. Is it getting better or worse? (recovery vs. average, time spent in each state)
3. What did StepCue do about it? (Nemotron's cue changes and reasoning)

The caregiver's action: check in on the person. In demo mode, presenters also trigger scenarios.

## Layout
```
+--------------------------------------------------------------+
| StepCue                                     Live, updated 0.3s|
+--------------------------------------------------------------+
| [icon] ALL STEADY / FREEZE IN PROGRESS / PLEASE CHECK IN      |
| one plain sentence                      Walking . cue 92 bpm  |   <- status banner (hero)
+--------------------------------------------------------------+
| Episodes today | Typical recovery | Cue setting               |   <- 3 key numbers with context
+-------------------------------------+------------------------+
| Freeze Index over the last minute   | Nemotron care agent    |
| [live chart, freeze spans shaded]   | brief (large)          |
| Last 10 minutes: walking/standing/  | decision feed:         |
| frozen bar                          | change + why + latency |
+-------------------------------------+------------------------+
| Episode log (newest first)                                   |
+--------------------------------------------------------------+
| Scope footer                                                  |
```
Alternatives considered: a left patient card with content on the right (wastes width on a phone), and a four-card grid (the old design: the answer was split across two equal cards).

## Status model (priority order, first match wins)
| Key | Trigger | Headline | Look |
| --- | --- | --- | --- |
| offline | state.json older than 3 s | No signal from the ankle sensor | slate, dashed border, values greyed |
| alert | ongoing episode over 4 s, or `alert_level == "alert"` | Please check in | solid red banner, white text, pulsing |
| freeze | ongoing episode (`recovery_sec` is null) | Freeze in progress, cue playing | amber tint, pulsing dot |
| watch | `alert_level == "watch"` | Keep an eye out | amber tint, no motion |
| steady | everything else | All steady | teal tint, calm |

## Tokens
| Name | Hex | Use |
| --- | --- | --- |
| ink | #16202B | text |
| muted | #556274 | secondary text (5.7:1 on bg) |
| bg | #F4F6F8 | page |
| surface | #FFFFFF | panels |
| ok | #0F766E | steady, brand accent |
| watch | #A35A00 | freeze / watch |
| alert | #9B1C1C | check in (filled) |
| offline | #475569 | no signal |

Alert is a filled banner and watch is a tint, so the two differ in fill and lightness as well as hue; every status also has its own icon and word.

Type: Atkinson Hyperlegible (designed by the Braille Institute for low-vision readers), with system sans as fallback. Body 16 px, banner headline 40 px, key numbers 32 px. Sentence-case labels, no all-caps.

## States
Empty first run, live steady, freeze, alert, watch, offline, Nemotron offline (rules fallback labelled honestly), 50 episodes (table scrolls).
