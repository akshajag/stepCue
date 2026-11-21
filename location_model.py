"""Pure view-model for the caregiver location view (no Streamlit, fully tested).

Joins three things: the GPS trail (location.json), the wearer's places
(places.json) and the freeze episodes (state.json), so the map can show
where freezes happen. Describes patterns only.
"""
import time

from location_io import haversine_m, place_at, trail_distance_m

LIVE_SEC = 30          # a fix newer than this counts as live
STALE_SEC = 300        # older than this: "last seen"
JOIN_SEC = 120         # a freeze is placed at the trail point within this many seconds
MIN_VISIT_FIXES = 2    # ignore a place if only one GPS fix fell inside it
GAP_SEC = 300          # trail gaps longer than this do not count as time anywhere


def midnight(now):
    lt = time.localtime(now)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def episode_epoch(stamp, now):
    """'HH:MM:SS' (local, today) -> epoch seconds; yesterday if it would be in the future."""
    try:
        h, m, s = (int(x) for x in str(stamp).split(":"))
    except ValueError:
        return None
    t = midnight(now) + h * 3600 + m * 60 + s
    return t - 86400 if t > now + 60 else t


def position_at(trail, t, max_gap=JOIN_SEC):
    """Trail point closest in time to t, or None if none within max_gap seconds."""
    if not trail or t is None:
        return None
    best = min(trail, key=lambda p: abs(p[2] - t))
    return (best[0], best[1]) if abs(best[2] - t) <= max_gap else None


def visits(trail, places):
    """Runs of fixes inside the same place -> [{place, arrive, leave}]."""
    out, cur, start, last, n = [], None, None, None, 0
    for lat, lon, t in trail + [[None, None, None]]:
        pl = place_at(lat, lon, places) if lat is not None else None
        p = pl["name"] if pl else None
        if p != cur or (last is not None and t is not None and t - last > GAP_SEC):
            if cur is not None and n >= MIN_VISIT_FIXES:
                out.append({"place": cur, "arrive": start, "leave": last})
            cur, start, n = p, t, 0
        n += 1
        last = t
    return out


def freshness(fix, now):
    if not fix:
        return {"key": "none", "text": "No location yet"}
    age = max(0.0, now - fix["t"])
    acc = f" ±{fix['acc']:.0f} m" if fix.get("acc") is not None else ""
    src = "simulated GPS" if fix.get("source") == "simulated" else "phone GPS"
    if age <= LIVE_SEC:
        return {"key": "live", "text": f"Live, {src}{acc}"}
    mins = age / 60
    when = f"{age:.0f} s ago" if age < 60 else (f"{mins:.0f} min ago" if mins < 90 else "over an hour ago")
    return {"key": "stale" if age > STALE_SEC else "recent", "text": f"Last seen {when}"}


def clock(t):
    return time.strftime("%H:%M", time.localtime(t))


def location_view(loc, places, episodes, now=None):
    now = time.time() if now is None else now
    trail = [p for p in (loc or {}).get("trail", []) if isinstance(p, (list, tuple)) and len(p) == 3]
    fix = (loc or {}).get("fix")
    day0 = midnight(now)
    today = [p for p in trail if p[2] >= day0]

    here = place_at(fix["lat"], fix["lon"], places) if fix else None
    home = next((p for p in places if p.get("kind") == "home"), None)

    # time away from home today
    away = 0.0
    for a, b in zip(today, today[1:]):
        dt = b[2] - a[2]
        if 0 < dt <= GAP_SEC and not (home and haversine_m(a[0], a[1], home["lat"], home["lon"]) <= home["radius_m"]):
            away += dt

    # freezes on the map
    spots, by_place = [], {}
    for e in episodes or []:
        t = episode_epoch(e.get("timestamp"), now)
        pos = position_at(trail, t)
        if pos is None:
            continue
        pl = place_at(pos[0], pos[1], places)
        name = pl["name"] if pl else "On the way"
        spots.append({"id": e.get("id"), "lat": pos[0], "lon": pos[1], "place": name, "t": t,
                      "time": clock(t), "duration_sec": float(e.get("duration_sec") or 0.0),
                      "during_turn": bool(e.get("during_turn")),
                      "ongoing": e.get("recovery_sec") is None})
        by_place[name] = by_place.get(name, 0) + 1

    vis = visits(today, places)
    events = []
    for v in vis:
        events.append({"t": v["arrive"], "kind": "arrive", "title": "Arrived", "place": v["place"]})
        if v["leave"] is not None and now - v["leave"] > LIVE_SEC * 2:
            events.append({"t": v["leave"], "kind": "leave", "title": "Left", "place": v["place"]})
    for s in spots:
        events.append({"t": s["t"], "kind": "freeze", "place": s["place"],
                       "title": f"Freeze {s['duration_sec']:.1f} s" + (" in a turn" if s["during_turn"] else "")})
    events.sort(key=lambda e: e["t"])
    for e in events:
        e["time"] = clock(e["t"])

    total = sum(by_place.values())
    ranked = sorted(by_place.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "fix": fix,
        "freshness": freshness(fix, now),
        "here": here["name"] if here else ("Out and about" if fix else None),
        "distance_km": trail_distance_m(today) / 1000.0,
        "away_min": away / 60.0,
        "places_visited": list(dict.fromkeys(v["place"] for v in vis)),
        "spots": spots,
        "freezes_by_place": [{"place": k, "count": n, "share": n / total} for k, n in ranked],
        "timeline": events[-6:],
        "trail": today,
    }
