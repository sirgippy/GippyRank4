"""Canonical RedditCFB logo handles for the published team universe.

The site keeps the stable CFBD team ID as its model key.  ``handle`` is a
separate presentation key: it is the exact path component expected by the
RedditCFB CDN.  The table below is deliberately checked in instead of being
guessed from a display name at render time.  That makes renamed teams and
ambiguous abbreviations fail closed during export.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

TEAM_LOGO_URL_TEMPLATE = "https://cdn.redditcfb.com/60x40/cfb/{handle}.png"
_HANDLE_PATTERN = re.compile(r"^[a-z0-9]+$")

# Verified with browser-style GET requests against the RedditCFB CDN on
# 2026-09-08.  The non-display-name handles are the known namespace aliases:
# South Florida/usf, Western Kentucky/wku, East Carolina/ecu, App State/
# appalachianstate, Florida Atlantic/fau, UConn/connecticut, Florida
# International/fiu, UL Monroe/ulm, Sam Houston/samhoustonstate, and
# Massachusetts/umass.
_TEAM_LOGO_ROWS = """
194|Ohio State|ohiostate
2483|Oregon|oregon
61|Georgia|georgia
87|Notre Dame|notredame
251|Texas|texas
333|Alabama|alabama
2390|Miami|miami
84|Indiana|indiana
245|Texas A&M|texasam
201|Oklahoma|oklahoma
2641|Texas Tech|texastech
30|USC|usc
130|Michigan|michigan
145|Ole Miss|olemiss
252|BYU|byu
2294|Iowa|iowa
254|Utah|utah
228|Clemson|clemson
2633|Tennessee|tennessee
264|Washington|washington
99|LSU|lsu
57|Florida|florida
142|Missouri|missouri
52|Florida State|floridastate
2579|South Carolina|southcarolina
152|NC State|ncstate
248|Houston|houston
12|Arizona|arizona
238|Vanderbilt|vanderbilt
2628|TCU|tcu
213|Penn State|pennstate
2306|Kansas State|kansasstate
135|Minnesota|minnesota
2567|SMU|smu
77|Northwestern|northwestern
97|Louisville|louisville
68|Boise State|boisestate
221|Pittsburgh|pittsburgh
158|Nebraska|nebraska
356|Illinois|illinois
21|San Diego State|sandiegostate
2|Auburn|auburn
59|Georgia Tech|georgiatech
265|Washington State|washingtonstate
2449|North Dakota State|northdakotastate
9|Arizona State|arizonastate
2711|Western Michigan|westernmichigan
275|Wisconsin|wisconsin
164|Rutgers|rutgers
239|Baylor|baylor
256|James Madison|jamesmadison
344|Mississippi State|mississippistate
258|Virginia|virginia
2655|Tulane|tulane
150|Duke|duke
25|California|california
58|South Florida|usf
66|Iowa State|iowastate
98|Western Kentucky|wku
120|Maryland|maryland
2649|Toledo|toledo
96|Kentucky|kentucky
154|Wake Forest|wakeforest
2636|UTSA|utsa
259|Virginia Tech|virginiatech
2132|Cincinnati|cincinnati
26|UCLA|ucla
2305|Kansas|kansas
278|Fresno State|fresnostate
153|North Carolina|northcarolina
277|West Virginia|westvirginia
127|Michigan State|michiganstate
326|Texas State|texasstate
2116|UCF|ucf
276|Marshall|marshall
309|Louisiana|louisiana
151|East Carolina|ecu
24|Stanford|stanford
2572|Southern Miss|southernmiss
62|Hawai'i|hawaii
195|Ohio|ohio
8|Arkansas|arkansas
295|Old Dominion|olddominion
235|Memphis|memphis
249|North Texas|northtexas
2348|Louisiana Tech|louisianatech
2509|Purdue|purdue
2439|UNLV|unlv
328|Utah State|utahstate
55|Jacksonville State|jacksonvillestate
183|Syracuse|syracuse
48|Delaware|delaware
2653|Troy|troy
38|Colorado|colorado
6|South Alabama|southalabama
349|Army|army
2032|Arkansas State|arkansasstate
193|Miami (OH)|miamioh
167|New Mexico|newmexico
2426|Navy|navy
2117|Central Michigan|centralmichigan
2026|App State|appalachianstate
103|Boston College|bostoncollege
218|Temple|temple
2226|Florida Atlantic|fau
204|Oregon State|oregonstate
2335|Liberty|liberty
290|Georgia Southern|georgiasouthern
2623|Missouri State|missouristate
5|UAB|uab
202|Tulsa|tulsa
41|UConn|connecticut
2199|Eastern Michigan|easternmichigan
2005|Air Force|airforce
2229|Florida International|fiu
324|Coastal Carolina|coastalcarolina
338|Kennesaw State|kennesawstate
2006|Akron|akron
197|Oklahoma State|oklahomastate
242|Rice|rice
2751|Wyoming|wyoming
36|Colorado State|coloradostate
16|Sacramento State|sacramentostate
2459|Northern Illinois|northernillinois
2440|Nevada|nevada
2309|Kent State|kentstate
189|Bowling Green|bowlinggreen
2084|Buffalo|buffalo
23|San José State|sanjosestate
2393|Middle Tennessee|middletennessee
2247|Georgia State|georgiastate
2050|Ball State|ballstate
2433|UL Monroe|ulm
166|New Mexico State|newmexicostate
2534|Sam Houston|samhoustonstate
2429|Charlotte|charlotte
2638|UTEP|utep
113|Massachusetts|umass
"""

TEAM_LOGO_HANDLES: dict[str, tuple[str, str]] = {
    team_id: (team_name, handle)
    for team_id, team_name, handle in (
        row.split("|") for row in _TEAM_LOGO_ROWS.strip().splitlines()
    )
}


def team_logo_handle(team_id: str, team_name: str) -> str | None:
    """Return the verified handle for an exact team identity, if available."""
    record = TEAM_LOGO_HANDLES.get(str(team_id))
    if record is None or record[0] != team_name:
        return None
    return record[1]


def logo_url(handle: str | None, template: str = TEAM_LOGO_URL_TEMPLATE) -> str | None:
    """Render one CDN URL from a validated handle and configurable template."""
    if handle is None:
        return None
    if not isinstance(handle, str) or not _HANDLE_PATTERN.fullmatch(handle):
        raise ValueError(f"Invalid RedditCFB logo handle: {handle!r}")
    if not isinstance(template, str) or template.count("{handle}") != 1:
        raise ValueError("Logo URL template must contain exactly one {handle} placeholder")
    rendered = template.replace("{handle}", handle)
    if "{" in rendered or "}" in rendered:
        raise ValueError("Logo URL template contains an unsupported placeholder")
    return rendered


def missing_team_logos(teams: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return active team identities without an exact verified logo mapping."""
    return [
        (str(team_id), team_name)
        for team_id, team_name in teams
        if team_logo_handle(str(team_id), team_name) is None
    ]
