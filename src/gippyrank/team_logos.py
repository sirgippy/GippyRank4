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
107|Holy Cross|holycross
119|Towson|towson
13|Cal Poly|calpoly
147|Montana State|montanastate
149|Montana|montana
155|North Dakota|northdakota
160|New Hampshire|newhampshire
2000|Abilene Christian|abilenechristian
2011|Alabama State|alabamastate
2016|Alcorn State|alcornstate
2029|Arkansas-Pine Bluff|uapb
2046|Austin Peay|austinpeay
2065|Bethune-Cookman|bethunecookman
2083|Bucknell|bucknell
2097|Campbell|campbell
2110|Central Arkansas|centralarkansas
2115|Central Connecticut|ccsu
2127|Charleston Southern|charlestonsouthern
2142|Colgate|colgate
2169|Delaware State|delawarestate
2184|Duquesne|duquesne
2193|East Tennessee State|easttennesseestate
2197|Eastern Illinois|easternillinois
2198|Eastern Kentucky|easternkentucky
2210|Elon|elon
222|Villanova|villanova
2230|Fordham|fordham
2241|Gardner-Webb|gardnerwebb
2261|Hampton|hampton
227|Rhode Island|rhodeisland
2287|Illinois State|illinoisstate
231|Furman|furman
2320|Lamar|lamar
233|South Dakota|southdakota
2341|Long Island University|liu
236|Chattanooga|chattanooga
2377|McNeese|mcneese
2382|Mercer|mercer
2385|Mercyhurst|mercyhurst
2400|Mississippi Valley State|mvsu
2405|Monmouth|monmouth
2415|Morgan State|morganstate
2428|North Carolina Central|nccu
2447|Nicholls|nicholls
2448|North Carolina A&T|northcarolinaat
2450|Norfolk State|norfolkstate
2453|North Alabama|northalabama
2458|Northern Colorado|northerncolorado
2460|Northern Iowa|northerniowa
2464|Northern Arizona|northernarizona
2466|Northwestern State|northwesternstate
2502|Portland State|portlandstate
2504|Prairie View A&M|prairieviewam
2523|Robert Morris|robertmorris
2529|Sacred Heart|sacredheart
253|Southern Utah|southernutah
2535|Samford|samford
2545|SE Louisiana|southeasternlouisiana
257|Richmond|richmond
2571|South Dakota State|southdakotastate
2582|Southern|southern
2619|Stony Brook|stonybrook
2627|Tarleton State|tarletonstate
2630|UT Martin|utmartin
2634|Tennessee State|tennesseestate
2635|Tennessee Tech|tennesseetech
2640|Texas Southern|texassouthern
2643|The Citadel|citadel
2678|VMI|vmi
2681|Wagner|wagner
2692|Weber State|weberstate
2698|West Georgia|westgeorgia
2710|Western Illinois|westernillinois
2717|Western Carolina|westerncarolina
2729|William & Mary|williammary
2747|Wofford|wofford
2754|Youngstown State|youngstownstate
2755|Grambling|grambling
2771|Merrimack|merrimack
2803|Bryant|bryant
2815|Lindenwood|lindenwood
282|Indiana State|indianastate
284|Stonehill|stonehill
2916|Incarnate Word|incarnateword
292|UT Rio Grande Valley|utrgv
302|UC Davis|ucdavis
304|Idaho State|idahostate
3101|Utah Tech|utahtech
311|Maine|maine
322|Lafayette|lafayette
331|Eastern Washington|easternwashington
399|UAlbany|albany
47|Howard|howard
50|Florida A&M|floridaam
70|Idaho|idaho
79|Southern Illinois|southernillinois
93|Murray State|murraystate
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
