"""Which tools ride in which prompt.

Every registered tool's schema used to go into every model turn -- 18 of
them against a budget of 11 -- which cost prefill time on each turn and,
worse, gave the small model a long menu to mis-pick from (it chose
unread_only=true for "my last email" with the flag documented in front of
it). A tool now rides only when the utterance wears one of its cues; the
CORE set (time, weather, calendar, notes) always rides, and a tool with no
entry here rides always too, so a new module is never silently unreachable.
Keep cues generous: a missed cue means the tool cannot be called that turn.
"""
import re

CORE = frozenset({"get_time", "get_weather", "get_calendar", "notes"})

TOOL_CUES = {
    "get_location": (r"\bwhere am i\b", r"\bmy location\b", r"\bwhere are we\b", r"\blocation\b"),
    "add_event": (r"\b(?:add|put|schedule|create|book|set up)\b.*\b(?:calendar|event|meeting|appointment)\b",
                  r"\bcalendar\b"),
    "set_reminder": (r"\bremind", r"\breminder"),
    "set_timer": (r"\btimer", r"\bcountdown\b", r"\bminutes?\b", r"\bseconds?\b"),
    "set_alarm": (r"\balarm", r"\bwake me\b"),
    "manage_schedule": (r"\b(?:timers?|alarms?|reminders?)\b", r"\bcancel\b", r"\bschedule\b"),
    "get_mail": (r"\be-?mails?\b", r"\bmail\b", r"\binbox\b", r"\bgmail\b", r"\bmessages?\b"),
    "get_briefing": (r"\bbrief", r"\bnews\b", r"\bheadlines?\b", r"\bmorning\b", r"\bupdate me\b"),
    "spotify_play": (r"\bplay\b", r"\bspotify\b", r"\bmusic\b", r"\bsongs?\b", r"\bplaylist",
                     r"\balbum\b", r"\bartist\b", r"\bput on\b"),
    "spotify_liked": (r"\bliked\b", r"\blikes\b", r"\blibrary\b", r"\bfavou?rites?\b", r"\bspotify\b"),
    "spotify_control": (r"\b(?:pause|resume|unpause|skip|next|previous|back|volume|louder|quieter|"
                        r"softer|mute|shuffle|repeat|seek|jump|rewind)\b", r"\b(?:this|the) (?:song|track)\b",
                        r"\bspotify\b", r"\bplaying\b", r"\btransfer\b", r"\bmove it\b"),
    "spotify_now_playing": (r"\bplaying\b", r"\bthis song\b", r"\bwhat song\b", r"\bwho is this\b",
                            r"\bspotify\b"),
    "spotify_queue": (r"\bqueue\b", r"\bup next\b", r"\bplay .{1,40} next\b", r"\bafter this\b"),
    "spotify_radio": (r"\bradio\b", r"\blike this\b", r"\bsimilar\b", r"\bsomething like\b",
                      r"\bmore (?:of|like|by)\b"),
    # tools that land with their own modules
    "canvas_due": (r"\bcanvas\b", r"\bdue\b", r"\bassignments?\b", r"\bhomework\b", r"\bcourse",
                   r"\bclass(?:es)?\b", r"\b(?:midterm|exam|quiz|lab|project)s?\b", r"\bthis week\b"),
    "canvas_grades": (r"\bgrades?\b", r"\bscores?\b", r"\bcanvas\b", r"\bcourse", r"\bclass(?:es)?\b",
                      r"\bhow am i doing\b"),
    "canvas_announcements": (r"\bannouncements?\b", r"\bcanvas\b", r"\bprofessor", r"\binstructor",
                             r"\bcourse", r"\bclass(?:es)?\b"),
    "ask_docs": (r"\b(?:documents?|docs?|syllabus|syllabi|notes?|pdfs?|files?|papers?|lectures?|slides|"
                 r"readings?|textbook|handout)\b", r"\baccording to\b", r"\bin my\b.*\b(?:notes|files)\b"),
    "docs_reindex": (r"\b(?:re)?index\b", r"\bdocuments?\b", r"\brescan\b"),
    "screen_qa": (r"\bscreen\b", r"\bwindow\b", r"\blooking at\b", r"\bthis error\b", r"\bwhat(?:'s| is) this\b",
                  r"\bon my (?:screen|display|monitor)\b", r"\bread (?:me )?this\b", r"\bsee this\b"),
    "system_health": (r"\b(?:system|spark|machine|computer|gpu|memory|ram|temperature|temp|health|"
                      r"diagnostics?|load|disk)\b", r"\bhow(?:'s| is) (?:the|this) (?:spark|machine|system|box)\b"),
}


def compile_cues() -> dict:
    return {name: tuple(re.compile(c, re.I) for c in cues) for name, cues in TOOL_CUES.items()}
