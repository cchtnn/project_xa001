import json
import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

# Path to BOR JSON file
BOR_JSON_PATH = os.path.join("data", "bor_json.json")

_bor_cache: Optional[Dict[str, Any]] = None


def _load_bor_data() -> Dict[str, Any]:
    """Load and cache BOR planner JSON data."""
    global _bor_cache
    if _bor_cache is None:
        if not os.path.exists(BOR_JSON_PATH):
            raise FileNotFoundError(f"BOR JSON not found at {BOR_JSON_PATH}")
        with open(BOR_JSON_PATH, "r", encoding="utf-8") as f:
            _bor_cache = json.load(f)
    return _bor_cache


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse ISO date string (YYYY-MM-DD) to date object."""
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_next_bor_meeting(today: date) -> Optional[Dict[str, Any]]:
    """Return the next BOR meeting strictly after 'today'."""
    data = _load_bor_data()
    meetings = data.get("bor_meetings", {}).get("specific_meetings", [])
    upcoming: List[Tuple[date, Dict[str, Any]]] = []

    for m in meetings:
        d = _parse_iso_date(m.get("meeting_date"))
        if d and d > today:
            upcoming.append((d, m))

    if not upcoming:
        return None

    upcoming.sort(key=lambda x: x[0])
    return upcoming[0][1]


def get_bor_meeting_by_month_year(month_name: str, year: int) -> Optional[Dict[str, Any]]:
    """Return BOR meeting for given month name and year, if present."""
    data = _load_bor_data()
    meetings = data.get("bor_meetings", {}).get("specific_meetings", [])
    month_normalized = month_name.strip().lower()

    for m in meetings:
        if (
            str(m.get("year")) == str(year)
            and str(m.get("month", "")).strip().lower() == month_normalized
        ):
            return m
    return None


def get_report_due_for_meeting(meeting_date_iso: str) -> Optional[Dict[str, Any]]:
    """Return reporting due-date record for a given ISO meeting date."""
    data = _load_bor_data()
    due_list = (
        data.get("reporting_requirements", {})
        .get("bi_monthly_written_reports", {})
        .get("due_dates", [])
    )
    for item in due_list:
        if item.get("meeting_date") == meeting_date_iso:
            return item
    return None


def get_committee_schedule(committee_name: str) -> Optional[Dict[str, Any]]:
    """Return committee record (name, time, specific dates)."""
    data = _load_bor_data()
    committees = (
        data.get("committee_meetings", {})
        .get("committees", [])
    )
    name_norm = committee_name.strip().lower()
    for c in committees:
        if c.get("committee_name", "").strip().lower() == name_norm:
            return c
    return None


def list_committee_meetings_for_month(month_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (committee_name, date_record) for a given month across all committees."""
    data = _load_bor_data()
    committees = (
        data.get("committee_meetings", {})
        .get("committees", [])
    )
    month_norm = month_name.strip().lower()
    results: List[Tuple[str, Dict[str, Any]]] = []

    for c in committees:
        cname = c.get("committee_name", "")
        for d in c.get("specific_dates", []):
            iso = d.get("date")
            if not iso:
                continue
            dt = _parse_iso_date(iso)
            if dt and dt.strftime("%B").lower() == month_norm:
                results.append((cname, d))
    return results


def get_association_reporting() -> Dict[str, Any]:
    """Return association reporting rules."""
    data = _load_bor_data()
    return (
        data.get("reporting_requirements", {})
        .get("association_reporting", {})
    )


def list_key_events() -> Dict[str, List[Dict[str, Any]]]:
    """Return all key events grouped by confirmed/pending."""
    data = _load_bor_data()
    return data.get("key_events", {})


def _extract_month_and_year_from_query(user_query: str) -> Tuple[Optional[str], Optional[int]]:
    """Very light month/year extraction from user query."""
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    q = user_query.lower()
    month_found: Optional[str] = None
    for m in months:
        if m in q:
            month_found = m.capitalize()
            break

    year_found: Optional[int] = None
    for token in q.split():
        token_clean = "".join(ch for ch in token if ch.isdigit())
        if len(token_clean) == 4:
            try:
                year_found = int(token_clean)
                break
            except ValueError:
                continue

    return month_found, year_found


def answer_bor_query(user_query: str, today: Optional[date] = None) -> str:
    """
    High-level BOR query handler.

    It stays deterministic and only uses the BOR JSON, so it is safe for
    production without affecting other domains.
    """
    if today is None:
        today = date.today()

    q_lower = user_query.lower().strip()

    # 1. "Next BOR meeting" style
    if "next" in q_lower and ("bor" in q_lower or "board of regents" in q_lower):
        m = get_next_bor_meeting(today)
        if not m:
            return "There are no upcoming Board of Regents meetings in the current planner."
        return (
            f"The next Board of Regents meeting is scheduled on "
            f"{m.get('meeting_date_formatted')}."
        )

    # 2. Direct "BOR meeting in <month> <year>" style
    if ("bor" in q_lower or "board of regents" in q_lower) and "meeting" in q_lower:
        month, year = _extract_month_and_year_from_query(user_query)
        if month and year:
            m = get_bor_meeting_by_month_year(month, year)
            if m:
                return (
                    f"The Board of Regents meeting for {month} {year} is on "
                    f"{m.get('meeting_date_formatted')}."
                )

    # 3. Reporting due dates
    if "report" in q_lower and ("due" in q_lower or "deadline" in q_lower):
        data = _load_bor_data()
        bi = (
            data.get("reporting_requirements", {})
            .get("bi_monthly_written_reports", {})
        )
        due_dates = bi.get("due_dates", [])
        if not due_dates:
            return "Reporting due dates are not available in the current BOR planner."

        month, year = _extract_month_and_year_from_query(user_query)
        if month:
            for item in due_dates:
                mdate = _parse_iso_date(item.get("meeting_date"))
                if not mdate:
                    continue
                if mdate.strftime("%B").lower() == month.lower():
                    return (
                        f"The report due date before the {month} Board of Regents meeting "
                        f"is {item.get('due_date_formatted')}."
                    )

        # Fallback generic description
        return (
            "Reports are due on the Wednesday prior to each Board of Regents meeting, "
            "with specific due dates defined in the planner for each meeting."
        )

    # 4. Committee meetings by committee name
    if "committee" in q_lower:
        if "finance" in q_lower:
            cname = "Finance/Audit/Investment Committee"
        elif "governance" in q_lower:
            cname = "Governance Committee"
        elif "student success" in q_lower or "academic" in q_lower:
            cname = "Academic & Student Success Committee"
        else:
            cname = ""

        if cname:
            c = get_committee_schedule(cname)
            if not c:
                return f"No schedule found for {cname} in the BOR planner."
            time = c.get("time_formatted") or c.get("time")
            dates = c.get("specific_dates", [])
            formatted_dates = [d.get("formatted") for d in dates if d.get("formatted")]
            if not formatted_dates:
                return f"{cname} meets as scheduled, but specific dates are not listed."
            return (
                f"{cname} meets at {time}, with the following scheduled dates: "
                + "; ".join(formatted_dates)
                + "."
            )

    # 5. Committee meetings filtered by month
    if "committee" in q_lower and any(m in q_lower for m in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
        month, _ = _extract_month_and_year_from_query(user_query)
        if month:
            rows = list_committee_meetings_for_month(month)
            if not rows:
                return f"No committee meetings are scheduled in {month} in the BOR planner."
            parts = []
            for cname, d in rows:
                parts.append(f"{cname} on {d.get('formatted')}")
            return (
                f"In {month}, the following committee meetings are scheduled: "
                + "; ".join(parts)
                + "."
            )

    # 6. Association reporting rules
    if "faculty association" in q_lower or "staff association" in q_lower or "association reporting" in q_lower:
        assoc = get_association_reporting()
        entities = assoc.get("entities", [])
        fmt = assoc.get("format", {})
        schedule = assoc.get("reporting_schedule", [])
        deadline = assoc.get("submission_deadline", "")
        entities_str = ", ".join(entities) if entities else "the associations"
        requirement = fmt.get("requirement") or ""
        return (
            f"{entities_str} report during the Board of Regents meetings in "
            f"{', '.join(schedule)}. {requirement} Reports are due "
            f"{deadline}."
        )

    # 7. Key events
    if "key events" in q_lower or "graduation" in q_lower or "acct" in q_lower or "aihec" in q_lower:
        events = list_key_events()
        confirmed = events.get("confirmed_events", [])
        if not confirmed:
            return "No confirmed key events are listed in the BOR planner."
        pieces = []
        for e in confirmed:
            name = e.get("event_name")
            start = e.get("start_date")
            end = e.get("end_date")
            dt = e.get("date")
            status = e.get("status")
            if dt:
                pieces.append(f"{name} on {dt} ({status})")
            elif start and end:
                pieces.append(f"{name} from {start} to {end} ({status})")
            elif start:
                pieces.append(f"{name} starting {start} ({status})")
        if not pieces:
            return "Key events are defined in the BOR planner but dates are incomplete."
        return "Key events in the Board of Regents planner include: " + "; ".join(pieces) + "."

    # 8. Fallback BOR answer: generic FAQ-style summary
    data = _load_bor_data()
    faq_list = data.get("faq", [])
    if faq_list:
        # Very lightweight: return a brief generic description instead of copying FAQ text
        return (
            "The Board of Regents planner defines bi-monthly BOR meetings, "
            "alternating committee meetings, reporting requirements, and key events "
            "for the 2025-2026 academic year."
        )

    return (
        "The Board of Regents planner is loaded, but this question does not match "
        "any of the supported BOR meeting or reporting patterns yet."
    )
