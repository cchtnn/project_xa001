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


def get_bi_monthly_written_reports() -> Dict[str, Any]:
    """Return bi-monthly written report requirements."""
    data = _load_bor_data()
    return (
        data.get("reporting_requirements", {})
        .get("bi_monthly_written_reports", {})
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

    Returns short explanatory text and HTML tables for date-heavy answers.
    """
    if today is None:
        today = date.today()

    q_lower = user_query.lower().strip()

    # 0. Document metadata / resolution / approval
    if "resolution" in q_lower or "approved" in q_lower or "planner" in q_lower:
        data = _load_bor_data()
        meta = data.get("document_metadata", {})
        title = meta.get("title", "Board of Regents Meeting Planner")
        approval = meta.get("approval_date", "")
        res_no = meta.get("resolution_number", "")
        year = meta.get("academic_year", "")
        return (
            f"{title} covers the {year} academic year, was approved on {approval}, "
            f"under resolution {res_no}."
        )

    # 1. "Next BOR meeting" style
    if "next" in q_lower and ("bor" in q_lower or "board of regents" in q_lower or "board meeting" in q_lower):
        m = get_next_bor_meeting(today)
        if not m:
            return "There are no upcoming Board of Regents meetings in the current planner."
        return (
            f"The next Board of Regents meeting is scheduled on "
            f"{m.get('meeting_date_formatted')}."
        )

    # 2. Direct "BOR meeting in <month> <year>" style
    if ("bor" in q_lower or "board of regents" in q_lower or "board meeting" in q_lower) and "meeting" in q_lower:
        month, year = _extract_month_and_year_from_query(user_query)
        if month and year:
            m = get_bor_meeting_by_month_year(month, year)
            if m:
                return (
                    f"The Board of Regents meeting for {month} {year} is on "
                    f"{m.get('meeting_date_formatted')}."
                )

    # Ask flags
    asks_meeting = (
        "meeting" in q_lower
        or "board meeting" in q_lower
        or "bor meeting" in q_lower
    )
    asks_report = "report" in q_lower or "reports" in q_lower

    # 3. BOTH meeting and report in the question → combined schedule + due table
    if asks_meeting and asks_report:
        data = _load_bor_data()
        meetings = data.get("bor_meetings", {}).get("specific_meetings", [])
        if not meetings:
            return "Board of Regents meeting and report dates are not defined in the current planner."

        rows_html = []
        for m in meetings:
            month = m.get("month")
            year = m.get("year")
            meeting_fmt = m.get("meeting_date_formatted")
            report_fmt = m.get("report_due_date_formatted")
            rows_html.append(
                f"<tr><td>{month} {year}</td><td>{meeting_fmt}</td><td>{report_fmt}</td></tr>"
            )

        pattern = data.get("bor_meetings", {}).get("schedule_pattern", {})
        freq = pattern.get("frequency", "Bi-Monthly")
        dow = pattern.get("day_of_week", "Friday")
        wom = pattern.get("week_of_month", "2nd Friday")

        table_html = (
            f"<p>Regular Board of Regents meetings are generally held {freq} on the {wom} "
            f"({dow}), and written reports are due on the Wednesday prior to each meeting.</p>"
            "<table border='1' cellspacing='0' cellpadding='4'>"
            "<thead><tr><th>Month / Year</th><th>BOR meeting date</th><th>Report due date</th></tr></thead>"
            "<tbody>"
            + "".join(rows_html) +
            "</tbody></table>"
        )
        return table_html

    # 4. Reporting due dates ONLY – table
    if asks_report and "due" in q_lower:
        data = _load_bor_data()
        bi = get_bi_monthly_written_reports()
        due_dates = bi.get("due_dates", [])
        if not due_dates:
            return "Reporting due dates are not available in the current BOR planner."

        meetings = data.get("bor_meetings", {}).get("specific_meetings", [])
        meeting_by_date = {m["meeting_date"]: m for m in meetings if "meeting_date" in m}

        rows_html = []
        for item in due_dates:
            meeting_date_iso = item.get("meeting_date")
            due_fmt = item.get("due_date_formatted")
            meeting = meeting_by_date.get(meeting_date_iso)
            meeting_fmt = meeting.get("meeting_date_formatted") if meeting else meeting_date_iso
            rows_html.append(
                f"<tr><td>{due_fmt}</td><td>for BOR on {meeting_fmt}</td></tr>"
            )

        table_html = (
            "<p>Reports are due on the Wednesday prior to each Board of Regents meeting.</p>"
            "<table border='1' cellspacing='0' cellpadding='4'>"
            "<thead><tr><th>Report due</th><th>Context</th></tr></thead>"
            "<tbody>"
            + "".join(rows_html) +
            "</tbody></table>"
        )
        return table_html

    # 5. Generic BOR meeting schedule ONLY – table
    if asks_meeting and ("when" in q_lower or "schedule" in q_lower or "date" in q_lower or "due" in q_lower):
        data = _load_bor_data()
        meetings = data.get("bor_meetings", {}).get("specific_meetings", [])
        if not meetings:
            return "Board of Regents meeting dates are not defined in the current planner."

        pattern = data.get("bor_meetings", {}).get("schedule_pattern", {})
        freq = pattern.get("frequency", "Bi-Monthly")
        dow = pattern.get("day_of_week", "Friday")
        wom = pattern.get("week_of_month", "2nd Friday")

        rows_html = []
        for m in meetings:
            month = m.get("month")
            year = m.get("year")
            meeting_fmt = m.get("meeting_date_formatted")
            rows_html.append(
                f"<tr><td>{month} {year}</td><td>{meeting_fmt}</td></tr>"
            )

        table_html = (
            f"<p>Regular Board of Regents meetings are generally held {freq} on the {wom} "
            f"({dow}).</p>"
            "<table border='1' cellspacing='0' cellpadding='4'>"
            "<thead><tr><th>Month / Year</th><th>BOR meeting date</th></tr></thead>"
            "<tbody>"
            + "".join(rows_html) +
            "</tbody></table>"
        )
        return table_html

    # 6. Bi-monthly written reports content / components
    if "bi-monthly" in q_lower or ("written reports" in q_lower) or ("bor reports" in q_lower and "include" in q_lower):
        bi = get_bi_monthly_written_reports()
        months = bi.get("reporting_months", [])
        components = bi.get("required_components", [])
        submission = bi.get("submission_deadline", "")

        comp_items = ", ".join(c.get("component") for c in components) if components else ""
        return (
            f"Bi-monthly written reports are submitted in {', '.join(months)}. "
            f"Reports are due {submission}. Each report must include: {comp_items}."
        )

    # 7. Association reporting schedule (Faculty & Staff)
    if "association" in q_lower or "faculty association" in q_lower or "staff association" in q_lower:
        assoc = get_association_reporting()
        entities = assoc.get("entities", [])
        fmt = assoc.get("format", {})
        schedule = assoc.get("reporting_schedule", [])
        deadline = assoc.get("submission_deadline", "")
        entities_str = ", ".join(entities) if entities else "the associations"
        requirement = fmt.get("requirement") or ""
        return (
            f"{entities_str} provide both written and oral reports bi-monthly in "
            f"{', '.join(schedule)}. {requirement} Reports are due {deadline}."
        )

    # 8. Committee meeting patterns (generic question)
    if "committee" in q_lower and ("when" in q_lower or "schedule" in q_lower or "pattern" in q_lower):
        data = _load_bor_data()
        pat = data.get("committee_meetings", {}).get("schedule_pattern", {})
        freq = pat.get("frequency", "Bi-Monthly")
        dow = pat.get("day_of_week", "Friday")
        wom = pat.get("week_of_month", "2nd Friday")
        note = pat.get("note", "Alternating months from regular BOR meetings")
        return (
            f"Standing committees meet {freq} on the {wom} ({dow}) in alternating months "
            f"from regular BOR meetings ({note}). Finance/Audit/Investment meets at 9:00 a.m., "
            f"Governance at 11:30 a.m., and Academic & Student Success at 2:00 p.m."
        )

    # 9. Committee meetings by committee name – table
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

            rows_html = [
                f"<tr><td>{d}</td><td>{time}</td></tr>"
                for d in formatted_dates
            ]
            table_html = (
                f"<p>{cname} meets at {time} on the scheduled dates below.</p>"
                "<table border='1' cellspacing='0' cellpadding='4'>"
                "<thead><tr><th>Date</th><th>Time</th></tr></thead>"
                "<tbody>"
                + "".join(rows_html) +
                "</tbody></table>"
            )
            return table_html

    # 10. Committee meetings filtered by month – table
    if "committee" in q_lower and any(
        m in q_lower
        for m in [
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
    ):
        month, _ = _extract_month_and_year_from_query(user_query)
        if month:
            rows = list_committee_meetings_for_month(month)
            if not rows:
                return f"No committee meetings are scheduled in {month} in the BOR planner."
            rows_html = []
            for cname, d in rows:
                rows_html.append(
                    f"<tr><td>{cname}</td><td>{d.get('formatted')}</td></tr>"
                )
            table_html = (
                f"<p>Committee meetings scheduled in {month} are listed below.</p>"
                "<table border='1' cellspacing='0' cellpadding='4'>"
                "<thead><tr><th>Committee</th><th>Date</th></tr></thead>"
                "<tbody>"
                + "".join(rows_html) +
                "</tbody></table>"
            )
            return table_html

    # 11. Key events (confirmed + pending, includes ACCT NLS '26) – specific or table
    if (
        "key events" in q_lower
        or "graduation" in q_lower
        or "acct" in q_lower
        or "aihec" in q_lower
        or "nls" in q_lower
        or "event" in q_lower
    ):
        events = list_key_events()
        confirmed = events.get("confirmed_events", [])
        pending = events.get("pending_events", [])

        # Specific ACCT NLS '26 handling
        if "acct nls" in q_lower:
            for e in pending:
                if "acct nls" in e.get("event_name", "").lower():
                    start = e.get("start_date")
                    status = e.get("status", "")
                    if start and "tba" in status.lower():
                        return (
                            f"{e.get('event_name')} starts on {start}. "
                            "The end date has not been announced."
                        )
                    elif start:
                        return (
                            f"{e.get('event_name')} starts on {start}. "
                            f"Status: {status}."
                        )
            for e in confirmed:
                if "acct nls" in e.get("event_name", "").lower():
                    start = e.get("start_date")
                    end = e.get("end_date")
                    if start and end:
                        return f"{e.get('event_name')} runs from {start} to {end}."
                    elif start:
                        return f"{e.get('event_name')} starts on {start}."
            return "ACCT NLS '26 is mentioned in the planner, but detailed dates are not fully defined."

        # Generic lookup by event name fragment (AIHEC FALL’25, ACCT GLI, graduations, etc.)
        for e in confirmed + pending:
            name = e.get("event_name", "")
            if not name:
                continue
            name_lower = name.lower()
            # Match on first word or major token
            if any(tok in q_lower for tok in name_lower.replace("’", "'").split()):
                start = e.get("start_date")
                end = e.get("end_date")
                dt = e.get("date")
                status = e.get("status", "")
                if dt:
                    return f"{name} is scheduled on {dt} ({status})."
                if start and end:
                    return f"{name} runs from {start} to {end} ({status})."
                if start and "tba" in status.lower():
                    return f"{name} starts on {start}. The end date has not been announced."
                if start:
                    return f"{name} starts on {start} ({status})."
                if "tba" in status.lower():
                    return f"{name} has dates to be announced ({status})."
                return f"{name} is listed in the planner with status: {status}."

        # Generic listing as table
        rows_html = []
        for e in confirmed:
            name = e.get("event_name")
            start = e.get("start_date")
            end = e.get("end_date")
            dt = e.get("date")
            status = e.get("status")
            if dt:
                date_text = dt
            elif start and end:
                date_text = f"{start} to {end}"
            elif start:
                date_text = f"Starting {start}"
            else:
                date_text = ""
            rows_html.append(
                f"<tr><td>{name}</td><td>{date_text}</td><td>{status}</td></tr>"
            )

        for e in pending:
            name = e.get("event_name")
            start = e.get("start_date")
            status = e.get("status")
            if start:
                date_text = f"Starting {start}"
            else:
                date_text = ""
            rows_html.append(
                f"<tr><td>{name}</td><td>{date_text}</td><td>{status}</td></tr>"
            )

        if not rows_html:
            return "Key events are defined in the BOR planner but dates are incomplete."

        table_html = (
            "<p>Key events in the Board of Regents planner are listed below.</p>"
            "<table border='1' cellspacing='0' cellpadding='4'>"
            "<thead><tr><th>Event</th><th>Date(s)</th><th>Status</th></tr></thead>"
            "<tbody>"
            + "".join(rows_html) +
            "</tbody></table>"
        )
        return table_html

    # 12. Fallback BOR answer: generic FAQ-style summary
    data = _load_bor_data()
    faq_list = data.get("faq", [])
    if faq_list:
        return (
            "The Board of Regents planner defines regular BOR meetings, "
            "alternating committee meetings, bi-monthly written reports, "
            "association reporting, and key events for the 2025-2026 academic year."
        )

    return (
        "The Board of Regents planner is loaded, but this question does not match "
        "any of the supported BOR meeting or reporting patterns yet."
    )
