"""Common date-format compatibility for the frozen calendar normalizers."""
from collections.abc import Callable
from datetime import date, datetime
from functools import wraps
import importlib
import inspect
from pathlib import Path
import re
from typing import Any

VERSION = 'strict-date-metadata-v1'
PROFILE = {
    'version': VERSION,
    'original_parser_first': True,
    'fallback_formats': ['YYYY/MM/DD (ddd) HH:MM', 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM'],
    'weekday_must_match_calendar': True,
    'invalid_fallback': 'return original None',
    'source_and_question_date_strings_mutated': False,
    'scope': 'Common parser compatibility; existing arm-specific temporal operations are unchanged',
    'patched_modules': ['continuous', 'continuous_v2'],
}
WEEKDAYS = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
SLASH_STAMP = re.compile(r'([0-9]{4})/([0-9]{2})/([0-9]{2}) \((Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) ([0-9]{2}):([0-9]{2})')
ISO_DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
ISO_STAMP = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}')


def fallback_date(text: str) -> date | None:
    """Accept complete, valid, timezone-free numeric dates; never guess locale."""
    match = SLASH_STAMP.fullmatch(text)
    try:
        if match:
            year, month, day, weekday, hour, minute = match.groups()
            parsed = datetime(int(year), int(month), int(day), int(hour), int(minute))
            return parsed.date() if WEEKDAYS[parsed.weekday()] == weekday else None
        if ISO_DATE.fullmatch(text):
            return date.fromisoformat(text)
        if ISO_STAMP.fullmatch(text):
            return datetime.fromisoformat(text).date()
    except ValueError:
        return None
    return None


def wrap_parser(original: Callable[[str], date | None]) -> Callable[[str], date | None]:
    """Preserve the original result/error before trying the additional formats."""
    @wraps(original)
    def parse_date(text: str) -> date | None:
        parsed = original(text)
        return parsed if parsed is not None else fallback_date(text)
    parse_date._date_metadata_adapter_version = VERSION
    return parse_date


def install_date_adapter() -> dict[str, Any]:
    """Patch both actual call-site globals once, only within this sibling clone."""
    source_root = Path(__file__).resolve().parent / 'source'
    pending = []
    for name in PROFILE['patched_modules']:
        module = importlib.import_module(name)
        expected = source_root / (name + '.py')
        if Path(module.__file__).resolve() != expected:
            raise ValueError('Use a fresh process; foreign calendar module: ' + name)
        original = module.parse_date
        if module.calendar_anchor.__globals__ is not vars(module) or module.anchor_units.__globals__ is not vars(module):
            raise ValueError('Calendar normalizer no longer uses its module parser: ' + name)
        marker = getattr(original, '_date_metadata_adapter_version', None)
        if marker is not None:
            if marker != VERSION or Path(inspect.getsourcefile(original)).resolve() != Path(__file__).resolve():
                raise ValueError('Conflicting date metadata adapter: ' + name)
            continue
        if Path(inspect.getsourcefile(original)).resolve() != expected:
            raise ValueError('Calendar parser is not the frozen source function: ' + name)
        pending.append((module, wrap_parser(original)))
    for module, parser in pending:
        module.parse_date = parser
    return PROFILE.copy()
