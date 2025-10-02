from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_best_effort(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    if "```" in s:
        parts = s.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if "\n" in block:
                first, rest = block.split("\n", 1)
                candidate = rest if first.strip().lower().startswith("json") else block
            else:
                candidate = block
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return {}


def extract_sql_from_text(s: str) -> str:
    """Best-effort extraction of SQL from arbitrary text.
    - Prefer fenced ```sql blocks; else first SELECT/WITH till fence/end
    """
    if not s:
        return ""
    txt = s.strip()
    if "```" in txt:
        parts = txt.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if "\n" in block:
                first, rest = block.split("\n", 1)
                body = rest if first.strip().lower().startswith("sql") else block
            else:
                body = block
            body = body.strip()
            low = body.lower()
            if low.startswith("select") or low.startswith("with"):
                return body
    import re
    # Prefer a real SELECT first
    m_sel = re.search(r"(?is)\bselect\b[\s\S]*$", txt)
    # Detect likely SQL CTE only if WITH is followed by identifier and AS (
    m_with = re.search(r"(?is)\bwith\b\s+[a-z_][\w]*\s+as\s*\(", txt)
    m_with_simple = re.search(r"(?is)\bwith\b\s*\(", txt)
    start_idx = None
    if m_sel:
        start_idx = m_sel.start()
    if m_with and (start_idx is None or m_with.start() < start_idx):
        start_idx = m_with.start()
    elif (not m_with) and m_with_simple and (start_idx is None or m_with_simple.start() < start_idx):
        # Fallback WITH( ... ) form
        start_idx = m_with_simple.start()
    if start_idx is not None:
        candidate = txt[start_idx:].strip()
        # Stop at fenced code or JSON
        cutoff = candidate.find("```")
        if cutoff != -1:
            candidate = candidate[:cutoff].strip()
        # If there are unquoted semicolons, keep only up to the first one (single statement)
        def _first_unquoted_semicolon(s2: str) -> int:
            in_s=in_d=in_lc=in_bc=False
            i=0; n=len(s2)
            while i<n:
                ch=s2[i]; nxt=s2[i+1] if i+1<n else ''
                if in_lc:
                    if ch=='\n': in_lc=False
                    i+=1; continue
                if in_bc:
                    if ch=='*' and nxt=='/': in_bc=False; i+=2; continue
                    i+=1; continue
                if in_s:
                    if ch=="'":
                        if nxt=="'": i+=2; continue
                        in_s=False
                    i+=1; continue
                if in_d:
                    if ch=='"': in_d=False
                    i+=1; continue
                if ch=='-' and nxt=='-': in_lc=True; i+=2; continue
                if ch=='/' and nxt=='*': in_bc=True; i+=2; continue
                if ch=="'": in_s=True; i+=1; continue
                if ch=='"': in_d=True; i+=1; continue
                if ch==';': return i
                i+=1
            return -1
        cut = _first_unquoted_semicolon(candidate)
        if cut != -1:
            candidate = candidate[:cut].strip()
        return candidate
    return ""


def sanitize_sql_string(sql: str) -> str:
    """Normalize common artifacts from LLM outputs:
    - Unescape textual newlines/whitespace ("\\n", "\\r", "\\t")
    - Remove stray backslashes before newlines
    - Trim code fences remnants and excessive spaces
    """
    if not sql:
        return sql
    txt = sql.strip()
    # Unescape common sequences produced in textual JSON-like outputs
    txt = txt.replace("\\n", "\n").replace("\\r", "\n").replace("\\t", " ")
    # Remove backslashes that precede real newlines (artifacts from double-escaping)
    txt = txt.replace("\\\n", "\n")
    # Strip fenced code if leaked
    if txt.startswith("```") and txt.endswith("```"):
        inner = txt.strip("`")
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            if first.strip().lower() in ("sql", "postgresql", "trino"):
                txt = rest
            else:
                txt = inner
        else:
            txt = inner
    # Collapse excessive spaces around newlines and drop stray trailing backslashes
    import re
    raw_lines = [ln.rstrip() for ln in txt.splitlines()]
    # First pass: strip, drop trailing backslashes
    lines = []
    for ln in raw_lines:
        s = ln.strip()
        s = re.sub(r"\\+$", "", s)
        lines.append(s)
    # Second pass: remove a trailing comma if the next non-empty line starts with a clause keyword
    clause_re = re.compile(r"^(from|where|group\s+by|having|order\s+by|limit|union|intersect|except)\b", re.IGNORECASE)
    cleaned = []
    for i, ln in enumerate(lines):
        if ln.endswith(','):
            # look ahead to next non-empty line
            j = i + 1
            next_nonempty = None
            while j < len(lines):
                if lines[j]:
                    next_nonempty = lines[j]
                    break
                j += 1
            if next_nonempty and clause_re.match(next_nonempty):
                ln = ln[:-1]
        cleaned.append(ln)
    txt = "\n".join(cleaned)
    return txt
