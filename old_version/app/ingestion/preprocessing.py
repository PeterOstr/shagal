RE_PREFIX = re.compile(r'^\s*(re|fw|fwd):\s*', flags=re.IGNORECASE)

def normalize_subject(subj: str) -> str:
    s = subj or ""
    while True:
        ns = RE_PREFIX.sub('', s).strip()
        if ns == s:
            break
        s = ns
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def participants_list(row) -> list[str]:
    def _norm(x):
        if not x:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        return [p.strip() for p in re.split(r'[;,]', str(x)) if p.strip()]
    people = _norm(row.get("from_addr")) + _norm(row.get("to_addr")) \
           + _norm(row.get("cc_addr")) + _norm(row.get("bcc_addr"))
    return sorted(set(people))

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n")
    t = RE_HDR.sub("\n", t)
    t = RE_QUOTED.sub("", t)
    t = "\n".join([ln.strip() for ln in t.split("\n") if ln.strip()])
    return t