
import re

URL_RE  = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
USER_RE = re.compile(r"(?<!\w)@\w+")
NUM_RE  = re.compile(r"(?<!\w)\d+([.,]\d+)?(?!\w)")

HTML_ENT = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "
}

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    for k, v in HTML_ENT.items():
        s = s.replace(k, v)

    s = URL_RE.sub("<url>", s)
    s = USER_RE.sub("<user>", s)
    s = NUM_RE.sub("<num>", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s
