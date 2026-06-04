"""Convert Report.md to Report.pdf via HTML + headless Chrome.

Pipeline:
  1. Markdown -> HTML (with `markdown` lib, tables + fenced_code + footnotes ext)
  2. Wrap in HTML template (academic CSS + CJK font + MathJax for $...$ LaTeX)
  3. Headless Chrome --print-to-pdf with virtual-time-budget so MathJax finishes
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

PROJ = Path(__file__).resolve().parent.parent
SRC_MD  = PROJ / "report" / "Report.md"
OUT_HTML = PROJ / "report" / "Report.html"
OUT_PDF  = PROJ / "report" / "Report.pdf"
CHROME = Path(os.environ.get("CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe"))

print(f"[info] reading {SRC_MD}")
md_text = SRC_MD.read_text(encoding="utf-8")

# Convert markdown -> body HTML
print("[info] markdown -> html")
md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "footnotes", "attr_list"],
    output_format="html5",
)
body_html = md.convert(md_text)

# Wrap in HTML template with CSS + MathJax
template = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>Term Project Report</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
@page {{
  size: A4;
  margin: 14mm 16mm 14mm 16mm;
}}
body {{
  font-family: "Microsoft JhengHei", "Microsoft YaHei", "PMingLiU", "SimSun", "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #1a1a1a;
  max-width: 760px;
  margin: 0 auto;
  orphans: 3;
  widows: 3;
}}
h1 {{
  font-size: 17pt;
  margin-top: 1.0em;
  margin-bottom: 0.5em;
  padding: 0.25em 0.5em 0.25em 0.7em;
  border-left: 5px solid #c44e52;
  background: linear-gradient(90deg, #fbeaea 0%, #fff 60%);
  color: #2a2a2a;
  page-break-after: avoid;
}}
h1:first-of-type {{
  font-size: 26pt;
  text-align: center;
  border-left: none;
  border-bottom: 3px double #c44e52;
  background: none;
  margin-top: 28%;
  padding: 0 0 0.3em 0;
  color: #1a1a1a;
  page-break-after: avoid;
}}
h2 {{
  font-size: 12.5pt;
  margin-top: 1.0em;
  margin-bottom: 0.3em;
  padding-left: 0.4em;
  border-left: 3px solid #4c72b0;
  color: #2a4d80;
  page-break-after: avoid;
}}
h3 {{
  font-size: 11pt;
  margin-top: 0.8em;
  margin-bottom: 0.2em;
  color: #333;
  page-break-after: avoid;
}}
h3::before {{
  content: "▸ ";
  color: #c44e52;
}}
/* explicit cover page break marker via <hr> after TOC */
hr.page-break {{
  border: 0;
  page-break-after: always;
  margin: 0;
}}
h2 {{
  font-size: 12.5pt;
  margin-top: 0.9em;
  margin-bottom: 0.25em;
  color: #222;
}}
h3 {{
  font-size: 11pt;
  margin-top: 0.7em;
  margin-bottom: 0.2em;
  color: #333;
}}
p {{
  margin: 0.4em 0;
  text-align: left;
  word-break: normal;
}}
strong {{
  font-weight: 700;
  color: #000;
}}
table {{
  border-collapse: collapse;
  margin: 0.8em auto;
  font-size: 9.5pt;
  max-width: 100%;
  page-break-inside: avoid;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
th, td {{
  border: 1px solid #d0d0d0;
  padding: 5px 9px;
  text-align: left;
  vertical-align: top;
  word-wrap: break-word;
}}
th {{
  background: #2a4d80;
  color: white;
  font-weight: 700;
  border-color: #2a4d80;
}}
tr:nth-child(even) td {{
  background: #f7f9fc;
}}
tr:nth-child(odd) td {{
  background: #fff;
}}
code {{
  font-family: Consolas, "Courier New", monospace;
  font-size: 9.5pt;
  background: #f3f3f3;
  padding: 1px 3px;
  border-radius: 3px;
}}
pre {{
  background: #f6f6f6;
  border-left: 3px solid #999;
  padding: 6px 10px;
  margin: 0.4em 0;
  font-size: 9pt;
  overflow-x: auto;
}}
pre code {{
  background: transparent;
  padding: 0;
}}
blockquote {{
  margin: 0.6em 0;
  padding: 0.5em 0.8em;
  border-left: 4px solid #f4b860;
  background: #fff8ec;
  color: #4a3a1a;
  font-size: 10pt;
  border-radius: 0 4px 4px 0;
}}
img {{
  max-width: 100%;
  max-height: 42vh;
  display: block;
  margin: 0.4em auto;
  page-break-inside: avoid;
}}
ul, ol {{
  margin: 0.3em 0;
  padding-left: 1.4em;
}}
li {{
  margin: 0.15em 0;
}}
hr {{
  border: 0;
  border-top: 1px solid #ccc;
  margin: 0.8em 0;
}}
/* figure caption rendered as bold + small text */
p strong:first-child {{
  font-weight: 700;
}}
/* keep image + caption together */
img + p {{
  page-break-before: avoid;
}}
/* figure caption: bold + light gray panel */
p strong:first-child {{
  font-weight: 700;
}}
img + p,
p:has(> strong:first-child) {{
  text-align: center;
  font-size: 9.5pt;
  color: #555;
  margin-top: -0.2em;
}}
/* highlight red numbers */
span[style*="color"] {{
  font-weight: 700;
  font-size: 11.5pt;
}}
/* inline code: subtle accent */
p code, li code {{
  color: #2a4d80;
}}
/* horizontal rule: more subtle */
hr {{
  border: 0;
  border-top: 1px solid #e0e0e0;
  margin: 1.0em 0;
}}
/* cover meta line: bigger and styled */
h1:first-of-type + p {{
  text-align: center;
  font-size: 13pt;
  margin-top: 0.8em;
  color: #2a4d80;
}}
h1:first-of-type + p + p {{
  text-align: center;
  font-size: 10.5pt;
  color: #666;
}}
/* table of contents: nicer styling */
h2:first-of-type ~ ul {{
  background: #f7f9fc;
  padding: 0.6em 1.5em;
  border-radius: 4px;
  border: 1px solid #e0e6ed;
}}
</style>
</head>
<body>
{body}
</body>
</html>
"""
html = template.format(body=body_html)

print(f"[info] writing {OUT_HTML}")
OUT_HTML.write_text(html, encoding="utf-8")

# Chrome headless print
print(f"[info] running headless Chrome -> {OUT_PDF}")
if not CHROME.exists():
    sys.exit(f"Chrome not found at {CHROME}")

cmd = [
    str(CHROME),
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--virtual-time-budget=15000",  # let MathJax finish
    "--no-pdf-header-footer",
    f"--print-to-pdf={OUT_PDF}",
    OUT_HTML.as_uri(),
]
print("[cmd]", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print(r.stdout)
if r.returncode != 0:
    print("[stderr]", r.stderr, file=sys.stderr)
    sys.exit(f"Chrome failed: rc={r.returncode}")

if not OUT_PDF.exists():
    sys.exit(f"PDF not created at {OUT_PDF}")

print(f"\n[ok] PDF size: {OUT_PDF.stat().st_size / 1024:.0f} KB")
print(f"[ok] {OUT_PDF}")
