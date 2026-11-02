"""references.bib from arXiv metadata (export API) — no hand-typed authors or titles."""
import re, sys, time, urllib.request, xml.etree.ElementTree as ET

IDS = dict(cdb="2608.09444", overfill="2508.08446", fixedpoints2="2610.06833", lla="2607.15456", benice="2610.31633",
           elr="2610.12582", plt="2510.24824", loopcoder2="2606.18023", sharedmem="2610.02383", decodebranch="2608.12385",
           phd="2504.14992", sps="2609.03807", heterofold="2609.32259", swat="2610.15095", cachelink="2610.15509",
           c2c="2510.03215", bellows="2610.35023", accordion="2610.32492", monitorloop="2610.18127", presenthard="2610.28498",
           loopedicl="2311.12424", padd="2503.08524", pop="2602.03295", speed="2605.06105", ouro="2510.25741",
           huginn="2502.05171", ruler="2404.06654", helmet="2410.02694", lora="2106.09685", mmlu="2009.03300",
           gsm8k="2110.14168", bbh="2210.09261", mmlupro="2406.01574", arc="1803.05457", math="2103.03874",
           humaneval="2107.03374", rope="2104.09864", mor="2507.10524", ut="1807.03819", evalplus="2305.01210",
           flashattn2="2307.08691", lstmloop="2610.33360", lightman2023lets="2305.20050", yoco="2405.05254", yocou="2604.01220", kite="2609.27294", lev="2610.20951",
           gdn="2412.06464", hymba="2411.13676", afendulev="2609.04434", latentport="2609.25053", gpqa="2311.12022", ifeval="2311.07911")
NS = {"a": "http://www.w3.org/2005/Atom"}
out = []
ids = list(IDS.items())
for i in range(0, len(ids), 20):
    chunk = ids[i:i + 20]
    url = "http://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(a for _, a in chunk)
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    got = {}
    for e in root.findall("a:entry", NS):
        aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/", 1)[-1])
        got[aid] = e
    for key, aid in chunk:
        e = got.get(aid)
        if e is None:
            print("MISSING", key, aid, file=sys.stderr); continue
        title = " ".join(e.find("a:title", NS).text.split()).replace("&", r"\&")   # LaTeX-safe
        authors = " and ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS))
        year = e.find("a:published", NS).text[:4]
        out.append(f"@article{{{key},\n  title = {{{{{title}}}}},\n  author = {{{authors}}},\n  journal = {{arXiv preprint arXiv:{aid}}},\n  year = {{{year}}}\n}}\n")
    time.sleep(3)
out.append("""@inproceedings{voita2019bottom,
  title = {{The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives}},
  author = {Voita, Elena and Sennrich, Rico and Titov, Ivan},
  booktitle = {Proceedings of EMNLP-IJCNLP 2019},
  pages = {4396--4406},
  year = {2019}
}
""")
TEX = {"ä": '{\\"a}', "ó": "{\\'o}", "Ł": "{\\L}", "Ż": "{\\.Z}", "é": "{\\'e}", "ü": '{\\"u}', "ö": '{\\"o}', "ç": "{\\c c}"}
text = "\n".join(out)
for k, v in TEX.items():
    text = text.replace(k, v)
open("references.bib", "w").write(text)
print(len(out), "entries")
