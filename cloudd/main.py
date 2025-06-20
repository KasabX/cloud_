# 📁 Cloud-Based Document Analyzer (Student Style)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os, fitz, docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import time

# 🛠️ Init Google Drive
ga = GoogleAuth()
ga.LocalWebserverAuth()
drv = GoogleDrive(ga)

# 📦 Config
dir = 'docs'
tree = {'edu': ['school', 'univ', 'study'], 'med': ['health', 'doctor'], 'tech': ['code', 'prog']}

# 📂 Collect files
def get_txt(p):
    if p.endswith('.pdf'):
        doc = fitz.open(p)
        return " ".join([pg.get_text() for pg in doc])
    elif p.endswith('.docx'):
        d = docx.Document(p)
        return " ".join([p.text for p in d.paragraphs])
    return ""

# 🔠 Sort by title
def sort_by_title(fs):
    return sorted(fs, key=lambda x: get_txt(x).strip().split("\n")[0][:30])

# 🔍 Search
def search_docs(q, fs):
    r = []
    for f in fs:
        txt = get_txt(f)
        if q.lower() in txt.lower():
            r.append(f)
    return r

# 🧠 Classify
def classify(fs):
    data, lbls = [], []
    for f in fs:
        txt = get_txt(f)
        data.append(txt)
        lbl = 'misc'
        for cat, kws in tree.items():
            if any(k in txt.lower() for k in kws): lbl = cat
        lbls.append(lbl)
    vec = TfidfVectorizer()
    X = vec.fit_transform(data)
    clf = MultinomialNB().fit(X, lbls)
    return dict(zip(fs, clf.predict(X)))

# ⏱️ Run Core Tasks
start = time.time()
files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(('.pdf', '.docx'))]
sorted_files = sort_by_title(files)
found = search_docs('data', files)
classes = classify(files)
end = time.time()

print(f"📊 Total: {len(files)} docs")
print(f"🔎 Search found: {len(found)}")
print(f"🧾 Classes: {classes}")
print(f"⏱️ Time: {round(end-start, 2)}s")

# ☁️ Upload to Drive
for f in files:
    f_n = os.path.basename(f)
    f_g = drv.CreateFile({'title': f_n})
    f_g.SetContentFile(f)
    f_g.Upload()
    print(f"☁️ Uploaded: {f_n}")
