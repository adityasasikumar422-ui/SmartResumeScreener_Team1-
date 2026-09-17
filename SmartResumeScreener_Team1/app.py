
# app.py  --  Smart Resume Screener
# Run:  python app.py
# Open: http://127.0.0.1:5000

import re, pickle
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

with open("resume_pipeline.pkl", "rb") as f:
    pl = pickle.load(f)
model, tfidf_v, le = pl["model"], pl["tfidf"], pl["le"]
JOB_SKILLS, JOBS_DB, ALL_SKILLS = pl["job_skills"], pl["jobs_db"], pl["all_skills"]

STOP = {"the","is","in","and","of","to","a","an","for","on","with"}

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 2)

def get_skills(text):
    return sorted([s for s in ALL_SKILLS if s in text.lower()])

def get_fit(rs, js):
    rs, js = set(rs), set(js)
    return round(len(rs & js) / len(js) * 100, 1) if js else 0.0

def screen(text):
    vec    = tfidf_v.transform([clean_text(text)])
    idx    = model.predict(vec)[0]
    skills = get_skills(text)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        conf = round(float(prob.max()) * 100, 1)
        top3_idx = prob.argsort()[::-1][:3]
        top3 = [(le.classes_[i], round(float(prob[i])*100,1)) for i in top3_idx]
    else:
        conf = 100.0; top3 = [(le.classes_[idx], 100.0)]
    recs = []
    for job in JOBS_DB:
        sc = get_fit(skills, job["skills"])
        matched = [s for s in skills if s in job["skills"]]
        missing = [s for s in job["skills"] if s not in skills]
        recs.append({**job, "fit_score": sc, "matched": matched, "missing": missing})
    recs.sort(key=lambda x: -x["fit_score"])
    return {"role": le.classes_[idx], "conf": conf,
            "top3": top3, "skills": skills, "jobs": recs[:4]}

PAGE = """
<!DOCTYPE html><html lang=en><head><meta charset=UTF-8>
<title>Smart Resume Screener</title>
<style>
  body{background:#0f172a;color:#e2e8f0;font-family:Arial,sans-serif;padding:2rem}
  .wrap{max-width:860px;margin:auto}
  h1{color:#38bdf8;font-size:2rem;margin-bottom:.3rem}
  .sub{color:#94a3b8;margin-bottom:1.8rem}
  textarea{width:100%;height:170px;background:#1e293b;color:#e2e8f0;
            border:1px solid #334155;border-radius:8px;
            padding:.9rem;font-size:.95rem;resize:vertical}
  button{margin-top:.9rem;padding:.7rem 2.4rem;background:#0ea5e9;
          color:#fff;border:none;border-radius:8px;
          font-size:1rem;cursor:pointer;font-weight:bold}
  button:hover{background:#0284c7}
  .card{background:#1e293b;border-radius:10px;padding:1.4rem;margin-top:1.4rem}
  .card h3{color:#38bdf8;margin-bottom:.9rem;font-size:1rem}
  .badge{background:#0ea5e9;color:#fff;padding:.35rem .9rem;
          border-radius:20px;font-weight:bold;font-size:1.05rem}
  .conf{color:#86efac;margin-left:1rem}
  .tag{background:#1e3a5f;color:#93c5fd;border-radius:4px;
        padding:.15rem .45rem;margin:.15rem;font-size:.8rem;display:inline-block}
  .miss{background:#3b1f1f;color:#fca5a5}
  .job{background:#0f172a;border-radius:8px;padding:.9rem;
        margin-bottom:.8rem;border-left:4px solid #0ea5e9}
  .job h4{color:#f1f5f9;font-size:.95rem;margin-bottom:.3rem}
  .meta{color:#94a3b8;font-size:.82rem;margin:.25rem 0}
  .high{color:#86efac;font-weight:bold}
  .mid{color:#fbbf24;font-weight:bold}
  .low{color:#f87171;font-weight:bold}
</style></head><body>
<div class=wrap>
  <h1>&#127807; Smart Resume Screener</h1>
  <p class=sub>Paste your resume to get role prediction and job recommendations.</p>
  <form method=POST>
    <textarea name=resume placeholder='Paste resume here...'>{{resume}}</textarea>
    <button type=submit>Analyse Resume &#10132;</button>
  </form>
  {%if result%}
  <div class=card>
    <h3>Predicted Job Role</h3>
    <span class=badge>{{result.role}}</span>
    <span class=conf>{{result.conf}}% confidence</span>
    <div style=margin-top:.7rem>
      Top-3: {%for r,c in result.top3%}<span class=tag>{{r}} {{c}}%</span>{%endfor%}
    </div>
  </div>
  <div class=card>
    <h3>Detected Skills ({{result.skills|length}})</h3>
    {%for s in result.skills%}<span class=tag>{{s}}</span>{%endfor%}
  </div>
  <div class=card>
    <h3>Job Recommendations</h3>
    {%for j in result.jobs%}
    <div class=job>
      <h4>{{j.title}} &mdash; {{j.company}}</h4>
      <div class=meta>{{j.location}} | {{j.salary}} | {{j.experience}}</div>
      <div>Fit: <span class={% if j.fit_score>=70 %}high{% elif j.fit_score>=40 %}mid{% else %}low{% endif %}>{{j.fit_score}}%</span></div>
      <div>Matched: {%for s in j.matched%}<span class=tag>{{s}}</span>{%endfor%}</div>
      <div>Missing: {%for s in j.missing%}<span class="tag miss">{{s}}</span>{%endfor%}</div>
    </div>
    {%endfor%}
  </div>
  {%endif%}
</div></body></html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    result = None; rt = ""
    if request.method == "POST":
        rt = request.form.get("resume","").strip()
        if rt: result = screen(rt)
    return render_template_string(PAGE, result=result, resume=rt)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or "resume" not in data:
        return jsonify({"error": "Missing resume field"}), 400
    return jsonify(screen(data["resume"]))

@app.route("/api/jobs")
def api_jobs():
    return jsonify(JOBS_DB)

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
