import os
import sqlite3
import csv
import io
from flask import Flask, render_template, request, send_file, session
import openai

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

def enhance_article(title, summary, link, api_key):
    openai.api_key = api_key
    prompt = f"""
    Given the following article title and summary:
    Title: {title}
    Summary: {summary}
    Write a 500-800 word blog post inspired by this article, tailored for business users. Focus on the use case’s value, explaining who it benefits (e.g., specific roles like managers, IT teams, or industries like tech) and why they should care (e.g., efficiency gains, cost savings, strategic advantages). Use original phrasing and structure, avoiding direct copying, but include actionable insights or recommendations for adoption. Add a citation to the original article (e.g., "Inspired by '{title}' at {link}"). Do not reproduce the original text verbatim—create a fresh perspective.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].message["content"], response.usage["total_tokens"]

@app.route("/", methods=["GET", "POST"])
def index():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT title, link, summary, score FROM articles ORDER BY score DESC")
    articles = [dict(row) for row in cursor.fetchall()]
    conn.close()

    usage_info = session.get("usage_info", {"tokens": 0, "cost": 0.0})

    if request.method == "POST":
        api_key = request.form.get("api_key")
        if not api_key:
            return render_template("index.html", articles=articles, error="Please provide an OpenAI API key!", usage_info=usage_info)

        selected_links = request.form.getlist("selected_articles")
        if not selected_links:
            return render_template("index.html", articles=articles, error="No articles selected!", usage_info=usage_info)

        selected_articles = [a for a in articles if a["link"] in selected_links]
        total_tokens = 0

        for article in selected_articles:
            try:
                blog_post, tokens_used = enhance_article(article["title"], article["summary"], article["link"], api_key)
                article["blog_post"] = blog_post
                total_tokens += tokens_used
            except Exception as e:
                return render_template("index.html", articles=articles, error=f"OpenAI error: {str(e)}", usage_info=usage_info)

        usage_info["tokens"] += total_tokens
        usage_info["cost"] += (total_tokens * 0.00175 / 1000)
        session["usage_info"] = usage_info

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["title", "link", "original_summary", "relevance_score", "blog_post"])
        for article in selected_articles:
            writer.writerow([
                article["title"],
                article["link"],
                article["summary"],
                article["score"],
                article.get("blog_post", "")
            ])
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            as_attachment=True,
            download_name="enhanced_articles.csv"
        )

    return render_template("index.html", articles=articles, usage_info=usage_info)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)ue)
