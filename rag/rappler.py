import requests
from bs4 import BeautifulSoup
import time
import json

BASE_URL = "https://www.rappler.com/newsbreak/fact-check/page/{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_PAGES = 5
OUTPUT_JSON = "rappler_fax_checks.json"



def get_all_article_links():
    all_links = set()

    for page in range (1, MAX_PAGES + 1):
        url = BASE_URL.format(page)
        print(f"Scraping page {page}")

        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            article_tags = soup.select("article a")

            for a in article_tags:
                href = a.get("href")
                if href and "fact-check" and "rappler" in href:
                    full_url = href if href.startswith("http") else f"https://www.rappler.com{href}"
                    all_links.add(full_url)

            time.sleep(1)

        except Exception as e:
             print(f"❌ Error on page {page}: {e}")
    
    return list(all_links)


def scrape_article(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip = True) if title_tag else "No Title"
        paragraphs = soup.select("div.post-single__content.entry-content p")
        content = "\n".join(p.get_text(strip = True) for p in paragraphs)

        return {
            "url": url,
            "title": title,
            "content": content
        }

    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return None


def main():
      print("📄 Starting full scrape of Rappler Fact Check (Pages 1–100)")
      article_links = get_all_article_links();

      results = []

      for i, link in enumerate(article_links):
        print(f"({i + 1}/{len(article_links)}) Scraping article: {link}")
        article = scrape_article(link)
        if article:
            results.append(article)
        time.sleep(1)

      with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


      print(f"\n🎉 DONE: {len(results)} articles saved to {OUTPUT_JSON}")



if __name__ == "__main__":
    main()
