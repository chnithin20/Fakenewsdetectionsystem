"""
article_fetcher.py
──────────────────
Given a URL, downloads the page and extracts clean article text + metadata
using the newspaper3k library.
"""

import httpx
from newspaper import Article
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ArticleFetchError(Exception):
    pass


class FetchedArticle:
    def __init__(self):
        self.title: str = ""
        self.text: str  = ""
        self.authors: list[str] = []
        self.publish_date = None
        self.domain: str = ""
        self.url: str = ""
        self.top_image: str = ""


async def fetch_article(url: str) -> FetchedArticle:
    """
    Download and parse an article from `url`.

    Returns a FetchedArticle with .title, .text, .authors, .domain etc.
    Raises ArticleFetchError if the URL can't be fetched or parsed.
    """
    result = FetchedArticle()
    result.url = url

    # Extract domain (used for source credibility check later)
    try:
        parsed = urlparse(url)
        result.domain = parsed.netloc.replace("www.", "")
    except Exception:
        result.domain = "unknown"

    try:
        # newspaper3k is synchronous — run in a thread pool to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_running_loop()

        def _download_and_parse():
            a = Article(url)
            a.download()
            a.parse()
            return a

        article = await loop.run_in_executor(None, _download_and_parse)

        result.title       = article.title or ""
        result.text        = article.text  or ""
        result.authors     = article.authors
        result.publish_date = article.publish_date
        result.top_image   = article.top_image or ""

        if not result.text:
            raise ArticleFetchError("No article text could be extracted from the URL.")

        logger.info(f"Fetched article: '{result.title[:60]}' from {result.domain}")
        return result

    except ArticleFetchError:
        raise
    except Exception as e:
        logger.warning(f"newspaper3k failed for {url}: {e}. Trying raw fetch.")
        return await _raw_fetch_fallback(url, result)


async def _raw_fetch_fallback(url: str, result: FetchedArticle) -> FetchedArticle:
    """
    Fallback: download the raw HTML with httpx and extract visible text
    with BeautifulSoup. Less accurate than newspaper3k but works on more sites.
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove nav / footer / script noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Grab paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 40)

        if not text:
            raise ArticleFetchError(f"Could not extract text from {url}")

        result.text  = text
        result.title = soup.title.string if soup.title else ""
        return result

    except ArticleFetchError:
        raise
    except Exception as e:
        raise ArticleFetchError(f"Failed to fetch article: {e}") from e
