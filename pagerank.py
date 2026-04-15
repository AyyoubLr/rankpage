import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    distribution = {}
    N = len(corpus)

    links = corpus[page]

    # If no outgoing links → treat as linking to all pages
    if len(links) == 0:
        for p in corpus:
            distribution[p] = 1 / N
        return distribution

    for p in corpus:
        # Base probability (random jump)
        distribution[p] = (1 - damping_factor) / N

        # Add probability if linked
        if p in links:
            distribution[p] += damping_factor / len(links)

    return distribution

def sample_pagerank(corpus, damping_factor, n):
    counts = {page: 0 for page in corpus}

    # Start from random page
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        counts[page] += 1

        model = transition_model(corpus, page, damping_factor)

        pages = list(model.keys())
        probabilities = list(model.values())

        page = random.choices(pages, weights=probabilities, k=1)[0]

    # Normalize
    pagerank = {}
    for page in counts:
        pagerank[page] = counts[page] / n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    N = len(corpus)

    # Start with equal ranks
    pagerank = {page: 1 / N for page in corpus}

    while True:
        new_rank = {}

        for page in corpus:
            total = 0

            for i in corpus:
                # Page with no links → acts like linking to all pages
                if len(corpus[i]) == 0:
                    total += pagerank[i] / N
                elif page in corpus[i]:
                    total += pagerank[i] / len(corpus[i])

            new_rank[page] = (1 - damping_factor) / N + damping_factor * total

        # Check convergence
        diff = max(abs(new_rank[p] - pagerank[p]) for p in pagerank)

        pagerank = new_rank

        if diff < 0.001:
            break

    return pagerank


if __name__ == "__main__":
    main()
