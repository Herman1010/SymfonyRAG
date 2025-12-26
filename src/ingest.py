import os
import glob
import re
import requests

from src import config


def download_symfony_docs():
    os.makedirs(config.RAW_DIR, exist_ok=True)

    for f in config.FILES:
        url = config.BASE_URL + f
        print("Téléchargement :", url)
        response = requests.get(url)

        if response.status_code == 200:
            with open(os.path.join(config.RAW_DIR, f), "w", encoding="utf-8") as file:
                file.write(response.text)
            print("Enregistré dans", os.path.join(config.RAW_DIR, f))
        else:
            print(f"Erreur pour {f} :", response.status_code)


def load_rsts(folder=None):
    folder = folder or config.RAW_DIR
    docs = []
    for path in glob.glob(os.path.join(folder, "*.rst")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "id": os.path.basename(path),
            "path": path,
            "text": text
        })
    return docs


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()

        if stripped.startswith(".. "):
            continue

        if stripped and len(stripped) >= 3 and set(stripped) <= set("=~`-^\"'*+_#-"):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    cleaned = re.sub(r":\w+:`([^`<]+)(?:<[^`]+>)?`", r"\1", cleaned)
    cleaned = re.sub(r"``([^`]+)``", r"\1", cleaned)

    final_lines = []
    for line in cleaned.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        final_lines.append(line)

    final_text = "\n".join(final_lines)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)

    return final_text.strip()


def extract_doc_metadata(doc):
    lines = doc["clean_text"].split("\n")
    title = None
    for line in lines:
        if line.strip():
            title = line.strip()
            break

    category = doc["id"].replace(".rst", "")
    return {"title": title, "category": category}


def extract_section_titles_from_raw(text: str):
    lines = text.split("\n")
    titles = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        deco = lines[i + 1].strip()

        if line and deco and len(deco) >= 3 and set(deco) <= set("=~`-^\"'*+_#-"):
            titles.append(line)
            i += 2
        else:
            i += 1
    return titles


def prepare_docs(docs):
    for doc in docs:
        doc["clean_text"] = clean_text(doc["text"])
        doc["metadata"] = extract_doc_metadata(doc)
        doc["section_titles"] = extract_section_titles_from_raw(doc["text"])
    return docs


if __name__ == "__main__":
    download_symfony_docs()
    docs = load_rsts()
    docs = prepare_docs(docs)
    print("Nombre de documents chargés :", len(docs))
    print("Exemple:", docs[0]["id"])
