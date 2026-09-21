# HKUST — MSc in AI and Entrepreneurship

Coursework from the [MSc(AIE)](https://seng.hkust.edu.hk/academics/taught-postgraduate/msc-aie/course-list) program at HKUST School of Engineering: source code, notebooks, reports, and project deliverables from ten courses across two semesters.

## Courses

### Semester 1 — Fall 2025

| Dir | Code | Course |
|---|---|---|
| [`cv`](semester-1/cv) | MAIE 5421 | Computer Vision |
| [`eai`](semester-1/eai) | MAIE 5103 | Artificial Intelligence Ethics |
| [`ml`](semester-1/ml) | MAIE 5212 | Machine Learning |
| [`mls`](semester-1/mls) | MAIE 5532 | Machine Learning System |
| [`nlp`](semester-1/nlp) | MAIE 5221 | Natural Language Processing |

### Semester 2 — Spring 2026

| Dir | Code | Course |
|---|---|---|
| [`aif`](semester-2/aif) | MAIE 5102 | AI Fundamentals: Concepts and Methods |
| [`em`](semester-2/em) | MAIE 5534 | Entrepreneurial Me |
| [`gaif`](semester-2/gaif) | MAIE 6000B | Generative AI in Finance |
| [`pai`](semester-2/pai) | MAIE 5101 | Programming for Artificial Intelligence |
| [`sus`](semester-2/sus) | MAIE 5535 | Startup Seminars for AI |

MAIE 5101, 5102, and 5103 are the program's core courses; the rest are electives.

## Highlights

- **[NLP term project](semester-1/nlp/final-project)** — an LLM-backed search engine with intent recognition, RAG retrieval, domain plugins, and multimodal input handling
- **[FedCough](semester-1/mls)** — federated learning for cough classification, with a centralized baseline for comparison
- **[Computer Vision project](semester-1/cv)** — video style transfer
- **[Generative AI in Finance mini-project](semester-2/gaif/mini-project)** — presentation deliverable, viewable in the browser

## Scope

This repository holds **my own work**: source code, notebooks, generated figures, written reports, and presentation decks.

Excluded by design:

- **Lecture slides, assignment handouts, and exam papers.** Copyright belongs to the instructors and HKUST, so they are not redistributed here.
- **Third-party papers** discussed in coursework.
- **Datasets and model weights.** CIFAR-10, `weatherAUS.csv`, HuggingFace caches, and `.pth` checkpoints are excluded by `.gitignore`. Each course README explains how to obtain them.

## Running the code

Most work is Jupyter notebooks and standalone Python scripts; open or run them directly. Two subprojects have their own dependency files and setup notes:

- [`semester-1/nlp/final-project`](semester-1/nlp/final-project) — `requirements.txt` plus `.env.example` for API keys
- [`semester-1/mls`](semester-1/mls) — `environment.yml` and `requirements.txt`

No API keys or credentials are committed. Copy `.env.example` to `.env` and supply your own.

## Academic integrity

Published as a record of completed coursework. If you are currently enrolled in any of these courses, consult your instructor's policy before looking at solutions.
