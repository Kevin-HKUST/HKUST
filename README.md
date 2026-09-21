# HKUST — MSc Coursework

Code and project artifacts from ten graduate courses at HKUST, organised by semester.

## What is in here

| | Semester 1 | Semester 2 |
|---|---|---|
| 1 | [CV — Computer Vision](semester-1/cv) | [AIF — AI Fundamentals](semester-2/aif) |
| 2 | [EAI — Ethics and AI](semester-1/eai) | [EM — Engineering Management Seminars](semester-2/em) |
| 3 | [ML — Machine Learning](semester-1/ml) | [GAIF — Generative AI in Finance](semester-2/gaif) |
| 4 | [MLS — Machine Learning Systems](semester-1/mls) | [PAI — Programming for AI](semester-2/pai) |
| 5 | [NLP — Natural Language Processing](semester-1/nlp) | [SUS — Sustainability](semester-2/sus) |

## Scope and exclusions

This repository holds **only my own work**: source code, notebooks, generated figures, and small input files.

Deliberately excluded:

- **Lecture slides, handouts, and exam papers.** Copyright belongs to the instructors and HKUST, so they are not redistributed here.
- **Report PDFs.** My submitted reports embed my name and student ID, so they are kept offline.
- **Datasets and model weights.** CIFAR-10, `weatherAUS.csv`, HuggingFace caches, and `.pth` checkpoints are excluded by `.gitignore`. Each course README explains how to obtain them.

Some courses therefore contain a README only — their deliverables were written reports rather than code.

## Running the code

Most work is Jupyter notebooks and standalone Python scripts; open or run them directly. Two subprojects have their own dependency files and setup notes:

- [`semester-1/nlp/final-project`](semester-1/nlp/final-project) — `requirements.txt` plus `.env.example` for API keys
- [`semester-1/mls`](semester-1/mls) — `environment.yml` and `requirements.txt`

No API keys or credentials are committed. Copy `.env.example` to `.env` and supply your own.

## Academic integrity

Published as a record of completed coursework. If you are currently enrolled in any of these courses, consult your instructor's policy before looking at solutions.
