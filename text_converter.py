import pdfplumber
import json
import re

# Path to your PDF
pdf_path = "Qset.pdf"
# Output JSON file
json_path = "career_data.json"

# Extract text from PDF
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

# Optional: split by lines
lines = text.split("\n")

# Parse Q&A
qa_list = []
question, answer = "", ""
for line in lines:
    line = line.strip()
    if not line:
        continue
    # Detect questions (assuming it starts with "Q:" or ends with "?")
    if line.startswith("Q:") or line.endswith("?"):
        if question and answer:
            qa_list.append({"question": question, "answer": answer})
            answer = ""
        question = line.replace("Q:", "").strip()
    # Detect answers (assuming it starts with "A:")
    elif line.startswith("A:"):
        answer = line.replace("A:", "").strip()
    else:
        # Append to previous answer if multi-line
        answer += " " + line

# Add the last Q&A
if question and answer:
    qa_list.append({"question": question, "answer": answer.strip()})

# Save to JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(qa_list, f, indent=4, ensure_ascii=False)

print(f"PDF converted to JSON successfully! Total Q&A: {len(qa_list)}")
