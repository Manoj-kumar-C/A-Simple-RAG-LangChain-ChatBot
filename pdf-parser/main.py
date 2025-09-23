import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

resume_text = extract_text_from_pdf("d:/AI Agent project/pdf-parser/resume.pdf")
print(resume_text)
