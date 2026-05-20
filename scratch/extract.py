import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        # The namespace for Word XML
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        # Find all text nodes
        texts = tree.findall('.//w:t', namespaces)
        
        full_text = []
        for text in texts:
            if text.text:
                full_text.append(text.text)
                
        return ''.join(full_text)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        text = extract_text_from_docx(input_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted text to {output_path}")
    else:
        print("Usage: python extract.py <input.docx> <output.txt>")
