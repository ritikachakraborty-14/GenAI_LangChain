import fitz  # This is PyMuPDF

doc = fitz.open("example.pdf")
for page_index in range(len(doc)):
    for img_index, img in enumerate(doc.get_page_images(page_index)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        with open(f"image_{page_index}_{img_index}.{image_ext}", "wb") as f:
            f.write(image_bytes)
