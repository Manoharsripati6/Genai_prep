from collections import Counter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, ListItem, Table, Image

# PDF Parsing
elments = partition_pdf(
    filename="data/Sample-Annual-report-and-financial-statement-formats.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    extract_images_in_pdf = True,
    languages = ["en"],
    extract_image_block_output_dir = "./data/images"
)

print("Document Elements")
counts = Counter(el.category for el in elments)
print(counts)

text_elements = [e for e in elments if isinstance(e, NarrativeText)]
print(len(text_elements))
print(text_elements)

for e in text_elements[:20]:
    print(e.text)
    print("\n")

image_elements = [e for e in elments if isinstance(e, Image)]
print(len(image_elements))

for image in image_elements[:20]:
    print(image.metadata.image_path)
    print("\n")

text_and_image_elements = [e for e in elments if isinstance(e, (NarrativeText, Image))]

for e in text_and_image_elements:
    if isinstance(e, NarrativeText):
        print(e.text)
    elif isinstance(e, Image):
        print(e.metadata.image_path)
