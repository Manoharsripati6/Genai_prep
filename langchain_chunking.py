from langchain_text_splitters  import RecursiveCharacterTextSplitter,CharacterTextSplitter,TokenTextSplitter


# Create an instance of the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunk=text_splitter.create_documents(["This is a sample text that will be split into smaller chunks. The text splitter will create chunks of 100 characters with an overlap of 20 characters between chunks. This allows for better context retention when processing the text."])
print(f"Number of chunks: {len(chunk)}")
for i in range(len(chunk)):
    print(f"Content of chunk {i}:\n{chunk[i].page_content}")

# You can also use the split_text method to split a single string into chunks
chunk2=CharacterTextSplitter(chunk_size=10, chunk_overlap=2).split_text("This is a sample text that will be split into smaller chunks. The text splitter will create chunks of 100 characters with an overlap of 20 characters between chunks. This allows for better context retention when processing the text.")
print(f"Number of chunks (split_text): {len(chunk2)}")
for i in range(len(chunk2)):
    print(f"Content of chunk {i} (split_text):\n{chunk2[i]}")
    

# Sample text
text = """
LangChain is a framework for building applications powered by large language models.
It helps with prompt management, retrieval, memory, agents, and chains.
Token-based chunking splits text according to token count instead of characters.
"""

# Create splitter
splitter = TokenTextSplitter(
    chunk_size=20,      
    chunk_overlap=5     
)

# Split text
chunks = splitter.split_text(text)

# Print chunks
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
