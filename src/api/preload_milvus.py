# Converts a text document into embeddings and loads milvus
# To run Script:
#   python preload_milvus.py --collection mycollection --from-file how-to-justin-clean.txt

import argparse
import os
import sys
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.milvus_helpers import AIAMilvus

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def read_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
        title = f.name
    return content, title


parser = argparse.ArgumentParser(description='Preload Milvus database')
parser.add_argument('--collection', help='Collection name to load or create if not existing' )
parser.add_argument('--from-dir', help='Load documents from directory of files')
parser.add_argument('--from-file', help='Filename to load text from' )
parser.add_argument('--chunk-size', help='Optional text chunk size to split the document into', default=1000)
parser.add_argument('--chunk-overlap', help='Optional chunk overlap', default=200)
args = parser.parse_args()

if not args.collection:
    print('Must specify --collection')
    sys.exit(1)
if not (args.from_dir or args.from_file):
    print('Must specify either --from-dir or --from-file')
    sys.exit(1)


# prepare documents
docs = []
if args.from_dir:
    for filename in os.listdir(args.from_dir):
        content, title = read_file(os.path.join(args.from_dir, filename))
        docs.append( Document(page_content=content, metadata={'page_title':title}) )
if args.from_file:
    content, title = read_file(args.from_file)
    docs.append( Document(page_content=content, metadata={'page_title':title}) )
print(f'docs: {len(docs)}')

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = args.chunk_size,
    chunk_overlap = args.chunk_overlap,
    length_function=len
)
source_chunks = text_splitter.split_documents(docs)
print(f'chunks: {len(source_chunks)}')

# prepare embeddings list
print('embedding chunks')
embedding_func = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
embeddings_list = embedding_func.embed_documents([d.page_content for d in source_chunks])

print('pushing embeddings to Milvus')

# load embeddings into milvus
AIAMilvus.from_documents(
    documents = source_chunks,
    embedding = embedding_func,
    collection_name = args.collection,
    embeddings_list = embeddings_list,
    milvus_bypass_proxy = None
)

