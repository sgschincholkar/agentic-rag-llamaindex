"""Load the PDF, split it into chunks, and build the two indexes."""

from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_SIZE, PDF_PATH


def load_nodes():
    """Read the PDF and split it into nodes (chunks)."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}. Place your file at data/state.pdf."
        )
    documents = SimpleDirectoryReader(input_files=[str(PDF_PATH)]).load_data()
    print(f"Loaded {len(documents)} document page(s).")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE)
    return splitter.get_nodes_from_documents(documents)


def build_indexes():
    """Return (summary_index, vector_index). Calls the embedding API."""
    nodes = load_nodes()
    return SummaryIndex(nodes), VectorStoreIndex(nodes)
