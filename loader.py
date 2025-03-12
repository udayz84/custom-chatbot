from langchain.document_loaders import UnstructuredURLLoader

def load_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()
