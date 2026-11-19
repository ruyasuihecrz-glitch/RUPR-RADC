from .evidence_reranking import reader_text
from .model_inputs import messages

def reader_messages(row, prompt, root):
    """Only legal image/news/ordered records enter the model."""
    lookup = {source['source_id']: source for source in row['sources']}
    if len(lookup) != len(row['sources']) or sorted(lookup) != sorted(row['order']):
        raise ValueError('Full evidence ordering must preserve each record exactly once')
    return messages(reader_text(row, [lookup[key] for key in row['order']], prompt), root / row['image_path'])
