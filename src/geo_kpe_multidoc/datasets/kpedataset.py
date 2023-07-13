from torch.utils.data import Dataset


class KPEDataset(Dataset):
    """Suported Evaluation Datasets"""

    def __init__(self, name, ids, documents, labels, transform=None):
        """
        Parameters
        ----------
            name: str = Name of the Dataset
            ids: Document or Topic names
            documents: List of documents, one per id, or List of List of Documents per topic, 1 topic to many documents.
            labels: List of keyphrases per document, or list of keyphrases per topic.
            transform: Optional[Callable]: Optional transform to be applied
                on a sample. NOT USED
        """
        self.name = name
        self.ids = ids
        self.documents = documents
        self.labels = labels
        self.transform = None

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        """
        Returns:
        --------
            name: id of the document
            document: txt content
            labels: gold keyphrases for document
        """
        return self.ids[idx], self.documents[idx], self.labels[idx]
