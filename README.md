# geo-kpe-multidoc

Key-phrase extraction is commonly framed as the task of retrieving a small set of phrases that encapsulate the core concepts of an input text, usually a single document. A number of diverse approaches, either supervised or unsupervised, have been explored in the recent literature. Still, few studies have addressed the specific task of multi-document key-phrase extraction ( see https://arxiv.org/abs/2110.01073 for an exception ), despite its utility for describing and summarizing sets of documents relating to a given topic. Possible approaches involve merging and re-ranking the results from single-document key-phrase extraction (e.g. preferring key-phrase candidates that appear on multiple documents associated to the topic), although there are many possibilities for improving uppon these simple methods. 
 
In the context of the MSc thesis proposal, and building on recent unsupervised approaches to key-phrase extraction from single documents that leverage Transformer language models (e.g., see https://arxiv.org/abs/1801.04470 or https://arxiv.org/abs/2110.06651 ), the candidate will study new methods for multi-document key-phrase extraction. Particular attention will be given to:
* (a) the ability to generalize towards different languages and application domains, and 
* (b) the use of geospatial association measures (e.g., Moran's I, Geary's C, or Getis-Ord general G) to help in the ranking of key-phrase candidates.
 
Relying on the results of text geoparsing ( e.g., using pre-existing tools such as https://github.com/openeventdata/mordecai ), the methods that are to be developed will explore the idea that key-phrase candidates that are semantically closer to topic the documents, and that simultaneously appear associated to locations mentioned in the documents that are clustered together in space, should be preferred.  
 
Experiments will be performed with pre-existing datasets ( e.g., see https://github.com/OriShapira/MkDUC-01 ) , assessing the degree to which the measures of geospatial association can indeed contribute to improved results.
