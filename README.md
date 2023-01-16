IR Project
This project is designed to demonstrate the implementation of various information retrieval techniques and algorithms. The goal is to provide a fundamental concepts and methods used in the field of information retrieval.

The project is organized into different modules, each containing a specific implementation of an IR technique or algorithm. To run the code, navigate to the appropriate module and run the main script.


The config.json file is responsible for storing the configurations for the inverted index. The configurations include the settings for the indexing process and the search process.

The engine_config.json file is responsible for storing the configurations for the search engine. The configurations include the settings for the search process and the ranking process.

Examples of the configurations that can be stored in this file include:

The type of ranking algorithm to be used for ranking the documents
The parameters for the ranking algorithm, such as the value of k for BM25
The path to the pre-trained word embeddings model used for query expansion
These configurations can be modified and updated as per the requirement of the user.

When the engine starts it will read the configurations from this file to set the search engine accordingly.

IR Wikipedia Search Engine
The search engine is system that allows users to search for and retrieve relevant collection of documents. It works by indexing the documents in a specific format, such as an inverted index, and then using a ranking algorithm to rank the documents based on their relevance to the user's query.Additionally, the search engine also include other features such as query expansion improve the search experience for the users

Getting Started

The config.json file is responsible for storing the configurations for the inverted index. The configurations include the settings for the indexing process and the search process.

The engine_config.json file is responsible for storing the configurations for the search engine. The configurations include the settings for the search process and the ranking process.

Engine_MultiCore.py
The Engine_MultiCore.py file contains the implementation of a search engine that utilizes multiple cores of the CPU to perform the search operations in parallel. The main goal of this implementation is to improve the performance and efficiency of the search engine by utilizing the multiple cores of the CPU.

The factory.py file contains the implementation of a factory pattern that is used to create and manage the different objects used in the search engine
such as tokenizer and stemmer 
It also contains methods for providing interface for creating objects and encapsulating the process of instantiating the objects. This allows for a more flexible and efficient way to manage the objects used by the search engine.

The InvertedIndex.py file contains the implementation of an inverted index data structure, which is a data structure used to improve the efficiency and performance of search engines.

The Query.py file  contains the implementation of a Query class, which is used to process the user's search query and retrieve relevant documents from the inverted index.
The Query class has methods for tokenizing and normalizing the query, removing stop words and stemming the query.

