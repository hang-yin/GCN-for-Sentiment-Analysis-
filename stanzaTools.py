import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt



def createAdjacencyMatrix(edgeList, numNodes):
    """
    Creates an numNodes x numNodes adjacency matrix from the edgeList
    Args:
        edgeList - (list) The list of edges from getNodeEdgeLists
        numNodes - (int) The number of nodes in the graph
    Returns:
        newAdjMatrix - (np.ndarray) The adjacency matrix
    """
    newAdjMatrix = np.zeros((numNodes, numNodes))
    for edge in edgeList:
        newAdjMatrix[edge['edgePair'][0]-1, edge['edgePair'][1]-1] = 1
    return newAdjMatrix


def convertToEmbedding(words, preprocessor, encoder):
    """
    Takes a list of words and converts it to a list of embeddings. 

    Args:
        words - (list) The list of words to convert
        preprocessor - (tensorflow_hub.keras_layer.KerasLayer) The preprocessor needed to process a string to tokens
        encoder - (tensorflow_hub.keras_layer.KerasLayer) The encoder needed to convert the tokens to embeddings
    Returns:
        embeddings - (list) A list of embeddings
    """

    convertedWords = np.array(
        encoder(preprocessor(tf.constant(words)))['pooled_output'])
    return convertedWords


def convertToCustomEmbedding(words, preprocessor, encoder):
    """
    Don't Use This Function, use the function in the DependencyParsing.ipynb
    Takes a list of words and converts it to a list of custom embeddings. 

    Args:
        words - (list) The list of words to convert
        preprocessor - (tensorflow_hub.keras_layer.KerasLayer) The preprocessor needed to process a string to tokens
        encoder - (tensorflow_hub.keras_layer.KerasLayer) The encoder needed to convert the tokens to embeddings
    Returns:
        embeddings - (list) A list of embeddings
    """
    
    convertedWords = np.array(
        encoder(preprocessor(words)['input_word_ids'])[:, 1:2, :])
    return convertedWords.reshape((convertedWords.shape[0], convertedWords.shape[2]))

def tokenRelationHead(sent_dict):
    """
    Prints the token - relation - head chart
    Args:
        sent_dict - (list) The dictionary from sentence.to_dict()
    Returns:
        
    """
    
    print ("{:<15} | {:<10} | {:<15} ".format('Token', 'Relation', 'Head'))
    print ("-" * 50)

    # iterate to print the token, relation and head
    for word in sent_dict:
      print ("{:<15} | {:<10} | {:<15} ".format(
          str(word['text']),
          str(word['deprel']),
          str(sent_dict[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')))


def drawDepGraph(nodeList, edgeList):
    """
    Draws the dependency graph for a sentence. The words are nodes and the edges are the relations
    """
    
    G = nx.DiGraph()
    G.add_nodes_from(range(1, len(nodeList) + 1))
    nodeLabels = dict((node['id'], str(node['id']) + " : " + node['text']) for node in nodeList)
    
    edgeLabels = []
    for edge in edgeList:
        G.add_edge(*edge['edgePair'])
        edgeLabels.append((edge['edgePair'], edge['edgeLabel']))
    
    edgeLabels = dict(edgeLabels)

    plt.figure(3,figsize=(12,12)) 
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, labels=nodeLabels, node_size=2000, node_color='#B5EAD7', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels, font_size=8)
    plt.show()


def getNodeEdgeLists(doc):
    """
    Parses all the edges in sent_dict and extracts the edges, node labels, and edge labels.
    Args:
        doc - (stanza.models.common.doc.Document) The doc object
    Returns:
        nodeList - (list) A list of dictionaries, the keys are the same as the items inside a sentence object.
        edgeList - (list) A list of dictionaries, the keys are "edgePair", "edgeLabel"
    """
    edgeList = []
    nodeList = []
    modifier = 0 
    wordLimit = 50
    maxSentences = 3
    sentences = []
    for sentence in doc.sentences:
        if sentence.sentiment != 1:
            sentences.append(sentence)
    if len(sentences) > 0:
        sentences = sentences[0:maxSentences]
    else:
        sentences = doc.sentences[0:maxSentences]
    for sentence in sentences:
        for node in sentence.to_dict()[0:wordLimit]:
            node['id'] += modifier
            node['head'] += modifier
            nodeList.append(node)
            
            # if modifier and node['id'] == modifier + 1:
            #     edgePair = (node['id'] - 1, node['id'])
            #     edgeLabel = 'nextSentence'
            #     edgeList.append(
            #         {
            #             "edgePair" : edgePair,
            #             "edgeLabel" : edgeLabel
            #         }
            #     )
            if (node['head'] != modifier and node['head'] <= modifier + wordLimit):
                # the first is the head and the second is dependent
                edgePair = (node['head'], node['id'])
                edgeLabel = node['deprel']
                edgeList.append(
                    {
                        "edgePair" : edgePair,
                        "edgeLabel" : edgeLabel,
                    }
                )
        modifier += len(sentence.to_dict()[0:wordLimit])

    return nodeList, edgeList