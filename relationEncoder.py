import numpy as np

_relations = "acl - acl:relcl - advcl - advmod - amod - appos - aux - aux:pass - case - cc - cc:preconj - ccomp - compound - compound:prt - conj - cop - csubj - csubj:pass - dep - det - det:predet - discourse - dislocated - expl - fixed - flat - flat:foreign - goeswith - iobj - list - mark - nmod - nmod:npmod - nmod:poss - nmod:tmod - nsubj - nsubj:pass - nummod - obj - obl - obl:npmod - obl:tmod - orphan - parataxis - punct - reparandum - root - vocative - xcomp"
_relationsList = _relations.split(' - ')

def getRelationsDict(relationsList=_relationsList):
    """
    Returns the dictionary mapping relation to some unique integer. 0 is reserved for 'none' and 1 is reserved for 'nextSentence'
    Args:
        relationsList - (list) The list of unique relations
    Returns:
        relationsDict - (dict) The dictionary where the keys are the relations and the values are the unique integers
    """
    relationsDict = dict()
    for index, relation in enumerate(relationsList):
        relationsDict[relation] = index + 2
    relationsDict['none'] = 0
    relationsDict['nextSentence'] = 1
    return relationsDict


def OneHotEncode(dictionary, values):
    """
    Returns a one-hot encoded n x m matrix where n = # of values and m = size of dictionary.
    Args:
        dictionary - (dict) The dictionary that maps values to unqiue ints
        values - (list) The list of values to be converted
    Returns:
        encoded - (np.ndarray) The one-hot encoded values
    """    
    encoded = np.zeros((len(values), len(dictionary)))

    for i in range(len(values)):
        value = values[i]
        if value in dictionary:
            index = dictionary[value]
        else:
            index = 0
        encoded[i, index] = 1
    return encoded
