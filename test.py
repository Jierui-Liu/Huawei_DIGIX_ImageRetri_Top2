
import sys
sys.path.append(".")
from SRC.utils.io import *
import numpy as np


file_name1_1='./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/rag/gallery.json'
file_name1_2='./densenet169_RAG_704/gallery.json'

dict1=file_to_dict(file_name1_1)
dict2=file_to_dict(file_name1_2)

print(np.max(np.sum((dict1['data']-dict2['data'])**2,1)))



file_name1_1='./densenet169_RAG_704/rag/gallery.json'
file_name1_2='./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/rag/gallery.json'

dict1=file_to_dict(file_name1_1)
dict2=file_to_dict(file_name1_2)

print(np.max(np.sum((dict1['data']-dict2['data'])**2,1)))




file_name1_1='./densenet169_RAG_704/rag/gallery.json'
file_name1_2='./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/nonlocal/gallery.json'

dict1=file_to_dict(file_name1_1)
dict2=file_to_dict(file_name1_2)

print(np.max(np.sum((dict1['data']-dict2['data'])**2,1)))



# file_name1_1='./models/test_B_nofc/embeddings/densenet169_RAG_704/gallery.json'
# file_name1_2='./densenet169_RAG_704/gallery.json'

# dict1=file_to_dict(file_name1_1)
# dict2=file_to_dict(file_name1_2)

# print(np.max(np.sum((dict1['data']-dict2['data'])**2,1)))



# file_name1_1='./models/test_B_nofc/embeddings/densenet169_RAG_704/gallery.json'
# file_name1_2='./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/rag/gallery.json'

# dict1=file_to_dict(file_name1_1)
# dict2=file_to_dict(file_name1_2)

# print(np.max(np.sum((dict1['data']-dict2['data'])**2,1)))
