# #train
# ./runmodel.sh ./models/test_B/mutation/densenet169_RAG_704.py
# ./runmodel.sh ./models/test_B/mutation/densenet169_nonlocal_640.py
# ./runmodel.sh ./models/test_B/mutation/densenet169_attention_704.py

# #remove fc
# ./copymodel_nofc.sh ./models/test_B_nofc/mutation/densenet169_RAG_704.py 
# ./copymodel_nofc.sh ./models/test_B_nofc/mutation/densenet169_nonlocal_640.py 
# ./copymodel_nofc.sh ./models/test_B_nofc/mutation/densenet169_attention_704.py 


#extract and index (inference)
./runmodel.sh ./models/test_B_nofc/mutation/densenet169_nonlocal_640.py
./runmodel.sh ./models/test_B_nofc/mutation/densenet169_RAG_704.py
./runmodel.sh ./models/test_B_nofc/mutation/densenet169_attention_704.py