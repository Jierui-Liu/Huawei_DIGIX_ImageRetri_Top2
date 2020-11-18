# #train
# ./runmodel.sh ./models/test_B/mutation/densenet169_RAG_704.py
# ./runmodel.sh ./models/test_B/mutation/densenet169_nonlocal_640.py
# ./runmodel.sh ./models/test_B/mutation/densenet169_attention_704.py

# #remove fc
# ./copymodel_nofc.sh ./models/test_B_nofc/mutation/densenet169_RAG_704.py 
# ./copymodel_nofc.sh ./models/test_B_nofc/mutation/densenet169_nonlocal_640.py 
# ./copymodel_nofc.sh ./models/test_B_nofc/mutation/densenet169_attention_704.py 

# #merge into one model with common backbone(densenet169)
# python ./models/test_B_nofc/mutation/m_densenet169_multi_nofc.py

#extract and index (inference)
./extract_index.sh ./models/test_B_nofc/mutation/m_densenet169_multi_nofc.py 

#fusion
./score_fusion.sh ./score_models/vote_n_fusion_coef/mutation/baseline_1.py 
