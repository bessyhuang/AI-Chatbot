from pymongo import MongoClient
import pandas

# 請先確認 MongoDB 是否開啟
#sudo service mongod restart

def get_mongodb_row(DatabaseName, CollectionName):
    client = MongoClient("localhost", 27017)
    db = client[DatabaseName]
    collection = db[CollectionName]

    cursor = collection.find({}, {"_id":1, "LibraryName":1, "ID":1, 
         "Question":1, "Answer":1, "Category":1, "Keyword":1, "RelatedQ":1, 
         "Q_WS":1, "A_WS":1, "QA_WS":1, 
         "Q_POS":1, "A_POS":1, "QA_POS":1, 
         "Q_NER":1, "A_NER":1, "QA_NER":1, 
         "Q_WS | POS":1, "A_WS | POS":1, "QA_WS | POS":1, 
         "adjusted_Category":1, "new_Cluster":1})

    Question_list = []
    Q_WS_list = []
    A_WS_list = []
    Answer_list = []
    new_Cluster_list = []
    AllField_list = []
    
    for row in cursor:
        Question_list.append(row['Question'])
        Q_WS_list.append(row['Q_WS'])
        A_WS_list.append(row['A_WS'])
        Answer_list.append(row['Answer'])
        new_Cluster_list.append(row['new_Cluster'])
        AllField_list.append((row['Question'], row['Q_WS'], row['A_WS'], row['Answer'], row['_id']))

    return Question_list, Q_WS_list, Answer_list, new_Cluster_list, AllField_list
    

def get_and_append_mongodb_row(DatabaseName, CollectionName, frame):
    client = MongoClient("localhost", 27017)
    db = client[DatabaseName]
    collection = db[CollectionName]
    
    cursor = collection.find({}, {"Question":1, "Q_WS":1, "Category":1, "_id":1})
    id_list = []    
    for row in cursor:
        id_list.append(row["_id"])
        
    frame['id'] = id_list
    print(frame.loc[0].Cluster, frame.loc[0].Q)
    for i in range(len(frame)):
        cursor = collection.find({"_id":frame.loc[i].id}, {"Question":1, "Category":1, "_id":1})
        # print('---', [i for i in cursor], frame.loc[i].Question, frame.loc[i].Cluster)
        new_cluster_label = "Cluster {}".format(frame.loc[i].Cluster)
        collection.update_many({ "_id" : frame.loc[i].id },
                           { "$set" : { "new_Cluster" : new_cluster_label } }, upsert=True)
         
    return "Complete!"
