from flask import Flask, request, jsonify
# from script import fn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def fn(usern, usercategory, productId):
    
    categoryd = pd.read_csv('category_w_desc.csv', index_col=None)
    usert = pd.read_csv('userinfo.csv', index_col=None)
    eventd = pd.read_csv('events.csv', index_col=None)
    productd = pd.read_csv('product.csv', index_col=None)

    if usern not in usert['UserId'].values:
      return np.array(['Invalid User ID'])

    def get_relevant_users_for_category(usercategory, eventd, usern, usert):
      result = eventd[(eventd["categoryId"] == usercategory) & (eventd["Event"] == 3)].groupby("UserId")["Event"].count()
      result = result.reset_index()
      result = result.sort_values(by = 'Event', ascending = False)
      result = result[0:100] # users who have reveiwed the product in this category
      new_entry = {'UserId': usern, "Event": 0}
      # Check if the primary key value exists
      if new_entry['UserId'] not in result['UserId'].values:
          new_df = pd.DataFrame([new_entry])
          result = pd.concat([result, new_df], ignore_index=True)
      temp = pd.merge(result["UserId"], eventd, on="UserId", how="left")
      usrr = temp[(temp["categoryId"] == usercategory) & (temp["Event"] == 4)].groupby("UserId")["Event"].count().reset_index()
      temp1 = pd.merge(result["UserId"], usrr, on="UserId", how="left")
      usrb = temp[(temp["categoryId"] == usercategory) & (temp["Event"] == 3)].groupby("UserId")["Event"].count().reset_index()
      temp1 = pd.merge(temp1, usrb, on="UserId", how="left", suffixes=('_review', '_buy'))
      usrw = temp[(temp["categoryId"] == usercategory) & (temp["Event"] == 2)].groupby("UserId")["Event"].count().reset_index()
      temp1 = pd.merge(temp1, usrw, on="UserId", how="left", suffixes=('_buy', '_wishlist'))
      usrc = temp[(temp["categoryId"] == usercategory) & (temp["Event"] == 1)].groupby("UserId")["Event"].count().reset_index()
      temp1 = pd.merge(temp1, usrc, on="UserId", how="left", suffixes=('_wishlist', '_cart'))
      temp1 = pd.merge(temp1, usert, on="UserId", how="left")
      temp1 = temp1.fillna(0)
      return temp1

    def get_similar_users(id, temp1):
        cosine_sim = cosine_similarity(temp1.values[:,1:],temp1.values[:,1:]) #exclude user ids
        indices = pd.Series(temp1.index, index=temp1['UserId'])
        idx = indices[id]

        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:11]

        user_ids = [i[0] for i in sim_scores]

        return temp1['UserId'].iloc[user_ids]

    def get_product_recommendation_based_on_user_similarity(usern,usercategory, eventd, usert):
      similar_users_in_category = get_relevant_users_for_category(usercategory, eventd, usern, usert)
      rec = get_similar_users(usern,similar_users_in_category).values
      rec = np.append(rec, usern)
      simuser = pd.DataFrame(rec, columns=["UserId"])
      simuser = simuser.assign(categoryId=usercategory)
      pdt = pd.merge(simuser, eventd, on=["UserId","categoryId"], how="left")
      pdt = pdt.dropna()
      pdt.drop('Rating', axis=1, inplace=True)
      pdt['similarity_score'] = 0
      for index, row in pdt.iterrows():
        even = pdt.loc[index, 'Event']
        usertemp = pdt.loc[index, 'UserId']
        tempc = pd.merge(pdt[(pdt["Event"] == even)&(pdt["UserId"] == usertemp)], pdt[(pdt["Event"] == 3)&(pdt["UserId"] == usern)], on = 'productId').shape[0]
        if tempc != 0:
          tempc = tempc/(pdt[(pdt["Event"] == even)&(pdt["UserId"] == usertemp)]["productId"].count())
        pdt.loc[index, 'similarity_score'] = tempc
      pdt = pdt.groupby('productId')['similarity_score'].sum().reset_index()
      pdt = pd.merge(pdt, productd[['productId', 'rating',	'quantity_sold']], on=["productId"], how="left")
      pdt['final_score'] = 0
      for index, row in pdt.iterrows():
        sscore = pdt.loc[index, 'similarity_score']
        quantity = pdt.loc[index, 'quantity_sold']
        ratings = pdt.loc[index, 'rating']
        pdt.loc[index, 'final_score'] = 1*tempc + 0.4*ratings + 0.2*quantity
      pdt = pdt.sort_values(by='final_score', ascending=False)
      return pdt['productId'][0:10]
    
    def get_product_similarity_based_recommendation(categoryd, usercategory, productd ,id):
      category_data  = categoryd[categoryd["categoryId"]==usercategory].reset_index(drop=True)
      df = productd[productd['categoryId']==usercategory].reset_index(drop=True)
      df = pd.merge(df,category_data, on=["productId", "categoryId"], how="left")
      name = df['name'].values
      indices = pd.Series(df.index, index=df['productId'])
      idx = indices[id]
      tfidf_vectorizer = TfidfVectorizer()
      tfidf_matrix = tfidf_vectorizer.fit_transform(name)
      cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
      cosine_sim1 = cosine_similarity(df.values[:,2:-1], df.values[:,2:-1])
      cosine_sim2 = 0.8*cosine_sim + 0.2*cosine_sim1
      sim_scores = list(enumerate(cosine_sim2[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      sim_scores = sim_scores[1:11]
      product_ids = [i[0] for i in sim_scores]
      return df['productId'].iloc[product_ids]
    
    if(productId != -1): rec = get_product_similarity_based_recommendation(categoryd, usercategory, productd, productId).values
    else : rec = get_product_recommendation_based_on_user_similarity(usern,usercategory, eventd, usert).values

    return rec

app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        data = request.json  # Assuming frontend sends JSON data
        input_value_1 = int(data['input_value_1'])
        input_value_2 = int(data['input_value_2']) 
        input_value_3 = int(data['input_value_3'])         
        
        print(input_value_3)
        print("Before calling fn") 
        result = fn(input_value_1, input_value_2, input_value_3)
        print(result)
        print("After calling fn")

        # file_path = 'rec.json'
        # with open(file_path, 'w') as json_file:
        #     json.dump(result.tolist(), json_file)
  
        return jsonify(result = result.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port = 3000)
