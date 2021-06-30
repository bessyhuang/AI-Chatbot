import pickle

with open('./food_expand_dict.pkl', 'rb') as fp:
    wiki_food_dict = pickle.load(fp)
fp.close()

print(wiki_food_dict['借書'])
