import pandas as pd
import numpy as np

def read_data(train_year=2020, val_year=2019, test_year=2021):
    # Train
    train_df_s1 = pd.read_csv(f"data/data-rf-{train_year}/{train_year}_S1_NB_FER.txt", sep="	")
    train_df_s2 = pd.read_csv(f"data/data-rf-{train_year}/{train_year}_S2_NB_FER.txt", sep="	")
    train_df = pd.concat([train_df_s1,train_df_s2], ignore_index=True)
    train_df = train_df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA'], axis=1)
    train_df['JOUR']= pd.to_datetime(train_df['JOUR'], dayfirst=True)
    # Validation
    val_df_s1 = pd.read_csv(f"data/data-rf-{val_year}/{val_year}_S1_NB_FER.txt", sep="	")
    val_df_s2 = pd.read_csv(f"data/data-rf-{val_year}/{val_year}_S2_NB_FER.txt", sep="	")
    val_df = pd.concat([val_df_s1,val_df_s2], ignore_index=True)
    val_df = val_df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA'], axis=1)
    val_df['JOUR']= pd.to_datetime(val_df['JOUR'], dayfirst=True)
    # Test
    test_df_s1 = pd.read_csv(f"data/data-rf-{test_year}/{test_year}_S1_NB_FER.txt", sep="	")
    test_df_s2 = pd.read_csv(f"data/data-rf-{test_year}/{test_year}_S2_NB_FER.txt", sep="	")
    test_df = pd.concat([test_df_s1,test_df_s2], ignore_index=True)
    test_df = test_df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA'], axis=1)
    test_df['JOUR']= pd.to_datetime(test_df['JOUR'], dayfirst=True)
    return train_df, val_df, test_df

def split_cat0(df, nom):
    df_list = []
    for cathegorie in df['CATEGORIE_TITRE'].unique():
        print("cathégorie", cathegorie)
        jours = []
        nb_val = []
        for i in df.index:
            if df['CATEGORIE_TITRE'][i] == cathegorie:
                #print(i)
                jours+=[df['JOUR'][i]]
                nb_val+=[df['NB_VALD'][i]]
        print("fin")
        #df_list+=[pd.DataFrame({'JOUR':jours[:100],'NB_VALD':nb_val[:100]})]
        df_list+=[jours]
        df_list+=[nb_val]
    print("j'ai fini")
    df_list = np.array(df_list)
    #df_list.to_pickle("df_list_cat.pkl")
    np.save(open(nom+'.pkl', 'wb'), df_list)
    return df_list


def split_cat(df, nom):
    df_list = []
    for cathegorie in df['CATEGORIE_TITRE'].unique():
        print("cathégorie", cathegorie)
        jours = []
        nb_val = []
        for i in df.index:
            if df['CATEGORIE_TITRE'][i] == cathegorie:
                #print(i)
                jours+=[df['JOUR'][i]]
                nb_val+=[df['NB_VALD'][i]]
        print("fin")
        #df_list+=[pd.DataFrame({'JOUR':jours[:100],'NB_VALD':nb_val[:100]})]
        df_list+=[jours]
        df_list+=[nb_val]
    print("j'ai fini")
    for k in range(0, len(df_list), 2):
        df_list0 = pd.DataFrame({'JOUR' :df_list[k], 'nb_val' : df_list[k+1]})
        df_list0 = df_list0.groupby(by=['JOUR']).sum()
        np.save(open(nom+'%s'%k+'.pkl', 'wb'), df_list0)
        print("j'ai save", nom, df_list0.sum())
    return df_list

'''
def split_cat_2(df):
    cat = df['CATEGORIE_TITRE'].unique()
    df_list = [[]*len(cat)]
    for i in df.index:
        for k in range(len(cat)):
            if df['CATEGORIE_TITRE'][i] == cat[k]:
               

    for k in range(len(df['CATEGORIE_TITRE'].unique())):
        cathegorie = df['CATEGORIE_TITRE'].unique()[k]
        print("cathégorie", cathegorie)
        jours = []
        nb_val = []
        for i in df.index:
            if df['CATEGORIE_TITRE'][i] == cathegorie:
                print(i)
                jours+=[df['JOUR']]
                nb_val+=['NB_VALD']
        print("fin")
        df_list+=[pd.DataFrame({'JOUR':jours,'NB_VALD':nb_val})]
    df_list.to_pickle("df_list_cat.pkl")
    return df_list
'''
    
def main():
    
    '''
    #pour enregistrer les données 
    train_df, val_df, test_df = read_data()
    train_df.to_pickle("data_train.pkl")
    val_df.to_pickle("data_val.pkl")
    test_df.to_pickle("data_test.pkl")
    '''
    
    #pour lire les données
    train_df, val_df, test_df = pd.read_pickle("data_train.pkl"), pd.read_pickle("data_val.pkl"), pd.read_pickle("data_test.pkl")
    #print("les types\n", val_df.dtypes)
    
    '''
    #info sur le dataframe
    print (val_df.info())
    print("DESCRIBE\n", val_df.describe())
    '''
    
    
    #remplacer les cases vides
    train_df['CATEGORIE_TITRE'].fillna('not defined', inplace = True)

    
    
    #supprimer une colonne
    '''
    train_df.pop(item = 'CATEGORIE_TITRE')
    val_df.pop(item = 'CATEGORIE_TITRE')
    test_df.pop(item = 'CATEGORIE_TITRE')
    '''
    train_df.pop(item = 'LIBELLE_ARRET')
    val_df.pop(item = 'LIBELLE_ARRET')
    test_df.pop(item = 'LIBELLE_ARRET')
    
    #print les valeurs unique : 
    #print(val_df['CATEGORIE_TITRE'].unique()) #array
    
    #print(train_df.info())
    #sépare
    
    print(split_cat(train_df, 'jr_val_train'))
    print(split_cat(val_df, 'jr_val_val'))
    print(split_cat(test_df, 'jr_val_test'))
    
    
    print("je suis sortie")
    
    
    

    
if __name__ == "__main__":
    main()
    
    
    '''
    
    df = val_df
    #parcourir 
    for classe in df:
        print("classe", classe)
        #for i in df.index:
           # print(val_df['CATEGORIE_TITRE'][i])
    jours = range(10)
    nb_val = range(10)
    np.save(open('list_jr_val', 'wb'), jours)
    a = pd.DataFrame({'JOUR':jours,'NB_VALD':nb_val})
    print("A", a.info())
    
    #val.insert(column = , value = [3, 4, 6, 7], loc=0)
    '''