import pandas as pd

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
    
    '''
    #remplacer les cases vides
    val_df['CATEGORIE_TITRE'].fillna('not defined', inplace = True)
    print(val_df)
    '''
    
    #supprimer une colonne
    '''
    train_df.pop(item = 'CATEGORIE_TITRE')
    val_df.pop(item = 'CATEGORIE_TITRE')
    test_df.pop(item = 'CATEGORIE_TITRE')
    '''
    train_df.pop(item = 'LIBELLE_ARRET')
    val_df.pop(item = 'LIBELLE_ARRET')
    test_df.pop(item = 'LIBELLE_ARRET')
    
    #val.insert(column = , value = [3, 4, 6, 7], loc=0)
    
    #print les valeurs unique : 
    print(type(val_df['CATEGORIE_TITRE'].unique())) #array
    
    df = val_df
    #parcourir 
    for classe in df:
        for i in df.index:
            print(val_df['CATEGORIE_TITRE'][i])
    
    
    
    

    
if __name__ == "__main__":
    main()