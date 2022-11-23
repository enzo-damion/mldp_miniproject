import pandas as pd
pd.options.display.float_format = '{:.1f}'.format

def read_data(train_year=2020, val_year=2019, test_year=2021):
    # Train
    train_df_s1 = pd.read_csv(f"data/data-rf-{train_year}/{train_year}_S1_NB_FER.txt", sep="	")
    train_df_s2 = pd.read_csv(f"data/data-rf-{train_year}/{train_year}_S2_NB_FER.txt", sep="	")
    train_df = pd.concat([train_df_s1,train_df_s2], ignore_index=True)
    train_df = train_df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA', 'LIBELLE_ARRET','CATEGORIE_TITRE'], axis=1)
    train_df['JOUR']= pd.to_datetime(train_df['JOUR'], dayfirst=True)
    train_df = train_df.groupby(['JOUR']).sum(numeric_only=True)
    # Validation
    val_df_s1 = pd.read_csv(f"data/data-rf-{val_year}/{val_year}_S1_NB_FER.txt", sep="	")
    val_df_s2 = pd.read_csv(f"data/data-rf-{val_year}/{val_year}_S2_NB_FER.txt", sep="	")
    val_df = pd.concat([val_df_s1,val_df_s2], ignore_index=True)
    val_df = val_df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA', 'LIBELLE_ARRET','CATEGORIE_TITRE'], axis=1)
    val_df['JOUR']= pd.to_datetime(val_df['JOUR'], dayfirst=True)
    val_df = val_df.groupby(['JOUR']).sum(numeric_only=True)
    # Test
    test_df_s1 = pd.read_csv(f"data/data-rf-{test_year}/{test_year}_S1_NB_FER.txt", sep="	")
    test_df_s2 = pd.read_csv(f"data/data-rf-{test_year}/{test_year}_S2_NB_FER.txt", sep="	")
    test_df = pd.concat([test_df_s1,test_df_s2], ignore_index=True)
    test_df = test_df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA', 'LIBELLE_ARRET','CATEGORIE_TITRE'], axis=1)
    test_df['JOUR']= pd.to_datetime(test_df['JOUR'], dayfirst=True)
    test_df = test_df.groupby(['JOUR']).sum(numeric_only=True)
    return train_df, val_df, test_df

def quick_analysis(train_df, val_df, test_df):
    df = pd.concat([train_df, val_df, test_df], axis=1, ignore_index=True, keys='JOUR')
    df.columns = ['train', 'val', 'test']
    print(df.describe())


def main():
    train_df, val_df, test_df = read_data()
    quick_analysis(train_df, val_df, test_df)

if __name__ == "__main__":
    main()