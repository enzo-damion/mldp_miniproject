import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
pd.options.display.float_format = '{:.1f}'.format

min_max_scaler = MinMaxScaler()

def load_year(year):
    df_s1 = pd.read_csv(f"data/data-rf-{year}/{year}_S1_NB_FER.txt", sep="\t")
    df_s2 = pd.read_csv(f"data/data-rf-{year}/{year}_S2_NB_FER.txt", sep="\t")
    df = pd.concat([df_s1,df_s2])
    df = df.drop(labels=['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET', 'ID_REFA_LDA', 'LIBELLE_ARRET','CATEGORIE_TITRE'], axis=1)
    df['NB_VALD'] = df['NB_VALD'].replace('Moins de 5', '0')
    df = df.groupby(['JOUR']).sum(numeric_only=True)
    df = df.reset_index(drop=True)
    df[['NB_VALD']] = min_max_scaler.fit_transform(df[['NB_VALD']])
    return df

def quick_analysis(train_df, val_df, test_df):
    df = pd.concat([train_df, val_df, test_df], axis=1)
    df.columns = ['train', 'val', 'test']
    print(df.describe())
    df.plot()
    plt.show()

def main():
    # Read data
    train_df = load_year(2019)
    val_df = load_year(2020)
    test_df = load_year(2021)
    quick_analysis(train_df, val_df, test_df)

if __name__ == "__main__":
    main()