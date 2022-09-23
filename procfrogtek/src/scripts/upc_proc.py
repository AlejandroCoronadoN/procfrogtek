import pandas as pd
import numpy as np

from datetime import datetime
import pandas as pd 
import sys


def rename_multiindex(df:pd.DataFrame,  id_cols:list, excluded_cols:list =[]) -> pd.DataFrame:
    """Converts multindex after pivot table

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    all_new_cols = []
    for mcol in df.columns:
        if mcol[0] not in excluded_cols:
            if mcol[0] in id_cols:
                all_new_cols.append(mcol[0])
            else:
                new_col = mcol[0] +"_"  + mcol[1]
                all_new_cols.append(new_col)
    initial_cols = [y[0] for y in df.columns]
    df.columns = all_new_cols
    return df 

# df_test =rename_multiindex(df=df_upc_unidades,  id_cols=id_cols)


def impute_values(df):
    processed_cols = []
    for col in df.columns:
        zero_condition = df[col] ==0
        zero_count = np.sum(zero_condition)
        if zero_count > len(df[col])*.5:
            del df[col]
            print(f"\n\nDELETED COL: {col}")
        else:
            impute_value = df.loc[~zero_condition, col].mean()
            df.loc[zero_condition, col] = impute_value
            unique_vals = df[col].unique()
            print(f"\n\ncol: {col} \n\tvals: {unique_vals} \n\tzero_count:{zero_count}")
            processed_cols.append(col)
    return df, processed_cols 


def trime_cols(df1:pd.DataFrame, df2:pd.DataFrame):
    """
    """
    del_df1 = [x for x in df1.columns if x not in df2.columns]
    del_df2 = [x for x in df2.columns if x not in df1.columns]

    for col in del_df1:
        del df1[col]
    for col in del_df2:
        del df2[col]
        
    return df1, df2


def impute_var_change(df):
    processed_cols = []
    for col in df.columns:
        df[col]


def melt_pivoted(df:pd.DataFrame, col:str):
    df_melt = df.reset_index().melt(id_vars=id_cols)
    df_melt = df_melt.rename(columns={"value":col})
    return df_melt


def create_experiment(col:str, df:pd.DataFrame, id_sort:list =[ 'region', 'sku', "year", "month"], id_rolling:list = [ 'region', 'sku', "year"]) -> pd.DataFrame:
    """Returns a Data Frame with all the required columns to calcula differential experiment variables
    the main columns required to perform the experiments are
    
    maginitude: The maximim change in price (accumulated)/Average price '
        of SKU at id_rolling level
    effect_strategy: Identifies when the major price change was made and lavels 
        all months aftrer the effect with a falf 1 variable

    Args:
        col (str): _description_
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_sorted = df.sort_values(id_sort)
    df_month = df_sorted["month"]
    df_diff = df_sorted.groupby(id_rolling)[col].rolling(window=2).apply(my_fun).reset_index()
    diff_col = col + "_diff"
    df_diff = df_diff.rename(columns= {col:diff_col})
    df_sorted[diff_col] = df_diff[diff_col]

    df_cumsum = df_diff.replace(np.nan, 0).groupby(id_rolling).cumsum()
    cumsum_col = col + "_cumsum"
    df_cumsum = df_cumsum.rename(columns= {diff_col:cumsum_col})
    df_sorted[cumsum_col] = np.abs(df_cumsum[cumsum_col])
    
    #? REQUIRED FORM IMPACT CALC
    df_min_cumsum = df_sorted[df_sorted[cumsum_col]!=0]
    df_min_cumsum = df_min_cumsum.replace(np.nan, 0).groupby(id_rolling).min()
    cumsum_min_col = col + "_cumsum_min"
    df_min_cumsum = df_min_cumsum.rename(columns= {cumsum_col:cumsum_min_col}).reset_index()
    df_min_cumsum = df_min_cumsum[id_rolling+ [cumsum_min_col]]
    df_sorted = df_sorted.merge(df_min_cumsum, on=id_rolling)
    
    df_max_cumsum = df_sorted[df_sorted[cumsum_col]!=0]
    df_max_cumsum = df_max_cumsum.replace(np.nan, 0).groupby(id_rolling).max()
    cumsum_max_col = col + "_cumsum_max"
    df_max_cumsum = df_max_cumsum.rename(columns= {cumsum_col:cumsum_max_col}).reset_index()
    df_max_cumsum = df_max_cumsum[id_rolling+ [cumsum_max_col]]
    df_sorted = df_sorted.merge(df_max_cumsum, on=id_rolling)

    #? NON_REQUIRED
    df_min = df_sorted[df_sorted[cumsum_col]!=0]
    df_min = df_min.replace(np.nan, 0).groupby(id_rolling).min()
    min_col = col + "_min"
    df_min = df_min.rename(columns= {cumsum_col:min_col}).reset_index()
    df_min = df_min[id_rolling+ [min_col]]
    df_sorted = df_sorted.merge(df_min, on=id_rolling)
    
    df_max = df_sorted[df_sorted[cumsum_col]!=0]
    df_max = df_max.replace(np.nan, 0).groupby(id_rolling).max()
    max_col = col + "_max"
    df_max = df_max.rename(columns= {col:max_col}).reset_index()
    df_max = df_max[id_rolling+ [max_col]]
    df_sorted = df_sorted.merge(df_max, on=id_rolling)

    #? REQUIRED FOR EFFECT -MAGNITUD
    df_min_diff = df_sorted[df_sorted[diff_col]!=0]
    df_min_diff = df_min_diff.replace(np.nan, 0).groupby(id_rolling).min()
    diff_min_col = col + "_diff_min"
    df_min_diff = df_min_diff.rename(columns= {diff_col:diff_min_col}).reset_index()
    df_min_diff = df_min_diff[id_rolling+ [diff_min_col]]
    df_sorted = df_sorted.merge(df_min_diff, on=id_rolling)
    
    df_max_diff = df_sorted[df_sorted[diff_col]!=0]
    df_max_diff = df_max_diff.replace(np.nan, 0).groupby(id_rolling).max()
    diff_max_col = col + "_diff_max"
    df_max_diff = df_max_diff.rename(columns= {diff_col:diff_max_col}).reset_index()
    df_max_diff = df_max_diff[id_rolling+ [diff_max_col]]
    df_sorted = df_sorted.merge(df_max_diff, on=id_rolling)
    
    #! DIRECTION 
    df_direction = df_sorted[df_sorted[col]!=0]
    df_direction = df_direction.replace(np.nan, 0).groupby(id_rolling)[cumsum_col].last().reset_index()
    dir_col = col + "_direction"
    df_direction[dir_col] =0 
    df_direction.loc[df_direction[cumsum_col]>0, dir_col] = 1
    df_direction.loc[df_direction[cumsum_col]<0, dir_col] = -1

    df_direction = df_direction[id_rolling+ [dir_col]]
    df_sorted = df_sorted.merge(df_direction, on=id_rolling)

    df_mean = df_sorted.groupby(id_rolling)[col].mean().reset_index()
    mean_col = col + "_mean"
    df_mean=df_mean.rename(columns={col:mean_col})
    df_sorted = df_sorted.merge(df_mean, on =['region', 'sku', "year"])
    
    #! MAGINITUDE
    positive_direction_bool = df_sorted[dir_col]>0
    df_sorted.loc[positive_direction_bool,"magnitude"]  = df_sorted[cumsum_max_col]/ df_sorted[mean_col]
    df_sorted.loc[~positive_direction_bool,"magnitude"]  = df_sorted[cumsum_min_col]/ df_sorted[mean_col]
    df_sorted.loc[df_sorted[dir_col]==0,"magnitude"]  = 0
    
    #! EXPERIMENT EFFECT
    max_pricechange_bool = df_sorted[diff_max_col] == df_sorted[diff_col]
    min_pricechange_bool = df_sorted[diff_min_col] == df_sorted[diff_col]
    df_sorted[dir_col]>0 
    df_sorted["effect"] =0
    df_sorted.loc[(positive_direction_bool&max_pricechange_bool), "effect"] =1 #De los positivos cuando sucede el mayor aumento de precio
    df_sorted.loc[(~positive_direction_bool&min_pricechange_bool), "effect"] =1#De los negativos cuando sucede el mayor caida de precio
    
    df_sorted = df_sorted.sort_values(id_sort)

    df_sorted["effect_strategy"] =np.nan
    df_sorted.loc[(positive_direction_bool&max_pricechange_bool), "effect_strategy"] =1 #De los positivos cuando sucede el mayor aumento de precio
    df_sorted.loc[(~positive_direction_bool&min_pricechange_bool), "effect_strategy"] =-1 #De los negativos cuando sucede el mayor caida de precio
    df_sorted["effect_strategy"] = df_sorted.groupby(id_rolling)["effect_strategy"].fillna(method='ffill').fillna(0)
    #Se etiqueta como 1 el perido a partir del cual se produce el mayor impacto de la eqtiqueta. Dependiendo de DIRECTION de la 
    # estrategia tendremos que mapear el valor para identificar AUMENTOS DE PRECIO VS ESTRATEGIA = AUMENTOS + CAIDAS

    df_sorted = df_sorted.sort_values(id_sort)

    df_sorted["effect_price_increment"] =np.nan
    df_sorted.loc[max_pricechange_bool, "effect_price_increment"] =1 #De los positivos cuando sucede el mayor aumento de precio
    df_sorted["effect_price_increment"] = df_sorted.groupby(id_rolling)["effect_price_increment"].fillna(method='ffill').fillna(0)

    return df_sorted



#########################################################################
#########################################################################
#########################################################################

if __name__ == "__main__":
    #TODO:  rename df_bitacoras_preproc and use it as df_bitacoras (until we get all client and sucursal data)
    DATAIKU = False
    PROCESS_SEMANA = False
    TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M") # Useful for saving files by date-time name.

    if DATAIKU:
        frogtek_upc_raw_prepared = dataiku.Dataset("frogtek_upc_raw_prepared")
        df_upc = frogtek_upc_raw_prepared.get_dataframe()
    else:
       #  entregas_path = "../../data/processed/preproc/entregas_preproc_" + TIMESTAMP + ".csv"
        # df_bitacoras.to_csv(entregas_path, index =False )
        upc_path = "../../data/raw/frogtek_upc_raw_prepared.csv"
        df_upc = pd.read_csv(upc_path )
        
    df_upc= df_upc.rename(columns ={"Plaza":"region", "Descripción UPC":"sku"})
    df_upc= df_upc[df_upc.region != "Total"]

    # Compute recipe outputs from inputs
    # TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
    # NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

    df_upc_group = df_upc.groupby(['month','region', 'sku']).mean().reset_index()

    id_cols=['month', 'year']
    df_upc_preciocompra =df_upc_group.pivot_table(index=id_cols, columns=['region', 'sku'],values= 'Precio Compra', aggfunc=np.mean, fill_value=0)
    df_upc_precioventa =df_upc_group.pivot_table(index=id_cols, columns=['region', 'sku'],values= 'Precio Venta', aggfunc=np.mean, fill_value=0)
    df_upc_unidades =df_upc_group.pivot_table(index=id_cols, columns=['region', 'sku'],values='Ventas Uds Promedio por Tienda', aggfunc=np.mean, fill_value=0)

    #! IMPUTE VALUES
    df_upc_preciocompra_impute, pcompra_imputed = impute_values(df_upc_preciocompra) 
    df_upc_precioventa_impute, pventa_imputed = impute_values(df_upc_precioventa)
    df_upc_unidades_impute, unidades_imputed = impute_values(df_upc_unidades)

    pc_pv = [x for x in pcompra_imputed if x not in pventa_imputed]
    pv_pc = [x for x in pventa_imputed if x not in pcompra_imputed]
    print(len(pc_pv))
    print(len(pv_pc))

    pc_u = [x for x in pcompra_imputed if x not in unidades_imputed]
    u_pc =[x for x in unidades_imputed if x not in pcompra_imputed]
    print(len(pc_u))
    print(len(u_pc))

    pv_u = [x for x in unidades_imputed if x not in pventa_imputed]
    u_pv =[x for x in pventa_imputed if x not in unidades_imputed]
    print(len(u_pv))
    print(len(pv_u))
    
    #! TRIME COLUMNS
    df_upc_preciocompra_impute, df_upc_precioventa_impute = trime_cols(df_upc_preciocompra_impute, df_upc_precioventa_impute)
    df_upc_preciocompra_impute, df_upc_unidades_impute = trime_cols(df_upc_preciocompra_impute, df_upc_unidades_impute)
    df_upc_precioventa_impute, df_upc_unidades_impute = trime_cols(df_upc_precioventa_impute, df_upc_unidades_impute)
    # Repeat again
    df_upc_preciocompra_impute, df_upc_precioventa_impute = trime_cols(df_upc_preciocompra_impute, df_upc_precioventa_impute)

    #! MELTED
    df_upc_preciocompra_melt = melt_pivoted(df_upc_preciocompra_impute, "precio_compra")
    df_upc_unidades_melt = melt_pivoted(df_upc_unidades_impute, "unidades")
    df_upc_precioventa_melt =melt_pivoted(df_upc_precioventa_impute, "precio_venta")
    
    def my_fun(x):
        return x.iloc[1]-x.iloc[0]
    
    # df_upc_precioventa_melt.set_index(id_sort).sort_index().rolling(window=2).apply(my_fun)
    
    def prepare_quantity(df_unidades:pd.DataFrame, group_cols:list, col:str):
        df_unidades_avg= df_unidades.groupby(group_cols)["unidades"].mean().reset_index()
        df_unidades_avg = df_unidades_avg.rename(columns={"unidades":"unidades_avg"})

        df_unidades_std= df_unidades.groupby(group_cols)["unidades"].std().reset_index()
        df_unidades_std = df_unidades_std.rename(columns={"unidades":"unidades_std"})
        
        df_unidades =df_unidades.merge(df_unidades_avg, how="left", on =group_cols )
        df_unidades =df_unidades.merge(df_unidades_std, how="left", on =group_cols)
        df_unidades[col] = (df_unidades["unidades"] - df_unidades["unidades_avg"])/df_unidades["unidades_std"]
        return df_unidades
    
    df_exp_pv = create_experiment("precio_venta", df_upc_precioventa_melt, id_sort =[ 'region', 'sku', "year", "month"],  id_rolling= [ 'region', 'sku', "year"])
    
    brand_dictionary = {
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'bimbo',
       'Kinder Delice Chocolate Bolsa 39 g': 'kinder',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'marinela',
       'Marinela Gansito Chocolate Bolsa 50 g': 'marinela',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'marinela',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'marinela',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'vuala',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'vuala',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'vuala',
       'Vuala Swich Chocolate Bolsa 32 g': 'vuala',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'vuala',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'vuala',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'vuala',
    }
    df_upc_unidades_melt["sku_brand"] = df_upc_unidades_melt['sku'].map(brand_dictionary)

    canibalization_dictionary = { #Canibalizacion
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'competencia_indirecta',
       'Kinder Delice Chocolate Bolsa 39 g': 'competencia_indirecta',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'competencia_indirecta',
       'Marinela Gansito Chocolate Bolsa 50 g': 'competencia_indirecta',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'competencia_indirecta',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'competencia_indirecta',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'vuala_other',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'vuala_other',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'vuala_other',
       'Vuala Swich Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
    }

    swich_competencia_dictionary = {
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'competencia_indirecta',
       'Kinder Delice Chocolate Bolsa 39 g': 'competencia_indirecta',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'competencia_indirecta',
       'Marinela Gansito Chocolate Bolsa 50 g': 'competencia_indirecta',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'competencia_directa',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'competencia_directa',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'competencia_indirecta',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'competencia_indirecta',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'competencia_indirecta',
       'Vuala Swich Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
    }
    
    swich_types_dictionary = { #Canibalizacion
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'other',
       'Kinder Delice Chocolate Bolsa 39 g': 'other',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'other',
       'Marinela Gansito Chocolate Bolsa 50 g': 'other',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'other',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'other',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'other',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'other',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'other',
       'Vuala Swich Chocolate Bolsa 32 g': 'swich',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'swich',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'roll',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'roll',
    }
        
        
    df_upc_unidades_melt["sku_brand"] = df_upc_unidades_melt['sku'].map(brand_dictionary)
    df_upc_unidades_melt["sku_swich"] = df_upc_unidades_melt['sku'].map(canibalization_dictionary)
    df_upc_unidades_melt["sku_competencia"] = df_upc_unidades_melt['sku'].map(swich_competencia_dictionary)
    df_upc_unidades_melt["sku_swich_types"] = df_upc_unidades_melt['sku'].map(swich_types_dictionary)

    id_cols = [ 'region', 'sku', "year", "month"]
    
    df = df_exp_pv.merge(df_upc_unidades_melt[id_cols + ["sku_brand", "sku_swich", 'sku_competencia']], on =id_cols)
    df = df.merge(df_upc_unidades_melt[id_cols + ["unidades"]], on =id_cols)
    df_unidades = prepare_quantity(df_upc_unidades_melt, group_cols = ['sku'], col = "unidades_sku_std")
    df = df.merge(df_unidades[id_cols + ["unidades_sku_std"]], on =id_cols)

    df_unidades = prepare_quantity(df_upc_unidades_melt, group_cols = ["sku_brand" ], col = "unidades_brand_std")
    df = df.merge(df_unidades[id_cols + ["unidades_brand_std"]], on =id_cols)

    df_unidades = prepare_quantity(df_upc_unidades_melt, group_cols = ["sku_swich" ], col = "unidades_swich_std")
    df = df.merge(df_unidades[id_cols + ["unidades_swich_std"]], on =id_cols)

    df_unidades = prepare_quantity(df_upc_unidades_melt, group_cols = ["sku_competencia" ], col = "unidades_competencia_std")
    df = df.merge(df_unidades[id_cols + ["unidades_competencia_std"]], on =id_cols)
    
    df["unidades_std"] =  (df["unidades"] - df["unidades"].mean())/df["unidades"].std()
    df = df.dropna()
    df.to_csv("../../data/processed/upc_proc.csv")